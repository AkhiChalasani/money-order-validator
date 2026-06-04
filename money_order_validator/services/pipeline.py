from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from money_order_validator.clients.azure_document_intelligence import OcrPage, adi_reader
from money_order_validator.clients.azure_openai import llm_client
from money_order_validator.schemas import BatchContext, Instrument, TokenUsage, ValidationResult
from money_order_validator.services.extraction import vision_extractor
from money_order_validator.services.page_classifier import PageKind, classify_page
from money_order_validator.services.regex_parsers import (
    build_property_aliases,
    normalize_bank_name,
    normalize_payee,
    normalize_serial,
    parse_batch_header,
    parse_batch_line_items,
    parse_deposit_info,
    parse_money,
    sanitize_instrument,
)
from money_order_validator.services.renderer import pdf_renderer
from money_order_validator.services.validation import compute_ocr_confidence, validate_instruments
from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


def _merge_non_empty(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (patch or {}).items():
        if value not in (None, "", [], {}):
            base[key] = value
    return base


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return parse_money(value)


def _clean_account(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    nums = re.findall(r"\d{4,}", str(value))
    return nums[-1] if nums else None


def _should_try_unknown_vision(text: str) -> bool:
    if not settings.vision_on_unknown_pages:
        return False
    if not text.strip():
        return True
    # Bank/register reports are intentionally not routed to instrument vision.
    if re.search(r"DETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT|TOTAL\s+OF\s+DEPOSITS\s+SUBMITTED|CAPTURE\s+SEQ", text, re.IGNORECASE):
        return False
    return bool(re.search(r"\b(PAY|ORDER|MONEY|CHECK|CASHIER|WESTERN|MONEYGRAM|INTERMEX|BARRI|FIDELITY|PLS)\b|\$\s*\d", text, re.IGNORECASE))


def _serials_match(left: Any, right: Any) -> bool:
    a = normalize_serial(left)
    b = normalize_serial(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # Check numbers in bank reports are often zero-padded while instruments are not.
    aa = re.sub(r"\D", "", a).lstrip("0")
    bb = re.sub(r"\D", "", b).lstrip("0")
    if aa and bb and aa == bb:
        return True
    # For MICR-derived strings, the check/serial number may be the suffix.
    return bool(aa and bb and (aa.endswith(bb) or bb.endswith(aa)) and min(len(aa), len(bb)) >= 4)


def _page_has_deposit_ticket(text: str, deposit_patch: Optional[Dict[str, Any]] = None) -> bool:
    if deposit_patch and any(
        deposit_patch.get(k)
        for k in (
            "deposit_amount",
            "deposit_total",
            "check_total",
            "item_count",
            "deposit_account",
            "account_name",
            "account_last4",
        )
    ):
        return True
    t = (text or "").upper()
    return bool(
        re.search(
            r"DEPOSIT\s+TICKET|TOTAL\s+ITEMS|PLEASE\s+BE\s+SURE|CHECKING\s+DEPOSIT|"
            r"TRANSACTION\s+SUMMARY|ACCOUNT\s+NUMBER\s+ENDING\s+IN|"
            r"CHECKS\s+AND\s+OTHER\s+ITEMS\s+ARE\s+RECEIVED\s+FOR\s+DEPOSIT|"
            r"FOR\s+CASH\s+DEPOSIT",
            t,
        )
    )


def _looks_like_back_page(text: str) -> bool:
    """Detect money-order/check backs even when OCR includes front-side bleed-through."""
    t = (text or "").upper()
    signals = [
        r"\bLOAD\s+THIS\s+DIRECTION\b",
        r"\bPURCHASER'?S\s+AGREEMENT\b",
        r"\bSERVICE\s+CHARGE\b",
        r"\bTERMS\s+AND\s+CONDITIONS\b",
        r"\bPAYEE\s+ENDORSEMENT\b",
        r"\bENDORSE\s+ABOVE\s+THIS\s+LINE\b",
        r"\bDEPOSITORY\s+BANK\s+ENDORSEMENT\b",
        r"\bDO\s+NOT\s+WRITE\s*/?\s*SIGN\s*/?\s*STAMP\s+BELOW\b",
        r"\bFOR\s+DEPOSIT\s+ONLY\b",
        r"\bWARNING\s+DO\s+NOT\s+CASH\b",
        r"\bNOT\s+CASH\s+UNLESS\s+THE\s+MACHINE\s+PRINTED\b",
    ]
    hits = sum(1 for pat in signals if re.search(pat, t, flags=re.IGNORECASE))
    strong_front_face = bool(
        re.search(r"\bPAY\s+EXACTLY\b|\bPAY\s+ONLY\b", t)
        and re.search(r"\bPAY\s+TO\s+THE\s+ORDER\b|\bPAY\s+TO\b", t)
        and re.search(r"\$\s*\d{2,5}[,.]\d{2}\b", t)
    )
    if hits >= 2 and not strong_front_face:
        return True
    if re.search(r"\bFOR\s+DEPOSIT\s+ONLY\b", t) and re.search(r"ENDORSE|DEPOSITORY\s+BANK|WARNING\s+DO\s+NOT\s+CASH", t):
        return True
    return False


def _is_back_page_artifact(inst: Dict[str, Any], ocr_text: str) -> bool:
    """Drop LLM rows hallucinated from the back of a money order/check."""
    if not _looks_like_back_page(ocr_text):
        return False
    has_legal_amount = bool(inst.get("amount_words"))
    has_payee = bool(normalize_payee(inst.get("payee_raw")))
    has_date = bool(inst.get("issue_date"))
    amount = _as_float(inst.get("amount_numeric"))
    if not has_legal_amount or not has_payee or not has_date:
        return True
    if amount is not None and amount < 20 and not has_legal_amount:
        return True
    return False


def _deposit_amount_value(row: Dict[str, Any]) -> Optional[float]:
    return _as_float(row.get("deposit_amount") or row.get("deposit_total") or row.get("check_total"))


def _deposit_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    amount = _deposit_amount_value(row)
    tx = row.get("deposit_transaction")
    acct = _clean_account(row.get("deposit_account") or row.get("account_last4"))
    date_value = row.get("deposit_date")
    item_count = row.get("item_count")
    if tx or acct or date_value:
        return (tx or "", acct or "", date_value or "", amount, item_count or "")
    return ("page", row.get("page_number"), amount, item_count or "")


def _is_meaningful_deposit(row: Dict[str, Any]) -> bool:
    if not row:
        return False
    return bool(_deposit_amount_value(row) is not None or row.get("item_count") or row.get("deposit_account") or row.get("deposit_transaction"))


def _add_deposit_slip(slips: List[Dict[str, Any]], patch: Dict[str, Any], page_number: int) -> None:
    if not _is_meaningful_deposit(patch):
        return
    row = dict(patch)
    row.setdefault("page_number", page_number)
    key = _deposit_key(row)
    for existing in slips:
        if _deposit_key(existing) == key:
            _merge_non_empty(existing, row)
            return
    slips.append(row)


def _aggregate_deposit_slips(slips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a single legacy deposit_slip object from one or more physical slips/receipts."""
    if not slips:
        return {}
    out: Dict[str, Any] = {}
    for row in slips:
        for key in ("bank_name", "source_system", "deposit_account", "account_last4", "account_name", "deposit_date"):
            if not out.get(key) and row.get(key) not in (None, "", [], {}):
                out[key] = row[key]
    amounts = [_deposit_amount_value(row) for row in slips if _deposit_amount_value(row) is not None]
    unique_amount_slips = [row for row in slips if _deposit_amount_value(row) is not None]
    if amounts:
        total = round(sum(amounts), 2) if len(unique_amount_slips) > 1 else amounts[0]
        out["deposit_amount"] = total
        out["deposit_total"] = total
        out["check_total"] = total
    item_counts = []
    for row in slips:
        try:
            if row.get("item_count") is not None:
                item_counts.append(int(row["item_count"]))
        except (TypeError, ValueError):
            pass
    if item_counts:
        out["item_count"] = sum(item_counts) if len(item_counts) > 1 else item_counts[0]
    txs = [str(row.get("deposit_transaction")) for row in slips if row.get("deposit_transaction")]
    if txs:
        out["deposit_transaction"] = ",".join(dict.fromkeys(txs))
    out["deposit_slip_count"] = len(slips)
    out["deposit_slips"] = slips
    return out


def _bad_property_candidate(value: Optional[str]) -> bool:
    if not value:
        return True
    compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return compact in {"CENTS", "DOLLARS", "DOLLAR", "CURRENCY", "COIN", "TOTAL", "CHASE", "JPMORGANCHASEBANK", "REGIONS"}


def _infer_property_from_instruments(instruments: List[Dict[str, Any]]) -> Optional[str]:
    counts: Counter[str] = Counter()
    for row in instruments:
        payee = normalize_payee(row.get("payee_raw"))
        if not payee or _bad_property_candidate(payee):
            continue
        if re.search(r"\b(BANK|CHASE|REGIONS|WESTERN\s+UNION|MONEYGRAM|INTERMEX|FIDELITY)\b", payee, flags=re.IGNORECASE):
            continue
        counts[payee.title()] += 1
    return counts.most_common(1)[0][0] if counts else None


def _is_deposit_ticket_artifact(
    inst: Dict[str, Any],
    *,
    ocr_text: str,
    page_deposit: Optional[Dict[str, Any]],
    page_instruments: List[Dict[str, Any]],
) -> bool:
    """Suppress deposit-ticket pseudo-instruments on mixed pages.

    Vision models sometimes treat the pre-printed deposit ticket at the top of a page as a
    check, borrowing the amount from a real money order below it. The pseudo-instrument
    usually has no date, no written amount, no payer signature, and a business entity
    rather than a real drawer.
    """
    if not _page_has_deposit_ticket(ocr_text, page_deposit):
        return False
    inst_type = str(inst.get("instrument_type") or "").lower()
    no_face_evidence = not inst.get("issue_date") and not inst.get("amount_words") and not inst.get("payer_signature")
    # Pure deposit tickets are sometimes returned as a MoneyOrder/Chase row because the MICR line
    # contains numeric groups. Drop any no-evidence row on a deposit-ticket page, not just checks.
    if no_face_evidence and not normalize_payee(inst.get("payee_raw")) and _as_float(inst.get("amount_numeric")) is None:
        return True
    if inst_type not in {"check", "cashierscheck"}:
        return False
    if not no_face_evidence:
        return False

    amount = _as_float(inst.get("amount_numeric"))
    same_amount_elsewhere = False
    if amount is not None:
        for other in page_instruments:
            if other is inst:
                continue
            other_amount = _as_float(other.get("amount_numeric"))
            if other_amount is not None and abs(other_amount - amount) < 0.01 and (
                other.get("amount_words") or other.get("instrument_type") == "MoneyOrder"
            ):
                same_amount_elsewhere = True
                break

    serial = normalize_serial(inst.get("serial_number")) or ""
    business_text = " ".join(
        str(inst.get(k) or "") for k in ("payer_name", "payer_address", "payee_raw", "micr_line")
    ).upper()
    looks_like_deposit_entity = bool(
        re.search(r"LIMITED\s+PARTNERSHIP|OPERATING\s+ACCOUNT|DBA\b|SUITE\b|DEPOSIT\s+TICKET", business_text)
    )
    weak_serial = bool(serial and len(re.sub(r"\D", "", serial)) <= 8)
    return same_amount_elsewhere or looks_like_deposit_entity or weak_serial


def _item_match_score(inst: Dict[str, Any], item: Dict[str, Any]) -> int:
    score = 0
    if _serials_match(inst.get("serial_number"), item.get("serial_number")) or _serials_match(inst.get("micr_line"), item.get("serial_number")):
        score += 100
    ai = _as_float(inst.get("amount_numeric"))
    aj = _as_float(item.get("amount_numeric"))
    if ai is not None and aj is not None:
        diff = abs(ai - aj)
        if diff < 0.01:
            score += 25
        elif diff <= 5.00:
            # Allows register reconciliation to correct common cents/OCR errors.
            score += 10
    if inst.get("unit") and item.get("unit") and str(inst.get("unit")) == str(item.get("unit")):
        score += 10
    return score


class DocumentProcessor:
    async def process_batch(self, file_payloads: List[Tuple[str, bytes]]) -> List[ValidationResult]:
        if len(file_payloads) > settings.max_files_per_batch:
            raise ValueError(f"Too many files. Maximum is {settings.max_files_per_batch}.")
        tasks = [self.process_file(file_name, content) for file_name, content in file_payloads]
        return await asyncio.gather(*tasks)

    async def process_file(self, file_name: str, content: bytes) -> ValidationResult:
        if not file_name.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files are supported: {file_name}")
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File exceeds {settings.max_file_size_mb} MB: {file_name}")

        logger.info("Processing %s (%d bytes)", file_name, len(content))
        render_task = asyncio.create_task(pdf_renderer.render(content))
        ocr_task = asyncio.create_task(adi_reader.analyze_pdf(content))
        images, ocr_pages = await asyncio.gather(render_task, ocr_task)
        if not images:
            raise ValueError(f"PDF rendered zero pages: {file_name}")

        ocr_texts = self._ocr_texts_by_page(ocr_pages, len(images))
        raw_instruments: List[Dict[str, Any]] = []
        batch_data: Dict[str, Any] = {}
        deposit_data: Dict[str, Any] = {}
        deposit_slips: List[Dict[str, Any]] = []
        register_items: List[Dict[str, Any]] = []
        page_logs: List[Dict[str, Any]] = []
        usage_total = TokenUsage()
        llm_calls = 0
        phase_tokens: Dict[str, int] = defaultdict(int)

        for page_idx, image in enumerate(images):
            page_number = page_idx + 1
            ocr_text = ocr_texts[page_idx]
            kind, scores = classify_page(ocr_text)
            # If Azure DI is unavailable or returned no text for an image-only page,
            # do not treat it as blank when an LLM vision client is configured.
            if not ocr_text.strip() and llm_client.available:
                kind = PageKind.UNKNOWN
                scores = {**scores, "no_ocr_vision_fallback": 1}
            log_row: Dict[str, Any] = {
                "page_number": page_number,
                "kind": kind.value,
                "scores": scores,
                "ocr_chars": len(ocr_text),
                "llm_used": False,
                "instruments": 0,
            }

            # Text-first parsing. Deposit receipts/tickets can appear on receipt pages or on
            # mixed pages above instruments, so parse deposit metadata on every OCR page. Batch
            # header/property parsing is still restricted to report-like pages to avoid treating a
            # check drawer as the property name.
            page_deposit = parse_deposit_info(ocr_text)
            if page_deposit:
                _merge_non_empty(deposit_data, page_deposit)
                _add_deposit_slip(deposit_slips, page_deposit, page_number)
                log_row["deposit_detected"] = True
            if kind in {PageKind.BATCH_HEADER, PageKind.DEPOSIT_REPORT, PageKind.REPORT_WITH_INSTRUMENTS, PageKind.DEPOSIT_SLIP, PageKind.UNKNOWN, PageKind.RECEIPT}:
                _merge_non_empty(batch_data, parse_batch_header(ocr_text))
                new_items = parse_batch_line_items(ocr_text)
                if new_items:
                    register_items.extend(new_items)

            # Receipts and mixed deposit-ticket+instrument pages may carry the authoritative
            # deposit total. This is page-scoped, not global: a single PDF can contain multiple
            # physical deposit slips/receipts and we need each one's total separately.
            needs_deposit_total = not (page_deposit.get("deposit_amount") or page_deposit.get("deposit_total") or page_deposit.get("check_total"))
            if needs_deposit_total and (
                kind in {PageKind.RECEIPT, PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP, PageKind.REPORT_WITH_INSTRUMENTS}
                or _page_has_deposit_ticket(ocr_text, page_deposit)
            ):
                patch, usage, used = await vision_extractor.extract_deposit(image, ocr_text)
                _merge_non_empty(deposit_data, patch)
                page_deposit = {**page_deposit, **(patch or {})}
                _add_deposit_slip(deposit_slips, page_deposit, page_number)
                if used:
                    llm_calls += 1
                    self._accumulate_usage(usage_total, usage)
                    phase_tokens["deposit"] += usage.total_tokens
                    log_row["llm_used"] = True
                    log_row["deposit_vision_used"] = True

            if kind in {PageKind.BACK, PageKind.BLANK, PageKind.RECEIPT}:
                page_logs.append(log_row)
                continue

            if kind in {PageKind.BATCH_HEADER, PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP}:
                # Ask vision only when OCR parsing did not produce enough info.
                if kind == PageKind.BATCH_HEADER and not (batch_data.get("batch_amount") or batch_data.get("total_items")):
                    patch, usage, used = await vision_extractor.extract_batch_header(image, ocr_text)
                    _merge_non_empty(batch_data, patch)
                    if used:
                        llm_calls += 1
                        self._accumulate_usage(usage_total, usage)
                        phase_tokens["batch_header"] += usage.total_tokens
                        log_row["llm_used"] = True
                elif kind in {PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP} and not (page_deposit.get("deposit_amount") or page_deposit.get("deposit_total") or page_deposit.get("check_total")):
                    patch, usage, used = await vision_extractor.extract_deposit(image, ocr_text)
                    _merge_non_empty(deposit_data, patch)
                    page_deposit = {**page_deposit, **(patch or {})}
                    _add_deposit_slip(deposit_slips, page_deposit, page_number)
                    if used:
                        llm_calls += 1
                        self._accumulate_usage(usage_total, usage)
                        phase_tokens["deposit"] += usage.total_tokens
                        log_row["llm_used"] = True
                page_logs.append(log_row)
                continue

            should_extract = kind in {PageKind.INSTRUMENT, PageKind.REPORT_WITH_INSTRUMENTS}
            if kind == PageKind.UNKNOWN:
                should_extract = _should_try_unknown_vision(ocr_text)
            if should_extract:
                instruments, usage, used = await vision_extractor.extract_instruments(image, ocr_text, kind)
                if used:
                    llm_calls += 1
                    self._accumulate_usage(usage_total, usage)
                    phase_tokens["instrument"] += usage.total_tokens
                    log_row["llm_used"] = True
                kept = 0
                sanitized_page: List[Dict[str, Any]] = [sanitize_instrument(inst, ocr_text=ocr_text) for inst in instruments]
                for inst in sanitized_page:
                    if _is_back_page_artifact(inst, ocr_text):
                        logger.info("Dropped back-page artifact on page %d: serial=%s amount=%s", page_number, inst.get("serial_number"), inst.get("amount_numeric"))
                        continue
                    if _is_deposit_ticket_artifact(
                        inst,
                        ocr_text=ocr_text,
                        page_deposit=page_deposit,
                        page_instruments=sanitized_page,
                    ):
                        logger.info("Dropped deposit-ticket artifact on page %d: serial=%s amount=%s", page_number, inst.get("serial_number"), inst.get("amount_numeric"))
                        continue
                    inst["page_number"] = page_number
                    inst["source_file"] = file_name
                    inst["llm_used"] = bool(used)
                    inst["processing_tier"] = 3 if used else 1
                    raw_instruments.append(inst)
                    kept += 1
                log_row["instruments"] = kept
                if kept != len(instruments):
                    log_row["deposit_artifacts_dropped"] = len(instruments) - kept

            page_logs.append(log_row)

        register_items = self._dedupe_register_items(register_items)
        raw_instruments = self._dedupe_raw_instruments(raw_instruments)
        raw_instruments = self._reconcile_register_items(raw_instruments, register_items)
        aggregate_deposit = _aggregate_deposit_slips(deposit_slips)
        if aggregate_deposit:
            deposit_data = {**deposit_data, **aggregate_deposit}
        batch = self._build_batch(batch_data, deposit_data, register_items, raw_instruments)
        instruments = self._build_instruments(batch, raw_instruments, file_name)
        validate_instruments(batch, instruments)
        batch.gl_summary = self._build_gl_summary(batch, instruments)
        batch.processing_stats = self._build_stats(
            total_pages=len(images),
            ocr_pages=sum(1 for t in ocr_texts if t.strip()),
            page_logs=page_logs,
            usage_total=usage_total,
            llm_calls=llm_calls,
            phase_tokens=phase_tokens,
            register_items=register_items,
            raw_instruments=raw_instruments,
        )
        batch.processing_stats["deposit_slips_extracted"] = len(deposit_slips)

        return ValidationResult(
            file_name=file_name,
            batch=batch,
            instruments=instruments,
            deposit_slip=deposit_data or None,
            deposit_slips=deposit_slips or None,
            pages=page_logs if settings.return_debug_pages else None,
        )

    @staticmethod
    def _ocr_texts_by_page(pages: List[OcrPage], expected_pages: int) -> List[str]:
        out = [""] * expected_pages
        for p in pages:
            idx = p.page_number - 1
            if 0 <= idx < expected_pages:
                out[idx] = p.text or ""
        return out

    @staticmethod
    def _accumulate_usage(total: TokenUsage, usage: TokenUsage) -> None:
        total.prompt_tokens += usage.prompt_tokens
        total.completion_tokens += usage.completion_tokens
        total.total_tokens += usage.total_tokens

    @staticmethod
    def _dedupe_raw_instruments(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for row in rows:
            serial = normalize_serial(row.get("serial_number")) or ""
            amount = _as_float(row.get("amount_numeric"))
            page = row.get("page_number")
            payee = normalize_payee(row.get("payee_raw")) or ""
            key = (page, serial, amount, payee)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _dedupe_register_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for row in rows:
            key = (
                row.get("source"),
                normalize_serial(row.get("serial_number")) or normalize_serial(row.get("check_number")) or "",
                _as_float(row.get("amount_numeric")),
                row.get("item_no"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _reconcile_register_items(self, instruments: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched_items = set()
        for inst in instruments:
            best_idx = None
            best_score = 0
            for idx, item in enumerate(items):
                if idx in matched_items:
                    continue
                score = _item_match_score(inst, item)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None and best_score >= 100:
                item = items[best_idx]
                matched_items.add(best_idx)
                # Preserve authoritative register ordering and metadata.
                if item.get("item_no") is not None:
                    inst["item_no"] = item["item_no"]
                    inst["register_item_no"] = item["item_no"]
                inst["register_match_score"] = best_score
                inst["register_source"] = item.get("source")
                for key in ("unit", "resident_name", "posted_date", "posted_by", "payment_description"):
                    if not inst.get(key) and item.get(key):
                        inst[key] = item[key]
                if item.get("amount_numeric") is not None:
                    old_amount = _as_float(inst.get("amount_numeric"))
                    new_amount = _as_float(item.get("amount_numeric"))
                    # Bank/register rows are authoritative after a serial/MICR match. This fixes
                    # cents drops such as 1323.00 -> 1323.74 and 1878.99 -> 1878.98.
                    if new_amount is not None and (old_amount is None or abs(old_amount - new_amount) <= 5.00 or best_score >= 100):
                        inst["amount_numeric"] = new_amount
                        inst["amount_numeric_source"] = item.get("source") or "register"
                        if old_amount is not None and abs(old_amount - new_amount) >= 0.01:
                            inst.setdefault("corrections", []).append(
                                {
                                    "field": "amount_numeric",
                                    "old": old_amount,
                                    "new": new_amount,
                                    "reason": "matched bank/register row",
                                }
                            )
                if not inst.get("serial_number") and item.get("serial_number"):
                    inst["serial_number"] = item["serial_number"]

        if settings.include_register_only_items:
            for idx, item in enumerate(items):
                if idx in matched_items:
                    continue
                # Add only credible register items. These help expose missing scans.
                if not (item.get("serial_number") or item.get("amount_numeric")):
                    continue
                inst_type = "MoneyOrder" if "Money" in (item.get("payment_description") or "") else "Check"
                instruments.append(
                    {
                        **item,
                        "instrument_type": inst_type,
                        "llm_used": False,
                        "processing_tier": 1,
                        "missing_from_scan": True,
                    }
                )
        return instruments

    @staticmethod
    def _build_batch(
        batch_data: Dict[str, Any],
        deposit_data: Dict[str, Any],
        register_items: List[Dict[str, Any]],
        instruments: List[Dict[str, Any]],
    ) -> BatchContext:
        # Normalize numbers/dates.
        batch_number = batch_data.get("batch_number")
        if batch_number:
            batch_number = re.sub(r"\D", "", str(batch_number)) or None
        bank_name = normalize_bank_name(batch_data.get("bank_name") or deposit_data.get("bank_name"))
        account_number = _clean_account(batch_data.get("account_number") or deposit_data.get("deposit_account"))

        amount = _as_float(batch_data.get("batch_amount"))
        deposit_amount = _as_float(deposit_data.get("deposit_amount") or deposit_data.get("deposit_total") or deposit_data.get("check_total"))
        instrument_sum = round(sum(_as_float(i.get("amount_numeric")) or 0.0 for i in instruments), 2) if instruments else None
        deposit_count = int(deposit_data.get("deposit_slip_count") or 0)
        if amount is None:
            amount = deposit_amount
        if amount is None and register_items:
            amount = round(sum(_as_float(i.get("amount_numeric")) or 0.0 for i in register_items), 2)
        if amount is None and instrument_sum is not None:
            amount = instrument_sum
        # When multiple physical deposit slips exist but only one total was captured, the
        # single-slip total may be much smaller than the actual batch. Prefer the reconciled
        # instrument sum in that case to avoid under-reporting the batch amount.
        if (
            amount is not None
            and instrument_sum is not None
            and len(instruments) >= 25
            and not register_items
            and deposit_count <= 1
            and abs(float(amount) - float(instrument_sum)) / max(float(amount), float(instrument_sum), 1.0) > 0.15
        ):
            amount = instrument_sum

        total_items = batch_data.get("total_items") or deposit_data.get("item_count")
        try:
            total_items = int(total_items) if total_items is not None else None
        except (TypeError, ValueError):
            total_items = None
        if total_items is None:
            total_items = len(register_items) or len([i for i in instruments if not i.get("missing_from_scan")]) or None

        property_name = batch_data.get("property_name") or deposit_data.get("account_name")
        if _bad_property_candidate(property_name):
            property_name = _infer_property_from_instruments(instruments) or None
        property_aliases = batch_data.get("property_aliases") or build_property_aliases(property_name)

        return BatchContext(
            batch_id=str(uuid.uuid4()),
            batch_number=batch_number,
            batch_type=batch_data.get("batch_type") or "Check/MO",
            batch_status=batch_data.get("batch_status"),
            pay_period=batch_data.get("pay_period"),
            bank_name=bank_name,
            account_number=account_number,
            property_name=property_name,
            property_aliases=property_aliases,
            property_address=batch_data.get("property_address"),
            deposited_date=batch_data.get("deposited_date") or deposit_data.get("deposit_date") or batch_data.get("printed_on") or date.today().isoformat(),
            deposit_transaction=batch_data.get("deposit_transaction") or deposit_data.get("deposit_transaction"),
            total_items=total_items,
            batch_amount=amount,
            printed_on=batch_data.get("printed_on"),
            source_system=batch_data.get("source_system") or deposit_data.get("source_system") or "YottaReal",
        )

    @staticmethod
    def _build_instruments(batch: BatchContext, raw_rows: List[Dict[str, Any]], file_name: str) -> List[Instrument]:
        instruments: List[Instrument] = []

        def sort_key(pair: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
            pos, row = pair
            try:
                item_no = int(row.get("item_no"))
                if item_no > 0:
                    return (0, item_no)
            except (TypeError, ValueError):
                pass
            return (1, pos)

        ordered_rows = [row for _, row in sorted(enumerate(raw_rows, start=1), key=sort_key)]
        used_item_nos = set()
        for idx, raw in enumerate(ordered_rows, start=1):
            raw = sanitize_instrument(raw, ocr_text="")
            amount = _as_float(raw.get("amount_numeric"))
            payee = normalize_payee(raw.get("payee_raw"))
            batch_no = batch.batch_number
            try:
                item_no = int(raw.get("item_no"))
                if item_no <= 0 or item_no in used_item_nos:
                    item_no = idx
            except (TypeError, ValueError):
                item_no = idx
            used_item_nos.add(item_no)
            instrument_id = f"INS-{batch_no or 'UNKNOWN'}-{item_no:03d}"
            inst_type = raw.get("instrument_type") or "MoneyOrder"
            payment_description = raw.get("payment_description")
            if not payment_description:
                payment_description = "Payment-Check" if inst_type in {"Check", "CashiersCheck"} else "Payment-MoneyOrder"
            inst = Instrument(
                item_no=item_no,
                instrument_id=instrument_id,
                batch_number=batch_no,
                unit=raw.get("unit"),
                resident_name=raw.get("resident_name") or raw.get("payer_name"),
                instrument_type=inst_type,
                payment_description=payment_description,
                issuer=raw.get("issuer"),
                issuer_agent=raw.get("issuer_agent"),
                serial_number=normalize_serial(raw.get("serial_number")),
                micr_line=raw.get("micr_line"),
                issue_date=raw.get("issue_date"),
                amount_numeric=amount,
                amount_words=raw.get("amount_words"),
                payee_raw=payee,
                payee_normalized=payee.title() if isinstance(payee, str) else payee,
                payer_name=raw.get("payer_name"),
                payer_address=raw.get("payer_address"),
                payer_signature=bool(raw.get("payer_signature")),
                payment_for_acct=raw.get("payment_for_acct"),
                mobile_deposit_prohibited=bool(raw.get("mobile_deposit_prohibited")),
                watermark_present=bool(raw.get("watermark_present")),
                posted_by=raw.get("posted_by"),
                posted_date=raw.get("posted_date"),
                ocr_confidence=compute_ocr_confidence(raw),
                processing_tier=int(raw.get("processing_tier") or 3),
                llm_used=bool(raw.get("llm_used")),
                missing_from_scan=bool(raw.get("missing_from_scan")),
                page_number=raw.get("page_number"),
                source_file=file_name,
            )
            instruments.append(inst)
        return instruments

    @staticmethod
    def _build_gl_summary(batch: BatchContext, instruments: List[Instrument]) -> List[Dict[str, Any]]:
        groups: Dict[str, float] = defaultdict(float)
        for inst in instruments:
            # GL should reconcile to the authoritative bank/register total. Missing physical scans
            # remain risk exceptions, but their register amounts still belong in the deposit batch.
            groups[inst.payment_description or "Payment-MoneyOrder"] += inst.amount_numeric or 0.0
        acct_last4 = (batch.account_number or "")[-4:]
        property_name = (batch.property_name or "UNKNOWN PROPERTY").upper()
        code_map = {
            "Payment-Check": ("2001", "1010-20"),
            "Payment-MoneyOrder": ("2003", "1010-20"),
            "Escrow Deposit Paid In": ("3001", "1010-20"),
        }
        out = []
        for desc in sorted(groups.keys(), key=lambda d: code_map.get(d, ("9999", ""))[0]):
            code, gl_account = code_map.get(desc, ("9999", "1010-20"))
            out.append(
                {
                    "code": code,
                    "description": desc,
                    "gl_account": gl_account,
                    "gl_account_name": f"Operating Bank Account ({property_name})" + (f" #{acct_last4}" if acct_last4 else ""),
                    "debit": round(groups[desc], 2),
                    "credit": 0.0,
                }
            )
        return out

    @staticmethod
    def _build_stats(
        *,
        total_pages: int,
        ocr_pages: int,
        page_logs: List[Dict[str, Any]],
        usage_total: TokenUsage,
        llm_calls: int,
        phase_tokens: Dict[str, int],
        register_items: List[Dict[str, Any]],
        raw_instruments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kinds = Counter(p["kind"] for p in page_logs)
        vision_pages = sum(1 for p in page_logs if p.get("llm_used"))
        return {
            "total_pages": total_pages,
            "ocr_pages": ocr_pages,
            "page_kinds": dict(kinds),
            "vision_pages": vision_pages,
            "skipped_pages": sum(kinds.get(k.value, 0) for k in (PageKind.BACK, PageKind.BLANK, PageKind.RECEIPT, PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP, PageKind.BATCH_HEADER)),
            "register_items_extracted": len(register_items),
            "instruments_extracted": len(raw_instruments),
            "llm_calls_total": llm_calls,
            "prompt_tokens": usage_total.prompt_tokens,
            "completion_tokens": usage_total.completion_tokens,
            "total_tokens": usage_total.total_tokens,
            "tokens_by_phase": dict(phase_tokens),
            "llm_provider": llm_client.mode,
            "llm_model_or_deployment": llm_client.model,
            "azure_document_intelligence_used": bool(ocr_pages),
            "token_saving_strategy": "ADI page routing + OCR context compression + skip back/receipt/deposit pages",
        }


document_processor = DocumentProcessor()
