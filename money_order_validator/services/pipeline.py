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
    return bool(re.search(r"\b(PAY|ORDER|MONEY|CHECK|CASHIER|WESTERN|MONEYGRAM|INTERMEX|BARRI|FIDELITY|PLS)\b|\$\s*\d", text, re.IGNORECASE))


def _item_match_score(inst: Dict[str, Any], item: Dict[str, Any]) -> int:
    score = 0
    si = normalize_serial(inst.get("serial_number"))
    sj = normalize_serial(item.get("serial_number"))
    if si and sj and si == sj:
        score += 100
    ai = _as_float(inst.get("amount_numeric"))
    aj = _as_float(item.get("amount_numeric"))
    if ai is not None and aj is not None and abs(ai - aj) < 0.01:
        score += 20
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

            # Text-first parsing. These are free and help reconcile instruments later.
            if kind in {PageKind.BATCH_HEADER, PageKind.DEPOSIT_REPORT, PageKind.REPORT_WITH_INSTRUMENTS, PageKind.DEPOSIT_SLIP, PageKind.UNKNOWN}:
                _merge_non_empty(batch_data, parse_batch_header(ocr_text))
                _merge_non_empty(deposit_data, parse_deposit_info(ocr_text))
                new_items = parse_batch_line_items(ocr_text)
                if new_items:
                    register_items.extend(new_items)

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
                elif kind in {PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP} and not (deposit_data.get("deposit_amount") or deposit_data.get("deposit_total") or deposit_data.get("check_total")):
                    patch, usage, used = await vision_extractor.extract_deposit(image, ocr_text)
                    _merge_non_empty(deposit_data, patch)
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
                for inst in instruments:
                    inst = sanitize_instrument(inst, ocr_text=ocr_text)
                    inst["page_number"] = page_number
                    inst["source_file"] = file_name
                    inst["llm_used"] = bool(used)
                    inst["processing_tier"] = 3 if used else 1
                    raw_instruments.append(inst)
                log_row["instruments"] = len(instruments)

            page_logs.append(log_row)

        raw_instruments = self._dedupe_raw_instruments(raw_instruments)
        raw_instruments = self._reconcile_register_items(raw_instruments, register_items)
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

        return ValidationResult(
            file_name=file_name,
            batch=batch,
            instruments=instruments,
            deposit_slip=deposit_data or None,
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

    def _reconcile_register_items(self, instruments: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched_items = set()
        for inst in instruments:
            best_idx = None
            best_score = 0
            for idx, item in enumerate(items):
                score = _item_match_score(inst, item)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None and best_score >= 100:
                item = items[best_idx]
                matched_items.add(best_idx)
                for key in ("unit", "resident_name", "posted_date", "posted_by", "payment_description"):
                    if not inst.get(key) and item.get(key):
                        inst[key] = item[key]
                if inst.get("amount_numeric") is None and item.get("amount_numeric") is not None:
                    inst["amount_numeric"] = item["amount_numeric"]
                if not inst.get("serial_number") and item.get("serial_number"):
                    inst["serial_number"] = item["serial_number"]

        if settings.include_register_only_items:
            for idx, item in enumerate(items):
                if idx in matched_items:
                    continue
                # Add only credible register items. These help expose missing scans.
                if not (item.get("serial_number") or item.get("amount_numeric")):
                    continue
                instruments.append(
                    {
                        **item,
                        "instrument_type": "MoneyOrder" if "Money" in (item.get("payment_description") or "") else "Check",
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
        if amount is None:
            amount = _as_float(deposit_data.get("deposit_amount") or deposit_data.get("deposit_total") or deposit_data.get("check_total"))
        if amount is None and register_items:
            amount = round(sum(_as_float(i.get("amount_numeric")) or 0.0 for i in register_items), 2)
        if amount is None and instruments:
            amount = round(sum(_as_float(i.get("amount_numeric")) or 0.0 for i in instruments), 2)

        total_items = batch_data.get("total_items")
        try:
            total_items = int(total_items) if total_items is not None else None
        except (TypeError, ValueError):
            total_items = None
        if total_items is None:
            total_items = len(register_items) or len([i for i in instruments if not i.get("missing_from_scan")]) or None

        return BatchContext(
            batch_id=str(uuid.uuid4()),
            batch_number=batch_number,
            batch_type=batch_data.get("batch_type") or "Check/MO",
            batch_status=batch_data.get("batch_status"),
            pay_period=batch_data.get("pay_period"),
            bank_name=bank_name,
            account_number=account_number,
            property_name=batch_data.get("property_name"),
            property_address=batch_data.get("property_address"),
            deposited_date=batch_data.get("deposited_date") or deposit_data.get("deposit_date") or batch_data.get("printed_on") or date.today().isoformat(),
            deposit_transaction=batch_data.get("deposit_transaction") or deposit_data.get("deposit_transaction"),
            total_items=total_items,
            batch_amount=amount,
            printed_on=batch_data.get("printed_on"),
            source_system="YottaReal",
        )

    @staticmethod
    def _build_instruments(batch: BatchContext, raw_rows: List[Dict[str, Any]], file_name: str) -> List[Instrument]:
        instruments: List[Instrument] = []
        for idx, raw in enumerate(raw_rows, start=1):
            raw = sanitize_instrument(raw, ocr_text="")
            amount = _as_float(raw.get("amount_numeric"))
            payee = normalize_payee(raw.get("payee_raw"))
            batch_no = batch.batch_number
            instrument_id = f"INS-{batch_no or 'UNKNOWN'}-{idx:03d}"
            inst_type = raw.get("instrument_type") or "MoneyOrder"
            payment_description = raw.get("payment_description")
            if not payment_description:
                payment_description = "Payment-Check" if inst_type in {"Check", "CashiersCheck"} else "Payment-MoneyOrder"
            inst = Instrument(
                item_no=idx,
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
            if inst.missing_from_scan:
                continue
            groups[inst.payment_description or "Payment-MoneyOrder"] += inst.amount_numeric or 0.0
        acct_last4 = (batch.account_number or "")[-4:]
        property_name = (batch.property_name or "UNKNOWN PROPERTY").upper()
        out = []
        for code, (desc, amount) in enumerate(groups.items(), start=2001):
            out.append(
                {
                    "code": str(code),
                    "description": desc,
                    "gl_account": "1010-20",
                    "gl_account_name": f"Operating Bank Account ({property_name})" + (f" #{acct_last4}" if acct_last4 else ""),
                    "debit": round(amount, 2),
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
            "skipped_pages": sum(kinds.get(k.value, 0) for k in (PageKind.BACK, PageKind.BLANK, PageKind.RECEIPT)),
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
