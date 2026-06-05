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
    is_deposit_detail_report,
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
from money_order_validator.services.validation import apply_batch_reconciliation, compute_ocr_confidence, validate_instruments
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
    source = (row.get("source_system") or row.get("bank_name") or "").upper()
    # Strong key for exact duplicate receipt pages.
    if tx and (acct or date_value):
        return ("tx", source, tx, acct or "", date_value or "", amount, item_count or "")
    # Report continuation pages (e.g. Regions "Page 2 of 2") repeat the same deposit
    # total/item count without the table. Treat same account/date/amount/count as the
    # same deposit even when one page has a transaction number and the other does not.
    if acct or date_value:
        return ("deposit", source, acct or "", date_value or "", amount, item_count or "")
    # Deposit tickets often have no transaction/account/date; use page number as last-resort key.
    return ("page", row.get("page_number"), amount, item_count or "")


def _same_deposit_slip(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True when two parsed rows describe the same physical/logical deposit."""
    amount_a = _deposit_amount_value(a)
    amount_b = _deposit_amount_value(b)
    if amount_a is None or amount_b is None or abs(amount_a - amount_b) >= 0.005:
        return False
    acct_a = _clean_account(a.get("deposit_account") or a.get("account_last4"))
    acct_b = _clean_account(b.get("deposit_account") or b.get("account_last4"))
    date_a = a.get("deposit_date")
    date_b = b.get("deposit_date")
    count_a = a.get("item_count")
    count_b = b.get("item_count")
    source_a = (a.get("source_system") or a.get("bank_name") or "").upper()
    source_b = (b.get("source_system") or b.get("bank_name") or "").upper()
    if source_a and source_b and source_a != source_b:
        return False
    if acct_a and acct_b and acct_a != acct_b:
        return False
    if date_a and date_b and date_a != date_b:
        return False
    if count_a is not None and count_b is not None:
        try:
            if int(count_a) != int(count_b):
                return False
        except (TypeError, ValueError):
            return False
    # Require at least one stable identity besides amount so unrelated same-value slips remain distinct.
    return bool((acct_a and acct_b) or (date_a and date_b) or (count_a is not None and count_b is not None))


def _deposit_exact_duplicate_key(row: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Stable identity for duplicate physical copies of the same deposit slip/receipt.

    The public batch total must not be doubled when the same Chase ticket or receipt is
    OCR'd more than once. Exact same transaction + exact cents amount is a duplicate.
    """
    amount = _deposit_amount_value(row)
    if amount is None:
        return None
    amount_key = f"{round(float(amount) + 1e-9, 2):.2f}"
    tx = re.sub(r"\s+", "", str(row.get("deposit_transaction") or "")).upper()
    acct = _clean_account(row.get("deposit_account") or row.get("account_last4"))
    date_value = str(row.get("deposit_date") or "")
    source = re.sub(r"[^A-Z0-9]", "", str(row.get("source_system") or row.get("bank_name") or "").upper())
    try:
        count = int(row.get("item_count")) if row.get("item_count") is not None else None
    except (TypeError, ValueError):
        count = None

    if tx:
        return ("tx_amount", source, tx, amount_key, count if count is not None else "")
    if acct and date_value:
        return ("acct_date_amount", source, acct, date_value, amount_key, count if count is not None else "")
    if source and count is not None:
        return ("source_count_amount", source, amount_key, count)
    return None


def _dedupe_deposit_slips_exact(slips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse exact duplicate deposit slips before aggregate totals are calculated."""
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in slips:
        key = _deposit_exact_duplicate_key(row)
        if key is None:
            out.append(row)
            continue
        existing = seen.get(key)
        if existing is None:
            seen[key] = row
            out.append(row)
            continue
        # Merge metadata only; never add amounts/item counts for duplicates.
        prior_page = existing.get("page_number")
        row_page = row.get("page_number")
        _merge_non_empty(existing, row)
        pages = existing.setdefault("source_pages", [])
        for pg in (prior_page, row_page):
            if pg and pg not in pages:
                pages.append(pg)
        if pages:
            existing["page_number"] = min(pages)
    return out


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
    exact_key = _deposit_exact_duplicate_key(row)
    for existing in slips:
        existing_exact_key = _deposit_exact_duplicate_key(existing)
        if (exact_key is not None and existing_exact_key == exact_key) or _deposit_key(existing) == key or _same_deposit_slip(existing, row):
            prior_page = existing.get("page_number")
            _merge_non_empty(existing, row)
            pages = existing.setdefault("source_pages", [])
            if prior_page and prior_page not in pages:
                pages.append(prior_page)
            if page_number not in pages:
                pages.append(page_number)
            if pages:
                existing["page_number"] = min(pages)
            return
    slips.append(row)


def _aggregate_deposit_slips(slips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a single legacy deposit_slip object from one or more physical slips/receipts."""
    if not slips:
        return {}
    slips = _dedupe_deposit_slips_exact(slips)
    out: Dict[str, Any] = {}
    for row in slips:
        # Non-total metadata: keep the first useful value. deposit_date handled separately
        # because upside-down handwritten slips can OCR 2026 as 2024.
        for key in ("bank_name", "source_system", "account_last4", "account_name"):
            if not out.get(key) and row.get(key) not in (None, "", [], {}):
                out[key] = row[key]
    accounts = [str(row.get("deposit_account")) for row in slips if row.get("deposit_account") not in (None, "", [], {})]
    if accounts:
        # Chase deposit tickets often OCR the first MICR group as the account on one page and the
        # true deposit account on a later ticket. Prefer the latest non-empty account for the batch.
        out["deposit_account"] = accounts[-1]
    dates = [str(row.get("deposit_date")) for row in slips if row.get("deposit_date")]
    if dates:
        out["deposit_date"] = sorted(dates)[-1]
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


def _collapse_deposit_detail_report_outputs(
    deposit_data: Dict[str, Any],
    deposit_slips: List[Dict[str, Any]],
    register_items: List[Dict[str, Any]],
    report_pages: List[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize Deposit Detail Report PDFs to one logical deposit.

    This document family has one aggregate deposit/credit total and one row per physical
    payment item. Treating each report page as its own deposit slip creates duplicated
    totals; this function collapses them to a single authoritative entry.
    """
    if not report_pages:
        return deposit_data, deposit_slips, {}

    report_items = [r for r in register_items if r.get("source") == "transaction_detail_report"]
    item_sum = round(sum(_as_float(r.get("amount_numeric")) or 0.0 for r in report_items), 2) if report_items else None

    normalized = dict(deposit_data or {})
    normalized["bank_name"] = normalize_bank_name(normalized.get("bank_name") or "JPMorgan Chase Bank")
    normalized["source_system"] = normalized.get("source_system") or "Deposit Detail Report"
    normalized["source_pages"] = sorted(set(report_pages))
    normalized["page_number"] = min(report_pages)

    parsed_total = _as_float(normalized.get("deposit_total") or normalized.get("check_total") or normalized.get("credit_total") or normalized.get("deposit_amount"))
    total = parsed_total if parsed_total is not None else item_sum
    if item_sum is not None and (total is None or abs(float(total) - float(item_sum)) <= 1.00 or float(total) > float(item_sum) * 1.5):
        total = item_sum
    if total is not None:
        normalized["deposit_amount"] = round(float(total), 2)
        normalized["deposit_total"] = round(float(total), 2)
        normalized["check_total"] = round(float(total), 2)

    expected_physical_count = None
    try:
        if normalized.get("debit_items") is not None:
            expected_physical_count = int(normalized.get("debit_items"))
        elif normalized.get("report_item_count") is not None:
            credit_items = int(normalized.get("credit_items") or 1)
            expected_physical_count = max(int(normalized.get("report_item_count")) - credit_items, 0)
    except (TypeError, ValueError):
        expected_physical_count = None

    if report_items:
        # Prefer Debit Items from the control block over raw row count.
        normalized["item_count"] = expected_physical_count or len(report_items)

    # If the row extractor still misses a small/blurred row, add a balancing REVIEW
    # placeholder rather than rejecting an internally balanced report.
    if expected_physical_count and total is not None and report_items:
        current_sum = round(sum(_as_float(r.get("amount_numeric")) or 0.0 for r in report_items), 2)
        gap = round(float(total) - current_sum, 2)
        missing_count = int(expected_physical_count) - len(report_items)
        if missing_count > 0 and abs(gap) >= 0.01:
            gap_item = {
                "item_no": len(report_items) + 1,
                "amount_numeric": gap,
                "instrument_type": "MoneyOrder",
                "payment_description": "Payment-MoneyOrder",
                "source": "transaction_detail_report_gap",
                "source_system": "Deposit Detail Report",
                "image_quality": "unread_report_row",
                "matched_register_item": True,
                "missing_from_scan": False,
                "review_flags": ["unread_deposit_detail_report_row", "manual_review_required"],
            }
            report_items.append(gap_item)
            register_items.append(gap_item)
            normalized["item_count"] = expected_physical_count

    for noisy in ("debit_total", "credit_total", "difference", "report_item_count", "credit_items", "debit_items"):
        normalized.pop(noisy, None)

    batch_patch: Dict[str, Any] = {}
    if normalized.get("deposit_amount") is not None:
        batch_patch["batch_amount"] = normalized["deposit_amount"]
    if normalized.get("item_count") is not None:
        batch_patch["total_items"] = normalized["item_count"]
    if normalized.get("deposit_date"):
        batch_patch["deposited_date"] = normalized.get("deposit_date")
        batch_patch["printed_on"] = normalized.get("deposit_date")
    if normalized.get("deposit_account"):
        batch_patch["account_number"] = normalized.get("deposit_account")
    if normalized.get("deposit_transaction"):
        batch_patch["deposit_transaction"] = normalized.get("deposit_transaction")
    if normalized.get("account_name"):
        batch_patch["property_name"] = normalized.get("account_name")
        batch_patch["property_aliases"] = build_property_aliases(normalized.get("account_name"))

    normalized["deposit_slip_count"] = 1
    public_row = dict(normalized)
    public_row.pop("deposit_slips", None)
    normalized["deposit_slips"] = [public_row]
    return normalized, [public_row], batch_patch


def _should_extract_register_with_vision(kind: PageKind, ocr_text: str, page_deposit: Dict[str, Any]) -> bool:
    """Use focused register vision only for likely table/report pages."""
    t = (ocr_text or "").upper()
    if kind not in {PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP, PageKind.BATCH_HEADER}:
        return False
    if page_deposit.get("source_system") == "Regions" and not re.search(r"\(\s*CONTINUED\s*\)|PAGE\s+2\s+OF\s+2", t):
        return True
    return bool(
        re.search(
            r"CAPTURE\s+SEQ|CHECK\s+NUMBER|POST\s+AMOUNT|TRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION|"
            r"DEPOSIT\s+CONTROL\s+INFORMATION|PAYMENT[-\s]?MONEYORDER|PAYMENT[-\s]?CHECK",
            t,
            flags=re.IGNORECASE,
        )
    )


def _bad_property_candidate(value: Optional[str]) -> bool:
    if not value:
        return True
    compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return compact in {
        "CENTS", "DOLLARS", "DOLLAR", "CURRENCY", "COIN", "TOTAL",
        "CHASE", "JPMORGANCHASEBANK", "REGIONS", "DEPOSIT", "DEPOSITTICKET",
        "DEPOSITSLIP", "PLEASEENTERLOCALBRANCHTOTAL",
    }


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
    issuer_text = " ".join(str(inst.get(k) or "") for k in ("issuer", "issuer_agent", "payer_name", "payee_raw", "micr_line")).upper()
    no_amount = _as_float(inst.get("amount_numeric")) is None
    no_payee = not normalize_payee(inst.get("payee_raw"))
    no_written_amount = not inst.get("amount_words")
    no_signature = not inst.get("payer_signature")
    # A Chase deposit ticket can be hallucinated as a MoneyOrder/Check because it has a MICR line,
    # a date, and a total box. It is not a physical instrument when it has no real payee/amount words.
    if ("CHASE" in issuer_text or "JPMORGAN" in issuer_text) and no_payee and no_written_amount and (no_amount or no_signature):
        return True
    no_face_evidence = no_payee and no_written_amount and no_signature
    # Pure deposit tickets are sometimes returned as a MoneyOrder/Chase row because the MICR line
    # contains numeric groups. Drop any no-evidence row on a deposit-ticket page, not just checks.
    if no_face_evidence and no_amount:
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


def _sequence_group_key(row: Dict[str, Any]) -> Optional[int]:
    if row.get("source") != "deposit_ticket_sequence":
        return None
    try:
        return int(row.get("slip_page_number"))
    except (TypeError, ValueError):
        return None


def _adjust_deposit_slips_from_sequence_items(slips: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> None:
    """Correct deposit-ticket totals using extracted row-level amounts.

    Handwritten Chase deposit-ticket totals are often OCR'd incorrectly (e.g. 3,448.80
    instead of 5,448.80). If the row count matches the slip item count, the sum of the
    filled table rows is more reliable than the handwritten total.
    """
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        page = _sequence_group_key(item)
        if page is not None:
            by_page[page].append(item)
    for slip in slips:
        try:
            page = int(slip.get("page_number"))
        except (TypeError, ValueError):
            continue
        rows = by_page.get(page) or []
        if not rows:
            continue
        row_sum = round(sum(_as_float(r.get("amount_numeric")) or 0.0 for r in rows), 2)
        if row_sum <= 0:
            continue
        try:
            item_count = int(slip.get("item_count") or 0)
        except (TypeError, ValueError):
            item_count = 0
        old_amount = _deposit_amount_value(slip)
        if (item_count and item_count == len(rows)) or old_amount is None or abs(row_sum - float(old_amount)) >= 10.0:
            slip["deposit_amount"] = row_sum
            slip["deposit_total"] = row_sum
            slip["check_total"] = row_sum
            slip["item_count"] = len(rows) if not item_count else item_count
            slip.setdefault("corrections", []).append({
                "field": "deposit_amount",
                "old": old_amount,
                "new": row_sum,
                "source": "deposit_ticket_row_sum",
            })


def _row_has_real_front_evidence(row: Dict[str, Any]) -> bool:
    """True when a row looks like a real instrument front, not a form/back artifact."""
    return bool(
        _as_float(row.get("amount_numeric")) is not None
        and (normalize_serial(row.get("serial_number")) or row.get("micr_line"))
        and (normalize_payee(row.get("payee_raw")) or row.get("amount_words"))
    )


def _strip_output_debug_fields(value: Any) -> Any:
    """Remove internal correction/debug metadata from the public API payload."""
    if isinstance(value, dict):
        return {
            k: _strip_output_debug_fields(v)
            for k, v in value.items()
            if k not in {"corrections", "_ocr_text", "_page_item_index"}
        }
    if isinstance(value, list):
        return [_strip_output_debug_fields(v) for v in value]
    return value


def _low_evidence_form_or_back_row(row: Dict[str, Any]) -> bool:
    """Rows emitted from backs/deposit tickets often have only issuer/amount and no face fields."""
    return bool(
        not normalize_serial(row.get("serial_number"))
        and not normalize_payee(row.get("payee_raw"))
        and not row.get("issue_date")
        and not row.get("amount_words")
        and not row.get("payer_signature")
    )


def _raw_confidence(row: Dict[str, Any]) -> float:
    fields = [
        row.get("serial_number"),
        row.get("amount_numeric"),
        row.get("issue_date"),
        row.get("payee_raw"),
        row.get("issuer") or row.get("micr_line"),
    ]
    return sum(1 for x in fields if x not in (None, "", [])) / max(1, len(fields))


def _looks_visually_unclear(row: Dict[str, Any]) -> bool:
    """Mark weak extractions for REVIEW instead of pretending they are exact.

    Conservative: does not delete the item, just forces manual review when the image
    or extraction evidence is weak (sideways/faint scans like upside-down MoneyGrams).
    """
    if row.get("missing_from_scan"):
        return False
    if row.get("image_quality") == "unclear":
        return True

    confidence = _raw_confidence(row)
    has_serial = bool(normalize_serial(row.get("serial_number")) or row.get("micr_line"))
    has_amount = _as_float(row.get("amount_numeric")) is not None or bool(row.get("amount_words"))
    has_payee = bool(normalize_payee(row.get("payee_raw")))
    has_date = bool(row.get("issue_date"))
    has_words = bool(str(row.get("amount_words") or "").strip())
    inst_type = str(row.get("instrument_type") or "").lower()
    ocr_text = str(row.get("_ocr_text") or "")

    if confidence < 0.65:
        return True
    if not has_serial and not has_amount:
        return True
    if inst_type in {"moneyorder", "money_order"} and not has_words and (not has_payee or not has_date):
        return True
    # Sideways/faint scans often produce impossible or suspicious dates.
    issue_date = str(row.get("issue_date") or "")
    if re.match(r"^(20[3-9][0-9]|19[0-9]{2})-", issue_date):
        return True
    if re.search(r"QUICKDEPOSIT|DEPOSIT\s+ACTIVITY|WE\s+FOUND\s+\d+\s+DEPOSIT\s+ITEMS", ocr_text, re.I):
        return True
    return False


def _mark_unclear_instruments(rows: List[Dict[str, Any]], batch_data: Dict[str, Any], deposit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply review-only image quality flags and cap over-extraction for unclear duplicates.

    When the bank/deposit register says there are N items but vision emits more than N
    rows, keep the strongest N rows and mark weak rows as unclear before validation.
    """
    try:
        total_items = int(batch_data.get("total_items") or deposit_data.get("item_count") or 0)
    except (TypeError, ValueError):
        total_items = 0

    for row in rows:
        if _looks_visually_unclear(row):
            row["image_quality"] = "unclear"
            row.setdefault("review_flags", [])
            for flag in ("unclear_instrument_image", "low_confidence_extraction", "manual_review_required"):
                if flag not in row["review_flags"]:
                    row["review_flags"].append(flag)

    if total_items > 0 and len(rows) > total_items:
        def rank(row: Dict[str, Any]) -> Tuple[int, int, int]:
            clear_score = 0 if row.get("image_quality") == "unclear" else 1
            return (clear_score, DocumentProcessor._instrument_quality_score(row), -int(row.get("page_number") or 999999))

        kept = sorted(rows, key=rank, reverse=True)[:total_items]
        kept_ids = {id(r) for r in kept}
        dropped = [r for r in rows if id(r) not in kept_ids]
        for row in kept:
            if dropped:
                row.setdefault("batch_review_notes", [])
                note = f"vision_extracted_{len(rows)}_rows_but_deposit_register_count_is_{total_items}; weakest_rows_suppressed"
                if note not in row["batch_review_notes"]:
                    row["batch_review_notes"].append(note)
        rows = sorted(kept, key=lambda r: (r.get("page_number") or 999999, r.get("_page_item_index") or 0))
    return rows


def _correct_deposit_slips_from_following_instruments(slips: List[Dict[str, Any]], instruments: List[Dict[str, Any]]) -> None:
    """Use the actual following physical instruments to correct handwritten Chase slip totals.

    The focused deposit-ticket row reader can still miss a leading digit on a row. The physical
    scan order is more stable: a Chase deposit ticket is followed by exactly its listed items,
    until the next deposit ticket. When the following real instruments count matches the slip
    item_count, their sum is the safest total.
    """
    slip_pages = sorted(
        int(s.get("page_number")) for s in slips
        if str(s.get("source_system") or s.get("bank_name") or "").lower().find("chase") >= 0
        and str(s.get("page_number") or "").isdigit()
    )
    if not slip_pages:
        return
    for idx, page in enumerate(slip_pages):
        next_page = slip_pages[idx + 1] if idx + 1 < len(slip_pages) else 10 ** 9
        slip = next((s for s in slips if int(s.get("page_number") or -1) == page), None)
        if not slip:
            continue
        targets = [
            r for r in instruments
            if (r.get("page_number") or 0) > page
            and (r.get("page_number") or 0) < next_page
            and not r.get("missing_from_scan")
            and _row_has_real_front_evidence(r)
        ]
        if not targets:
            continue
        try:
            item_count = int(slip.get("item_count") or 0)
        except (TypeError, ValueError):
            item_count = 0
        if item_count and len(targets) != item_count:
            continue
        target_sum = round(sum(_as_float(r.get("amount_numeric")) or 0.0 for r in targets), 2)
        old_amount = _deposit_amount_value(slip)
        if target_sum > 0 and (old_amount is None or abs(target_sum - float(old_amount)) >= 1.00):
            slip["deposit_amount"] = target_sum
            slip["deposit_total"] = target_sum
            slip["check_total"] = target_sum
            slip["item_count"] = len(targets) if not item_count else item_count
            slip.setdefault("corrections", []).append({
                "field": "deposit_amount",
                "old": old_amount,
                "new": target_sum,
                "source": "following_instrument_sum",
            })


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
        ocr_angles = self._ocr_angles_by_page(ocr_pages, len(images))
        raw_instruments: List[Dict[str, Any]] = []
        batch_data: Dict[str, Any] = {}
        deposit_data: Dict[str, Any] = {}
        deposit_slips: List[Dict[str, Any]] = []
        register_items: List[Dict[str, Any]] = []
        deposit_detail_pages: List[int] = []
        page_logs: List[Dict[str, Any]] = []
        usage_total = TokenUsage()
        llm_calls = 0
        phase_tokens: Dict[str, int] = defaultdict(int)

        for page_idx, image in enumerate(images):
            page_number = page_idx + 1
            ocr_text = ocr_texts[page_idx]
            page_angle = ocr_angles[page_idx] if page_idx < len(ocr_angles) else None
            kind, scores = classify_page(ocr_text)
            # If Azure DI is unavailable or returned no text for an image-only page,
            # do not treat it as blank when an LLM vision client is configured.
            if not ocr_text.strip() and llm_client.available:
                kind = PageKind.UNKNOWN
                scores = {**scores, "no_ocr_vision_fallback": 1}
            log_row: Dict[str, Any] = {
                "page_number": page_number,
                "kind": kind.value,
                "ocr_angle": page_angle,
                "scores": scores,
                "ocr_chars": len(ocr_text),
                "llm_used": False,
                "instruments": 0,
            }

            # Text-first parsing. Deposit receipts/tickets can appear on receipt pages or on
            # mixed pages above instruments, so parse deposit metadata on every OCR page. Batch
            # header/property parsing is still restricted to report-like pages to avoid treating a
            # check drawer as the property name.
            page_is_deposit_detail_report = is_deposit_detail_report(ocr_text)
            if page_is_deposit_detail_report:
                deposit_detail_pages.append(page_number)
                log_row["deposit_detail_report"] = True
            page_deposit = parse_deposit_info(ocr_text)
            if page_deposit:
                _merge_non_empty(deposit_data, page_deposit)
                if not page_is_deposit_detail_report:
                    _add_deposit_slip(deposit_slips, page_deposit, page_number)
                log_row["deposit_detected"] = True
            if kind in {PageKind.BATCH_HEADER, PageKind.DEPOSIT_REPORT, PageKind.REPORT_WITH_INSTRUMENTS, PageKind.DEPOSIT_SLIP, PageKind.UNKNOWN, PageKind.RECEIPT}:
                _merge_non_empty(batch_data, parse_batch_header(ocr_text))
                new_items = parse_batch_line_items(ocr_text)
                if page_is_deposit_detail_report:
                    # Deposit Detail Report rows are authoritative. Always use the
                    # focused row-table extractor because Azure OCR often misses
                    # rows from the black report table or confuses thumbnail text.
                    detail_items, usage, used = await vision_extractor.extract_deposit_detail_report_items(image, ocr_text)
                    if used:
                        llm_calls += 1
                        self._accumulate_usage(usage_total, usage)
                        phase_tokens["deposit_detail_rows"] += usage.total_tokens
                        log_row["llm_used"] = True
                        log_row["deposit_detail_rows_vision_used"] = True
                    if detail_items:
                        new_items = detail_items
                elif not new_items and _should_extract_register_with_vision(kind, ocr_text, page_deposit):
                    new_items, usage, used = await vision_extractor.extract_register_items(image, ocr_text)
                    if used:
                        llm_calls += 1
                        self._accumulate_usage(usage_total, usage)
                        phase_tokens["register"] += usage.total_tokens
                        log_row["llm_used"] = True
                        log_row["register_vision_used"] = True
                if new_items:
                    for row in new_items:
                        row.setdefault("report_page_number", page_number)
                    register_items.extend(new_items)
                    if page_is_deposit_detail_report:
                        log_row["deposit_detail_report_items"] = len(new_items)

            # Receipts and mixed deposit-ticket+instrument pages may carry the authoritative
            # deposit total. This is page-scoped, not global: a single PDF can contain multiple
            # physical deposit slips/receipts and we need each one's total separately.
            needs_deposit_total = not (page_deposit.get("deposit_amount") or page_deposit.get("deposit_total") or page_deposit.get("check_total"))
            if (not page_is_deposit_detail_report) and needs_deposit_total and (
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

            if (not page_is_deposit_detail_report) and _page_has_deposit_ticket(ocr_text, page_deposit) and (page_deposit.get("item_count") or kind in {PageKind.DEPOSIT_SLIP, PageKind.DEPOSIT_REPORT, PageKind.REPORT_WITH_INSTRUMENTS, PageKind.UNKNOWN}):
                seq_items, usage, used = await vision_extractor.extract_deposit_ticket_items(
                    image,
                    ocr_text,
                    expected_count=int(page_deposit.get("item_count") or 0) or None,
                    expected_total=_deposit_amount_value(page_deposit),
                )
                if used:
                    llm_calls += 1
                    self._accumulate_usage(usage_total, usage)
                    phase_tokens["deposit_items"] += usage.total_tokens
                    log_row["llm_used"] = True
                    log_row["deposit_ticket_items_vision_used"] = True
                if seq_items:
                    for row in seq_items:
                        row["slip_page_number"] = page_number
                    register_items.extend(seq_items)
                    log_row["deposit_ticket_items"] = len(seq_items)

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
                elif kind in {PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP} and not page_is_deposit_detail_report and not (page_deposit.get("deposit_amount") or page_deposit.get("deposit_total") or page_deposit.get("check_total")):
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

            should_extract = False if page_is_deposit_detail_report else (kind == PageKind.INSTRUMENT or kind == PageKind.REPORT_WITH_INSTRUMENTS)
            if kind == PageKind.UNKNOWN and not page_is_deposit_detail_report:
                should_extract = _should_try_unknown_vision(ocr_text)
            if should_extract:
                instruments, usage, used = await vision_extractor.extract_instruments(image, ocr_text, kind, page_angle=page_angle)
                if used:
                    llm_calls += 1
                    self._accumulate_usage(usage_total, usage)
                    phase_tokens["instrument"] += usage.total_tokens
                    log_row["llm_used"] = True
                kept = 0
                sanitized_page: List[Dict[str, Any]] = [sanitize_instrument(inst, ocr_text=ocr_text) for inst in instruments]
                for local_idx, inst in enumerate(sanitized_page, start=1):
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
                    inst["_page_item_index"] = local_idx
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
        if deposit_detail_pages:
            deposit_data, deposit_slips, deposit_detail_batch_patch = _collapse_deposit_detail_report_outputs(
                deposit_data, deposit_slips, register_items, deposit_detail_pages
            )
            _merge_non_empty(batch_data, deposit_detail_batch_patch)
            # For Deposit Detail Report packets, the printed report rows are the
            # authoritative instruments. Clear any thumbnail-extracted raw instruments.
            raw_instruments = []
        raw_instruments = self._dedupe_raw_instruments(raw_instruments)
        raw_instruments = self._drop_form_and_back_artifacts(raw_instruments, deposit_slips)
        _adjust_deposit_slips_from_sequence_items(deposit_slips, register_items)
        raw_instruments = self._apply_deposit_ticket_sequences(raw_instruments, register_items)
        raw_instruments = self._drop_form_and_back_artifacts(raw_instruments, deposit_slips)
        _correct_deposit_slips_from_following_instruments(deposit_slips, raw_instruments)
        non_sequence_register_items = [i for i in register_items if i.get("source") != "deposit_ticket_sequence"]
        raw_instruments = self._reconcile_register_items(raw_instruments, non_sequence_register_items)
        raw_instruments = _mark_unclear_instruments(raw_instruments, batch_data, deposit_data)
        deposit_slips = _dedupe_deposit_slips_exact(deposit_slips)
        aggregate_deposit = _aggregate_deposit_slips(deposit_slips)
        if aggregate_deposit:
            deposit_data = {**deposit_data, **aggregate_deposit}
        batch = self._build_batch(batch_data, deposit_data, register_items, raw_instruments)
        instruments = self._build_instruments(batch, raw_instruments, file_name)
        validate_instruments(batch, instruments)
        apply_batch_reconciliation(batch, instruments)
        # gl_summary is internal-only and excluded from the public API output.
        batch.gl_summary = []
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

        # Hide internal reconciliation/correction history from public JSON output.
        # This keeps the API response clean while preserving the corrected values.
        public_deposit_data = _strip_output_debug_fields(deposit_data) if deposit_data else None
        public_deposit_slips = _strip_output_debug_fields(deposit_slips) if deposit_slips else None
        public_pages = _strip_output_debug_fields(page_logs) if settings.return_debug_pages else None

        return ValidationResult(
            file_name=file_name,
            batch=batch,
            instruments=instruments,
            deposit_slip=public_deposit_data,
            deposit_slips=public_deposit_slips,
            pages=public_pages,
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
    def _ocr_angles_by_page(pages: List[OcrPage], expected_pages: int) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * expected_pages
        for p in pages:
            idx = p.page_number - 1
            if 0 <= idx < expected_pages:
                out[idx] = p.angle
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
                row.get("slip_page_number") if row.get("source") == "deposit_ticket_sequence" else None,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _instrument_quality_score(row: Dict[str, Any]) -> int:
        """Integer quality rank used to select the best rows when over-extraction occurs."""
        score = 0
        if normalize_serial(row.get("serial_number")):
            score += 3
        if _as_float(row.get("amount_numeric")) is not None:
            score += 2
        if row.get("amount_words"):
            score += 2
        if normalize_payee(row.get("payee_raw")):
            score += 1
        if row.get("issue_date"):
            score += 1
        if row.get("micr_line"):
            score += 1
        return score

    @staticmethod
    def _drop_form_and_back_artifacts(rows: List[Dict[str, Any]], deposit_slips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove pseudo instruments created from deposit tickets and adjacent back pages."""
        deposit_pages = {
            int(s.get("page_number")) for s in deposit_slips
            if str(s.get("page_number") or "").isdigit()
        }
        valid_front_pages = {
            int(r.get("page_number")) for r in rows
            if str(r.get("page_number") or "").isdigit() and _row_has_real_front_evidence(r)
        }
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                page = int(row.get("page_number") or 0)
            except (TypeError, ValueError):
                page = 0
            issuer_text = " ".join(str(row.get(k) or "") for k in ("issuer", "issuer_agent", "payer_name", "payee_raw", "micr_line")).upper()
            on_deposit_page = page in deposit_pages
            if on_deposit_page and ("CHASE" in issuer_text or "JPMORGAN" in issuer_text) and not normalize_payee(row.get("payee_raw")) and not row.get("amount_words"):
                continue
            # In these scans, a low-evidence row on the page immediately after a real instrument
            # is nearly always the instrument back, not a new item.
            if (page - 1) in valid_front_pages and _low_evidence_form_or_back_row(row):
                continue
            out.append(row)
        return out

    @staticmethod
    def _apply_deposit_ticket_sequences(instruments: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply ordered handwritten deposit-ticket amounts to following instruments.

        Deposit tickets in these batches are immediately followed by the physical items
        in the same row order. The slip rows often contain the only reliable values when
        an instrument front is upside down.
        """
        seq_pages = sorted({p for p in (_sequence_group_key(i) for i in items) if p is not None})
        if not seq_pages:
            return instruments

        for n, slip_page in enumerate(seq_pages):
            next_slip_page = seq_pages[n + 1] if n + 1 < len(seq_pages) else 10**9
            seq_rows = [i for i in items if _sequence_group_key(i) == slip_page]
            seq_rows.sort(key=lambda r: int(r.get("item_no") or 999999))
            targets = [
                inst for inst in instruments
                if (inst.get("page_number") or 0) > slip_page
                and (inst.get("page_number") or 0) < next_slip_page
                and not inst.get("missing_from_scan")
                and not _is_deposit_ticket_artifact(
                    inst,
                    ocr_text="",
                    page_deposit=None,
                    page_instruments=instruments,
                )
            ]
            targets.sort(key=lambda r: (r.get("page_number") or 999999, r.get("_page_item_index") or 0))
            if not seq_rows or not targets:
                continue
            if abs(len(seq_rows) - len(targets)) > 1:
                continue
            for row, inst in zip(seq_rows, targets):
                seq_amount = _as_float(row.get("amount_numeric"))
                if seq_amount is None:
                    continue
                old_amount = _as_float(inst.get("amount_numeric"))

                # Deposit-ticket row OCR is weaker than the instrument itself. It is useful
                # for filling blanks and restoring missing cents, but it must not overwrite
                # a real instrument amount/amount_words with a lower-confidence handwritten
                # row read. The repeated failure was inverted MOs like 442.00 -> 400.00 or
                # 525.50 -> 525.00 being made worse by ticket-row extraction.
                has_words = bool(str(inst.get("amount_words") or "").strip())
                should_apply_amount = False
                if old_amount is None:
                    should_apply_amount = True
                elif not has_words:
                    same_dollars = int(float(old_amount)) == int(float(seq_amount))
                    old_has_no_cents = abs(float(old_amount) - int(float(old_amount))) < 0.005
                    seq_has_cents = abs(float(seq_amount) - int(float(seq_amount))) >= 0.005
                    # Safe cents-only repair, e.g. 621 -> 621.50.
                    should_apply_amount = bool(same_dollars and old_has_no_cents and seq_has_cents)

                if should_apply_amount and old_amount != seq_amount:
                    inst.setdefault("corrections", []).append({
                        "field": "amount_numeric",
                        "old": old_amount,
                        "new": seq_amount,
                        "source": "deposit_ticket_sequence_fill_only",
                        "slip_page_number": slip_page,
                        "row_item_no": row.get("item_no"),
                    })
                    inst["amount_numeric"] = seq_amount
                elif old_amount != seq_amount:
                    inst.setdefault("corrections", []).append({
                        "field": "amount_numeric",
                        "old": old_amount,
                        "rejected_new": seq_amount,
                        "source": "deposit_ticket_sequence_rejected_instrument_has_words",
                        "slip_page_number": slip_page,
                        "row_item_no": row.get("item_no"),
                    })

                if row.get("unit") and not inst.get("unit"):
                    inst["unit"] = str(row.get("unit"))
                inst["matched_deposit_ticket_item"] = True
        return instruments

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
                source = str(item.get("source") or "")
                report_row = source == "transaction_detail_report" or str(item.get("source_system") or "") == "Deposit Detail Report"
                instruments.append(
                    {
                        **item,
                        "instrument_type": inst_type,
                        "payment_description": item.get("payment_description") or "Payment-Check",
                        "llm_used": False,
                        "processing_tier": 1,
                        "missing_from_scan": False if report_row else True,
                        "matched_register_item": bool(report_row),
                        "image_quality": item.get("image_quality") or ("thumbnail_report_image" if report_row else None),
                        "review_flags": item.get("review_flags") or (["report_thumbnail_item", "manual_review_required"] if report_row else []),
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
        register_sum = round(sum(_as_float(i.get("amount_numeric")) or 0.0 for i in register_items), 2) if register_items else None
        deposit_count = int(deposit_data.get("deposit_slip_count") or 0)

        # A parsed Chase/bank deposit amount is authoritative for the batch. This also fixes
        # duplicate-scan cases where the same exact slip is parsed twice and batch_data carries
        # a doubled total.
        if deposit_amount is not None:
            amount = deposit_amount
        if amount is None and register_sum is not None:
            amount = register_sum
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
        if (
            amount is not None
            and register_sum is not None
            and abs(float(amount) - float(register_sum)) > 1.00
            and (instrument_sum is None or abs(float(register_sum) - float(instrument_sum)) <= 1.00)
        ):
            amount = register_sum

        deposit_item_count = deposit_data.get("item_count")
        total_items = deposit_item_count if deposit_item_count is not None else batch_data.get("total_items")
        try:
            total_items = int(total_items) if total_items is not None else None
        except (TypeError, ValueError):
            total_items = None
        if total_items is None:
            total_items = len(register_items) or len([i for i in instruments if not i.get("missing_from_scan")]) or None

        property_name = batch_data.get("property_name") or deposit_data.get("account_name")
        inferred_property = False
        if _bad_property_candidate(property_name):
            property_name = _infer_property_from_instruments(instruments) or None
            inferred_property = True
        property_aliases = batch_data.get("property_aliases") or build_property_aliases(property_name)
        if inferred_property or any(_bad_property_candidate(a) for a in (property_aliases or [])):
            property_aliases = build_property_aliases(property_name)

        deposited_date = batch_data.get("deposited_date") or deposit_data.get("deposit_date") or batch_data.get("printed_on") or date.today().isoformat()
        # For Chase deposit-ticket batches, the teller/deposit date should match the printed ticket
        # date; OCR commonly flips 06/03 into 03/04 on handwritten rotated tickets.
        if (deposit_data.get("source_system") == "Chase" or deposit_data.get("bank_name") == "Chase") and batch_data.get("printed_on"):
            deposited_date = batch_data.get("printed_on")

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
            deposited_date=deposited_date,
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
                image_quality=raw.get("image_quality"),
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
