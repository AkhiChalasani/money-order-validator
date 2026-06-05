from __future__ import annotations

from collections import Counter, defaultdict
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from money_order_validator.schemas import BatchContext, Instrument
from money_order_validator.services.regex_parsers import similarity

STANDARD_ISSUERS = {
    "Western Union",
    "MoneyGram",
    "PLS",
    "DolEx",
    "Intermex",
    "Fidelity Express",
    "JPMorgan Chase",
    "Wells Fargo",
    "Comerica Bank",
    "Prosperity Bank",
    None,
}


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def compute_ocr_confidence(raw: Dict[str, Any]) -> float:
    fields = [
        raw.get("serial_number"),
        raw.get("amount_numeric"),
        raw.get("issue_date"),
        raw.get("payee_raw"),
        raw.get("issuer") or raw.get("micr_line"),
    ]
    return round(sum(1 for x in fields if x not in (None, "", [])) / len(fields), 2)


def _compact_name(value: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _property_aliases(property_name: Optional[str]) -> List[str]:
    if not property_name:
        return []
    aliases = {property_name.strip()}
    raw = property_name.strip()

    m = re.search(r"\bdba\b\s+(.+)$", raw, flags=re.IGNORECASE)
    if m:
        aliases.add(m.group(1).strip())

    for value in list(aliases):
        cleaned = re.sub(r"\b(?:LLC|LP|L\.P\.|LTD|INC|LIMITED|PARTNERSHIP)\b", " ", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,-.")
        if cleaned:
            aliases.add(cleaned)
        # Many reports use account names like "Arella Forest in Woodland" while
        # tenants write only "Arella" or "Arella Forest" on the payee line.
        for sep in (" in ", " at ", " dba "):
            if sep in value.lower():
                head = re.split(sep, value, flags=re.IGNORECASE)[0].strip(" ,-.")
                tail = re.split(sep, value, flags=re.IGNORECASE)[-1].strip(" ,-.")
                if head:
                    aliases.add(head)
                if tail and len(tail.split()) > 1:
                    aliases.add(tail)

    return [a for a in aliases if a]


def apply_batch_reconciliation(batch: BatchContext, instruments: List[Instrument]) -> None:
    """Add batch-level amount/count reconciliation and adjust the final decision.

    A matched dollar total is necessary for ACCEPT, but not sufficient. If any
    item still has review/manual flags, the batch remains REVIEW. If totals or
    counts do not match, the batch becomes REJECT.
    """
    instrument_rows = [i for i in instruments if not i.missing_from_scan]
    instrument_sum = round(sum(float(i.amount_numeric or 0.0) for i in instrument_rows), 2)
    deposit_total = round(float(batch.batch_amount), 2) if batch.batch_amount is not None else None
    difference = None if deposit_total is None else round(instrument_sum - deposit_total, 2)
    amount_tolerance = 0.01
    amounts_match = bool(difference is not None and abs(difference) <= amount_tolerance)

    instrument_count = len(instrument_rows)
    expected_count = int(batch.total_items) if batch.total_items is not None else None
    item_count_match = bool(expected_count is None or expected_count == instrument_count)

    item_flags = []
    for inst in instruments:
        item_flags.extend(inst.validation.get("flags", []) if inst.validation else [])
    has_manual_review_flags = any(
        flag in item_flags
        for flag in (
            "unclear_instrument_image",
            "low_confidence_extraction",
            "manual_review_required",
            "missing_serial_number",
            "missing_amount",
            "missing_issue_date",
            "missing_payee",
            "payee_mismatch",
            "duplicate_serial_number",
            "date_outside_120_days",
        )
    )
    has_invalid_items = any((inst.validation or {}).get("overall_status") == "INVALID" for inst in instruments)

    if not amounts_match or not item_count_match:
        decision = "FAIL"
        overall = "REJECT"
    elif has_invalid_items or has_manual_review_flags:
        decision = "PASS_WITH_REVIEW"
        overall = "REVIEW"
    else:
        decision = "PASS"
        overall = "ACCEPT"

    flags = []
    if amounts_match:
        flags.append("amounts_reconciled")
    else:
        flags.append("amount_mismatch")
    if item_count_match:
        flags.append("item_count_reconciled")
    else:
        flags.append("item_count_mismatch")
    if decision == "PASS_WITH_REVIEW":
        flags.append("manual_review_required_for_item_flags")

    batch.reconciliation = {
        "instrument_sum": instrument_sum,
        "deposit_total": deposit_total,
        "batch_amount": deposit_total,
        "difference": difference,
        "amounts_match": amounts_match,
        "instrument_count": instrument_count,
        "expected_item_count": expected_count,
        "item_count_match": item_count_match,
        "decision": decision,
        "flags": flags,
    }

    # Keep risk_summary aligned with reconciliation.
    batch.overall_decision = overall
    batch.risk_summary["overall_decision"] = overall
    batch.risk_summary["reconciliation_decision"] = decision
    batch.risk_summary["reconciliation_flags"] = flags


def _payee_match_score(payee: Optional[str], property_name: Optional[str], extra_aliases: Optional[List[str]] = None) -> float:
    if not payee or not property_name:
        return 0.0
    aliases = _property_aliases(property_name)
    for alias in (extra_aliases or []):
        if alias and alias not in aliases:
            aliases.append(alias)
    scores = [similarity(payee, alias) for alias in aliases]
    payee_c = _compact_name(payee)
    for alias in aliases:
        alias_c = _compact_name(alias)
        if payee_c and alias_c and min(len(payee_c), len(alias_c)) >= 5:
            if payee_c in alias_c or alias_c in payee_c:
                scores.append(0.95)
        for token in re.findall(r"[A-Za-z0-9]{5,}", alias):
            token_score = similarity(payee, token)
            if token_score >= 0.80:
                scores.append(max(0.88, token_score))
    return round(max(scores or [0.0]), 3)


def validate_instruments(batch: BatchContext, instruments: List[Instrument]) -> None:
    serials = [i.serial_number for i in instruments if i.serial_number]
    serial_counts = Counter(serials)

    split_groups: Dict[tuple, List[Instrument]] = defaultdict(list)
    for inst in instruments:
        if inst.unit and inst.issue_date:
            split_groups[(inst.unit, inst.issue_date)].append(inst)

    items_valid = items_review = items_invalid = items_flagged = 0
    today = date.today()

    for inst in instruments:
        flags: List[str] = []
        score = 0.0

        if inst.missing_from_scan:
            flags.append("missing_physical_instrument - present in batch/register but no matching scan was extracted")
            score += 0.35

        if inst.image_quality == "unclear":
            flags.append("unclear_instrument_image")
            flags.append("low_confidence_extraction")
            flags.append("manual_review_required")
            score += 0.18

        if not inst.serial_number:
            flags.append("missing_serial_number")
            score += 0.20

        if inst.amount_numeric is None:
            flags.append("missing_amount")
            score += 0.30
        elif inst.amount_numeric >= 1000:
            flags.append(f"high_value - ${inst.amount_numeric:,.2f} at or above review threshold")
            score += 0.10

        if inst.serial_number and serial_counts[inst.serial_number] > 1 and not inst.missing_from_scan:
            flags.append("duplicate_serial_number")
            score += 0.35

        dt = _parse_iso_date(inst.issue_date)
        date_ok = True
        if dt:
            age = abs((today - dt).days)
            date_ok = age <= 120
            if not date_ok:
                flags.append("date_outside_120_days")
                score += 0.15
        else:
            flags.append("missing_issue_date")
            score += 0.12
            date_ok = False

        payee_match_score = _payee_match_score(inst.payee_raw, batch.property_name, list(batch.property_aliases or [])) if batch.property_name else 0.0
        if batch.property_name and inst.payee_raw and payee_match_score < 0.62:
            flags.append("payee_mismatch")
            score += 0.20
        elif batch.property_name and not inst.payee_raw and not inst.missing_from_scan:
            flags.append("missing_payee")
            score += 0.10

        if inst.issuer not in STANDARD_ISSUERS:
            flags.append("non_standard_issuer")
            score += 0.10

        if inst.mobile_deposit_prohibited:
            flags.append("mobile_deposit_prohibited - physical deposit required")

        grouped = split_groups.get((inst.unit, inst.issue_date), []) if inst.unit and inst.issue_date else []
        if len(grouped) >= 3:
            flags.append("split_payment_group - three or more instruments same unit/date")
            score += 0.05

        status = "VALID"
        if score >= 0.60 or inst.amount_numeric is None:
            status = "INVALID"
            items_invalid += 1
        elif score >= 0.18 or flags:
            status = "REVIEW"
            items_review += 1
        else:
            items_valid += 1
        if flags:
            items_flagged += 1

        inst.validation = {
            "overall_status": status,
            "risk_score": round(min(score, 1.0), 3),
            "payee_match_score": payee_match_score,
            "date_within_120_days": date_ok,
            "serial_duplicate": bool(inst.serial_number and serial_counts[inst.serial_number] > 1),
            "fraud_check": {"status": "PASS" if status != "INVALID" else "FAIL", "findings": flags},
            "flags": flags,
        }

    avg = round(sum(i.validation.get("risk_score", 0) for i in instruments) / max(1, len(instruments)), 3)
    overall = "ACCEPT"
    if items_invalid > 0:
        overall = "REJECT"
    elif items_review > 0 or items_flagged > 0:
        overall = "REVIEW"

    split_summary = []
    for (unit, issue_date), group in split_groups.items():
        if len(group) >= 3:
            split_summary.append(
                {
                    "unit": unit,
                    "date": issue_date,
                    "item_nos": [g.item_no for g in group],
                    "total": round(sum(g.amount_numeric or 0 for g in group), 2),
                    "note": f"{len(group)} instruments same unit/date - possible split payment",
                }
            )

    duplicate_serials = sorted([s for s, c in serial_counts.items() if c > 1])
    batch.risk_summary = {
        "average_risk_score": avg,
        "overall_decision": overall,
        "items_valid": items_valid,
        "items_review": items_review,
        "items_invalid": items_invalid,
        "items_flagged": items_flagged,
        "split_payment_groups": split_summary,
        "duplicate_serials": duplicate_serials,
    }
    batch.overall_decision = overall
