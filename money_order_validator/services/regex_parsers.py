from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


ISSUER_PATTERNS: List[Tuple[str, str, Optional[str]]] = [
    (r"WESTERN\s+UNION|\bWALMART\b|\bKROGER\b|\bH[-\s]?E[-\s]?B\b|\bCVS\b", "Western Union", None),
    (r"MONEY\s*GRAM|MONEYGRAM|CITIZENS\s+ALLIANCE", "MoneyGram", None),
    (r"INTERMEX", "Intermex", "Intermex Wire Transfer LLC"),
    (r"FIDELITY\s+EXPRESS", "Fidelity Express", "Fidelity Express"),
    (r"BARRI|DOLEX", "DolEx", "Bank of Texas"),
    (r"\bPLS\b|BANCFIRST", "PLS", "BancFirst"),
    (r"JPMORGAN|JP\s*MORGAN|CHASE", "JPMorgan Chase", None),
    (r"WELLS\s+FARGO", "Wells Fargo", None),
    (r"PROSPERITY\s+BANK", "Prosperity Bank", None),
    (r"COMERICA", "Comerica Bank", None),
]

BANK_PATTERNS: List[Tuple[str, str]] = [
    (r"J\s*P\s*MORGAN|JPMORGAN|JP\s*MORGAN|CHASE", "JPMorgan Chase Bank"),
    (r"WELLS\s+FARGO", "Wells Fargo Bank"),
    (r"REGIONS", "Regions Bank"),
    (r"BANK\s+OF\s+TEXAS", "Bank of Texas"),
    (r"MORGAN\s+CHASE", "JPMorgan Chase Bank"),
    (r"IMPERIAL\s+CHASE", "Imperial Chase"),
]

ONES = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
}
TENS = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FOURTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}


def norm_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"[ \t]+", " ", text)


def normalize_bank_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    for pat, name in BANK_PATTERNS:
        if re.search(pat, value, flags=re.IGNORECASE):
            return name
    return value.strip()


def detect_issuer(text: str) -> Tuple[Optional[str], Optional[str]]:
    for pat, issuer, agent in ISSUER_PATTERNS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return issuer, agent
    return None, None


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    # Remove max-cap language such as NOT VALID OVER $1000.00.
    s = re.sub(r"NOT\s+VALID\s+OVER\s*\$?\s*[\d,.]+", "", s, flags=re.IGNORECASE)
    m = re.search(r"[-+]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(?:[.](\d{1,2}))?", s)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    cents = m.group(2) or "00"
    if len(cents) == 1:
        cents += "0"
    try:
        val = float(f"{raw}.{cents}")
    except ValueError:
        return None
    if 0.0 <= val <= 1000000:
        return round(val, 2)
    return None


def parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s)
    if m:
        return _date_from_parts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[./|\-](\d{1,2})[./|\-](\d{2,4})\b", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int("20" + y) if len(y) == 2 else int(y)
        # Fix common OCR year 2006 -> 2026 if plausible.
        if 2000 <= year <= 2020:
            candidate = int(str(year)[:2] + "2" + str(year)[3:])
            if 2021 <= candidate <= 2035:
                year = candidate
        return _date_from_parts(year, a, b) or _date_from_parts(year, b, a)
    # Western Union internal date code: D 050926.
    m = re.search(r"\bD\s*(\d{2})(\d{2})(\d{2})\b", s, flags=re.IGNORECASE)
    if m:
        return _date_from_parts(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s.title(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(20\d{2})\b", s)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{m.group(1).title()} {m.group(2)} {m.group(3)}", fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def _date_from_parts(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_amount_from_words(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.upper()
    t = re.sub(r"[^A-Z0-9/ -]", " ", t)
    t = t.replace(" AND NO/100", " AND 00/100")
    tokens = [x for x in re.split(r"[\s-]+", t) if x and x not in {"DOLLAR", "DOLLARS", "ONLY", "EXACTLY", "PAY", "THE", "SUM", "OF", "AND"}]
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok in ONES:
            current += ONES[tok]
            seen = True
        elif tok in TENS:
            current += TENS[tok]
            seen = True
        elif tok == "HUNDRED":
            current = max(1, current) * 100
            seen = True
        elif tok == "THOUSAND":
            total += max(1, current) * 1000
            current = 0
            seen = True
        elif re.match(r"\d{2}/100", tok):
            cents = int(tok.split("/", 1)[0])
            total += current
            return round(total + cents / 100.0, 2)
        elif tok in {"CENT", "CENTS"}:
            break
    if seen:
        total += current
        m = re.search(r"(\d{1,2})\s*/\s*100", t)
        if m:
            total += int(m.group(1)) / 100.0
        elif re.search(r"NO\s+CENTS|ZERO\s+CENTS|00\s+CENTS", t):
            pass
        if 0 < total <= 100000:
            return round(float(total), 2)
    return None


def extract_labeled_amount(text: str) -> Optional[float]:
    t = norm_text(text)
    t = re.sub(r"NOT\s+VALID\s+OVER[^\n]*", "", t, flags=re.IGNORECASE)
    patterns = [
        r"PAY\s+EXACTLY[^$\n]{0,80}\$\s*([\d,]+(?:\.\d{1,2})?)",
        r"PAY\s+ONLY[^$\n]{0,50}\$\s*([\d,]+(?:\.\d{1,2})?)",
        r"\*+\s*\$?\s*([\d,]+\.\d{1,2})\s*\*+",
        r"\$\s*([\d,]+\.\d{1,2})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            val = parse_money(m.group(1))
            if val is not None and 0.01 <= val <= 100000:
                return val
    return None


def extract_serial(text: str) -> Optional[str]:
    t = norm_text(text).upper()
    candidates: List[str] = []
    # Western Union / Walmart / Kroger serials.
    candidates += [re.sub(r"\D", "", m.group(0)) for m in re.finditer(r"\b(?:19|22)[-\s]?\d{7,10}\b", t)]
    # MoneyGram explicit check/no.
    for pat in (r"CHECK\s*NO\.?\s*[:#]?\s*(\d{6,12})", r"SERIAL\s*[:#]?\s*(\d{5,12})"):
        for m in re.finditer(pat, t):
            candidates.append(m.group(1))
    # BARRI/Intermex/Fidelity vertical serials often 7-10 digits near issuer text.
    for m in re.finditer(r"\b\d{7,11}\b", t):
        raw = m.group(0)
        if raw.startswith(("102100400", "091", "111", "000")):
            continue
        candidates.append(raw)
    for c in candidates:
        c = re.sub(r"\D", "", c)
        if 5 <= len(c) <= 12:
            return c
    return None


def extract_micr(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    candidates = []
    for ln in lines:
        digits = re.sub(r"[^0-9 ]", " ", ln)
        groups = re.findall(r"\d{4,}", digits)
        if len(groups) >= 2 and sum(len(g) for g in groups) >= 14:
            candidates.append(" ".join(groups))
    if candidates:
        return max(candidates, key=len)
    return None


def extract_unit_from_text(*values: Optional[str]) -> Optional[str]:
    joined = "\n".join(v for v in values if v)
    if not joined:
        return None
    patterns = [
        r"(?:APT|APARTMENT|UNIT|SUITE|STE|#)\s*\.?\s*#?\s*([A-Z]?\d{3,5}[A-Z]?)\b",
        r"\b(?:RENT|MEMO|FOR|ACCOUNT|ACCT)\D{0,20}([A-Z]?\d{3,5}[A-Z]?)\b",
        r"\b([A-Z]?\d{3,5}[A-Z]?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, joined, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            unit = m.group(1).upper()
            if re.match(r"^\d+$", unit) and not (3 <= len(unit) <= 5):
                continue
            return unit
    return None


def normalize_serial(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip().upper()
    if s in {"NULL", "NONE", "N/A", "NA"}:
        return None
    s = s.replace("O", "0").replace("I", "1").replace("L", "1")
    s = re.sub(r"[^0-9A-Z-]", "", s)
    if re.match(r"^(19|22)-?X+$", s):
        return None
    # WU serials normalize to digits only so matching against register/MICR works.
    if re.match(r"^(19|22)-?\d{7,10}$", s):
        return re.sub(r"\D", "", s)
    return s or None


def normalize_payee(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip(" .,-")
    if not s or s.upper() in {"PAY TO THE ORDER OF", "FOR DEPOSIT ONLY", "ADDRESS", "PURCHASER"}:
        return None
    # Strip unit suffix/prefix from payee value.
    s = re.sub(r"\s+(?:APT|APARTMENT|UNIT|#)\s*\.?\s*#?\s*[A-Z]?\d{3,5}[A-Z]?\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^#\s*[A-Z]?\d{3,5}[A-Z]?\s+", "", s, flags=re.IGNORECASE)
    return s.strip(" .,-") or None


def similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    aa = re.sub(r"[^A-Z0-9]", "", a.upper())
    bb = re.sub(r"[^A-Z0-9]", "", b.upper())
    if not aa or not bb:
        return 0.0
    return round(SequenceMatcher(None, aa, bb).ratio(), 3)


def sanitize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "visible"}


def sanitize_instrument(raw: Dict[str, Any], ocr_text: str = "") -> Dict[str, Any]:
    out = dict(raw or {})
    issuer, agent = detect_issuer("\n".join([str(out.get("issuer") or ""), ocr_text]))
    if issuer:
        out["issuer"] = issuer
        out["issuer_agent"] = out.get("issuer_agent") or agent

    inst_type = out.get("instrument_type") or ""
    joined = f"{ocr_text}\n{out.get('issuer') or ''}\n{out.get('payment_description') or ''}".upper()
    if not inst_type:
        if "CASHIER" in joined or "OFFICIAL CHECK" in joined:
            inst_type = "CashiersCheck"
        elif "CHECK" in joined and "MONEY" not in joined:
            inst_type = "Check"
        else:
            inst_type = "MoneyOrder"
    if inst_type not in {"MoneyOrder", "Check", "CashiersCheck", "Escrow"}:
        inst_type = "MoneyOrder"
    out["instrument_type"] = inst_type

    if not out.get("payment_description"):
        if inst_type == "Escrow":
            out["payment_description"] = "Escrow Deposit Paid In"
        elif inst_type in {"Check", "CashiersCheck"}:
            out["payment_description"] = "Payment-Check"
        else:
            out["payment_description"] = "Payment-MoneyOrder"

    out["serial_number"] = normalize_serial(out.get("serial_number")) or extract_serial(ocr_text)
    out["issue_date"] = parse_date(out.get("issue_date")) or parse_date(ocr_text)

    amt = parse_money(out.get("amount_numeric"))
    if amt is None:
        amt = extract_labeled_amount(ocr_text)
    words_amt = parse_amount_from_words(out.get("amount_words"))
    if words_amt is not None and (amt is None or amt <= 0 or amt / words_amt < 0.80 or amt / words_amt > 1.25):
        amt = words_amt
    out["amount_numeric"] = amt

    out["payee_raw"] = normalize_payee(out.get("payee_raw"))
    out["unit"] = extract_unit_from_text(out.get("unit"), out.get("payment_for_acct"), out.get("payer_address"), out.get("payee_raw"), ocr_text)

    for key in ("payer_name", "payer_address", "amount_words", "payment_for_acct", "micr_line", "issuer_agent"):
        val = out.get(key)
        if isinstance(val, str):
            val = re.sub(r"\s+", " ", val).strip()
            if val.upper() in {"NULL", "NONE", "N/A", "NA", "PURCHASER", "ADDRESS", "REMITTER"}:
                val = None
        out[key] = val or None

    if not out.get("micr_line"):
        out["micr_line"] = extract_micr(ocr_text)

    out["payer_signature"] = sanitize_bool(out.get("payer_signature"))
    out["mobile_deposit_prohibited"] = sanitize_bool(out.get("mobile_deposit_prohibited")) or bool(re.search(r"MOBILE\s+DEPOSIT\s+PROHIBITED|NOT\s+FOR\s+MOBILE\s+DEPOSIT", ocr_text or "", re.IGNORECASE))
    out["watermark_present"] = sanitize_bool(out.get("watermark_present"))

    return out


def parse_basic_instrument_from_ocr(text: str) -> Optional[Dict[str, Any]]:
    issuer, agent = detect_issuer(text)
    serial = extract_serial(text)
    amount = extract_labeled_amount(text)
    issue_date = parse_date(text)
    if not any([issuer, serial, amount, issue_date]):
        return None
    inst_type = "MoneyOrder" if issuer or re.search(r"MONEY\s*ORDER", text or "", re.IGNORECASE) else "Check"
    return sanitize_instrument(
        {
            "instrument_type": inst_type,
            "issuer": issuer,
            "issuer_agent": agent,
            "serial_number": serial,
            "amount_numeric": amount,
            "issue_date": issue_date,
            "micr_line": extract_micr(text),
        },
        text,
    )


def parse_batch_header(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    result: Dict[str, Any] = {}

    def after_label(label_pat: str, stop: str = r"\n|Batch\s+|Bank\s+|Account\s+|Actual\s+|Line\s+|Period\s+|Printed\s+|Deposit\s+") -> Optional[str]:
        m = re.search(r"(?:" + label_pat + r")\s*:?[ \t]*([^\n]+)", t, flags=re.IGNORECASE)
        if not m:
            return None
        value = re.split(stop, m.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        return value.strip(" :-") or None

    m = re.search(r"\bBatch\s*#\s*:?[ \t]*([0-9]{6,12})\b", t, flags=re.IGNORECASE)
    if m:
        result["batch_number"] = m.group(1)

    for key, pat in [
        ("batch_type", r"Batch\s+Type"),
        ("batch_status", r"Batch\s+Status"),
        ("pay_period", r"Period"),
    ]:
        val = after_label(pat)
        if val:
            result[key] = val

    bank = after_label(r"Bank\s+Name") or after_label(r"Bank")
    if bank:
        result["bank_name"] = normalize_bank_name(bank)

    acct = after_label(r"Account\s*#|Account\s+Number|Deposit\s+Account")
    if acct:
        nums = re.findall(r"\d{4,}", acct)
        if nums:
            # Avoid returning 9-digit routing number when there is a later account group.
            result["account_number"] = nums[-1]

    total = None
    for pat in [r"Actual\s+Items\s*:?[ \t]*(\d+)", r"Line\s+Items\s*:?[ \t]*(\d+)", r"Total\s+Items\s*:?[ \t]*(\d+)", r"Totals?\s*\((\d+)\)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            total = int(m.group(1))
            break
    if total is not None:
        result["total_items"] = total

    for pat in [
        r"Batch\s+Amount\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
        r"Deposit\s+Amount\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
        r"Deposit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
        r"Totals?\s*\(\d+\).*?\$?\s*([\d,]+\.\d{2})",
    ]:
        m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            result["batch_amount"] = parse_money(m.group(1))
            break

    m = re.search(r"Printed\s+On\s*:?[ \t]*([^\n]+)", t, flags=re.IGNORECASE)
    if m:
        result["printed_on"] = parse_date(m.group(1))
    if not result.get("printed_on"):
        result["printed_on"] = parse_date(t)

    m = re.search(r"Transaction\s+(?:ID|#)\s*:?[ \t]*([0-9]{5,})", t, flags=re.IGNORECASE)
    if m:
        result["deposit_transaction"] = m.group(1)

    # Property name/address heuristic from first lines.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for i, line in enumerate(lines[:8]):
        if re.search(r"BATCH\s+DETAIL|DEPOSIT\s+BATCH|YOTTAREAL|PRINTED\s+ON", line, re.IGNORECASE):
            continue
        if re.search(r"\b[A-Z][A-Z .'-]{3,}\b", line) and not re.search(r"BANK|ACCOUNT|BATCH|PERIOD", line, re.IGNORECASE):
            result.setdefault("property_name", re.sub(r"\s+", " ", line).title())
            if i + 1 < len(lines) and re.search(r"\b[A-Z]{2}\s+\d{5}\b", lines[i + 1], re.IGNORECASE):
                result.setdefault("property_address", lines[i + 1].title())
            break

    return {k: v for k, v in result.items() if v not in (None, "")}


def parse_deposit_info(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    out: Dict[str, Any] = {}
    if m := re.search(r"Transaction\s+(?:ID|#)\s*:?[ \t]*([0-9]{5,})", t, flags=re.IGNORECASE):
        out["deposit_transaction"] = m.group(1)
    if m := re.search(r"Deposit\s+Account\s*:?[ \t]*([0-9]{4,})(?:\s*-\s*([^\n]+))?", t, flags=re.IGNORECASE):
        out["deposit_account"] = m.group(1)
        if m.group(2):
            out["account_name"] = m.group(2).strip()
    for key, pat in [
        ("deposit_amount", r"Deposit\s+Amount\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_total", r"Deposit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("check_total", r"Checks?\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("credit_total", r"Credit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("debit_total", r"Debit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("difference", r"Difference\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("cash_back", r"Cash\s+Back\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
    ]:
        if m := re.search(pat, t, flags=re.IGNORECASE):
            out[key] = parse_money(m.group(1))
    if m := re.search(r"Item\s+Count\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Report\s+Time\s*:?[ \t]*(20\d{2}[-/]\d{1,2}[-/]\d{1,2})", t, flags=re.IGNORECASE):
        out["deposit_date"] = parse_date(m.group(1))
    return out


def parse_batch_line_items(text: str) -> List[Dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    items: List[Dict[str, Any]] = []
    # Single-line YottaReal register rows.
    row_pat = re.compile(
        r"^(?P<unit>[A-Z]?\d{1,5}[A-Z]?)\s+"
        r"(?P<resident>.+?)\s+"
        r"(?P<desc>Payment[- ](?:MoneyOrder|Check)|Escrow[^\s]*)\s+"
        r"(?P<serial>[A-Z0-9-]{4,14})\s+"
        r"(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+"
        r"\$?(?P<amount>[\d,]+\.\d{2})"
        r"(?:\s+(?P<posted_by>.+))?$",
        flags=re.IGNORECASE,
    )
    for line in lines:
        m = row_pat.match(line)
        if not m:
            continue
        serial = normalize_serial(m.group("serial"))
        items.append(
            {
                "unit": m.group("unit"),
                "resident_name": m.group("resident").strip(),
                "payment_description": m.group("desc").replace(" ", "-"),
                "serial_number": serial,
                "posted_date": parse_date(m.group("date")),
                "amount_numeric": parse_money(m.group("amount")),
                "posted_by": (m.group("posted_by") or "").strip() or None,
            }
        )
    return items
