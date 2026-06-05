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
    s = str(value).strip()
    if not s:
        return None
    # Remove max-cap language such as NOT VALID OVER $1000.00.
    s = re.sub(r"NOT\s+VALID\s+OVER\s*\$?\s*[\d,.]+", "", s, flags=re.IGNORECASE)

    # OCR often drops the decimal point in the amount box: "$356 56" or "$356,56".
    # Treat these as dollars+cents only when the text is clearly amount-like.
    amount_like = bool(re.search(r"\$|DOLLARS?|CENTS?|PAY\s+EXACTLY|PAY\s+ONLY|AMOUNT", s, re.IGNORECASE))
    m_sep = re.search(r"(?:\$\s*)?\b([0-9]{1,5})\s+([0-9]{2})\b", s)
    if m_sep and (amount_like or re.fullmatch(r"\s*[0-9]{1,5}\s+[0-9]{2}\s*", s)):
        dollars = int(m_sep.group(1).replace(",", ""))
        cents_i = int(m_sep.group(2))
        if 0 <= cents_i <= 99 and 0 <= dollars <= 100000:
            return round(dollars + cents_i / 100.0, 2)

    # Decimal comma in OCR: "$356,56" should be 356.56, but "1,234" is thousands.
    m_comma_decimal = re.search(r"(?:\$\s*)?\b([0-9]{1,5}),([0-9]{2})\b", s)
    if m_comma_decimal and (amount_like or re.fullmatch(r"\s*[0-9]{1,5},[0-9]{2}\s*", s)):
        dollars = int(m_comma_decimal.group(1))
        cents_i = int(m_comma_decimal.group(2))
        if 0 <= cents_i <= 99 and 0 <= dollars <= 100000:
            return round(dollars + cents_i / 100.0, 2)

    m = re.search(r"[-+]?\$?\s*((?:[0-9]{1,3}(?:,[0-9]{3})+)|[0-9]+)(?:[.](\d{1,2}))?", s)
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


def _parse_cardinal_words(value: str) -> Optional[int]:
    """Parse simple English cardinal words up to hundreds of thousands."""
    if not value:
        return None
    value = re.sub(r"[^A-Z0-9 -]", " ", value.upper())
    tokens = [tok for tok in re.split(r"[\s-]+", value) if tok]
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok in {"AND", "DOLLAR", "DOLLARS", "CENT", "CENTS", "ONLY", "NO"}:
            continue
        if tok == "ZERO":
            seen = True
            continue
        if tok.isdigit():
            current += int(tok)
            seen = True
        elif tok in ONES:
            current += ONES[tok]
            seen = True
        elif tok in TENS:
            current += TENS[tok]
            seen = True
        elif tok == "HUNDRED":
            current = max(current, 1) * 100
            seen = True
        elif tok == "THOUSAND":
            total += max(current, 1) * 1000
            current = 0
            seen = True
    if not seen:
        return None
    return total + current


def parse_amount_from_words(text: Optional[str]) -> Optional[float]:
    """Parse written check/MO amounts.

    Separates the dollar segment from the cents segment so cents are not counted
    as extra dollars. Handles common handwritten/OCR variants:
      - "FOURTEEN HUNDRED NINETY NINE 74" -> 1499.74
      - "TWO HUNDRED SIX AND 17 DOLLARS" -> 206.17
      - "THIRTEEN HUNDRED TWENTY THREE DOLLARS 74/XX" -> 1323.74
    """
    if not text:
        return None
    t = str(text).upper()
    t = t.replace("NO/100", "00/100")
    # OCR sometimes reads /100 as /XX on handwritten checks.
    t = re.sub(r"/\s*(?:XX|X{2})\b", "/100", t, flags=re.IGNORECASE)
    t = re.sub(r"[^A-Z0-9/ .'-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None

    cents = 0
    dollar_part = t

    # Numeric cents: 04/100, 74/XX->74/100, 09 CENTS, 9 CENTS.
    m = re.search(r"\b(\d{1,2})\s*/\s*100\b", t)
    if m:
        cents = int(m.group(1))
        dollar_part = t[:m.start()]
    else:
        # Handwriting/OCR: "Two Hundred Six and 17 Dollars" — cents before DOLLARS label.
        m = re.search(r"\bAND\s+(\d{1,2})\s+DOLLARS?\b", t)
        if m:
            cents = int(m.group(1))
            dollar_part = t[:m.start()]
        else:
            m = re.search(r"\b(\d{1,2})\s+CENTS?\b", t)
            if m:
                cents = int(m.group(1))
                dollar_part = t[:m.start()]
            elif re.search(r"\bCENTS?\b", t):
                cents_head = t[: re.search(r"\bCENTS?\b", t).start()]
                m_words = re.search(r"\bDOLLARS?\b\s*(?:AND\s+)?([A-Z -]+)$", cents_head)
                if m_words:
                    cents_words = m_words.group(1).strip()
                    dollar_part = cents_head[: m_words.start()]
                else:
                    m_words = re.search(r"\bAND\s+([A-Z -]+)$", cents_head)
                    if m_words:
                        cents_words = m_words.group(1).strip()
                        dollar_part = cents_head[: m_words.start()]
                    else:
                        cents_words = cents_head.strip()
                        dollar_part = ""
                if re.fullmatch(r"(?:NO|ZERO|00)", cents_words.strip()):
                    cents = 0
                else:
                    parsed_cents = _parse_cardinal_words(cents_words)
                    cents = parsed_cents if parsed_cents is not None else 0
            else:
                # Bare trailing two-digit cents with no DOLLARS marker, e.g.
                # "FOURTEEN HUNDRED NINETY NINE 74".
                m = re.search(r"\b(?:AND\s+)?(\d{1,2})\s*$", t)
                if m and re.search(r"[A-Z]", t[:m.start()]):
                    cents = int(m.group(1))
                    dollar_part = t[:m.start()]

    m_dollars = re.search(r"\bDOLLARS?\b", dollar_part)
    if m_dollars:
        dollar_part = dollar_part[:m_dollars.start()]
    dollar_part = re.sub(r"\b(PAY|EXACTLY|ONLY|THE|SUM|OF|AMOUNT|PAYABLE)\b", " ", dollar_part)
    dollar_part = re.sub(r"\bAND\s*$", " ", dollar_part).strip()

    dollars = _parse_cardinal_words(dollar_part)
    if dollars is None:
        return None
    if not (0 <= cents <= 99):
        return None
    total = round(float(dollars) + cents / 100.0, 2)
    if 0 < total <= 100000:
        return total
    return None


def extract_labeled_amount(text: str) -> Optional[float]:
    t = norm_text(text)
    t = re.sub(r"NOT\s+VALID\s+OVER[^\n]*", "", t, flags=re.IGNORECASE)
    patterns = [
        # OCR/vision sometimes reads the amount box as "$356 56" or "$356,56".
        # Put these before whole-dollar patterns so cents are not dropped.
        r"PAY\s+EXACTLY[^$\n]{0,80}\$\s*([0-9]{1,5}\s+[0-9]{2})",
        r"PAY\s+ONLY[^$\n]{0,50}\$\s*([0-9]{1,5}\s+[0-9]{2})",
        r"\$\s*([0-9]{1,5}\s+[0-9]{2})(?!\d)",
        r"\$\s*([0-9]{1,5},[0-9]{2})(?!\d)",
        # Normal decimal amount.
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
    labeled_amt = extract_labeled_amount(ocr_text)
    if labeled_amt is not None:
        if amt is None:
            amt = labeled_amt
        else:
            # If the model/OCR captured only whole dollars but the labeled amount box
            # has cents, restore the cents. Example: model read 356, box reads 356.56.
            same_dollar_part = int(float(amt)) == int(float(labeled_amt))
            amt_has_no_cents = abs(float(amt) - int(float(amt))) < 0.005
            labeled_has_cents = abs(float(labeled_amt) - int(float(labeled_amt))) >= 0.005
            if same_dollar_part and amt_has_no_cents and labeled_has_cents:
                out.setdefault("corrections", []).append(
                    {
                        "field": "amount_numeric",
                        "old": amt,
                        "new": labeled_amt,
                        "source": "labeled_amount_cents",
                    }
                )
                amt = labeled_amt
    words_amt = parse_amount_from_words(out.get("amount_words"))
    # Written amount is the legal amount on checks/MOs. Prefer parseable words over any numeric read.
    if words_amt is not None and (amt is None or abs(float(amt) - float(words_amt)) >= 0.005):
        out.setdefault("corrections", []).append(
            {
                "field": "amount_numeric",
                "old": amt,
                "new": words_amt,
                "source": "amount_words",
            }
        )
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
    # Trust only the explicit printed phrase for this flag. Vision models sometimes infer this
    # on ordinary money orders/checks, which creates noisy risk flags.
    mdp_text = "\n".join([ocr_text or "", str(out.get("mobile_deposit_prohibited") or "")])
    out["mobile_deposit_prohibited"] = bool(
        re.search(r"MOBILE\s+DEPOSIT\s+PROHIBITED|NOT\s+FOR\s+MOBILE\s+DEPOSIT", mdp_text, re.IGNORECASE)
    )
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


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" :-")


def is_deposit_detail_report(text: str) -> bool:
    """Detect bank Deposit Detail Report pages with transaction item rows.

    These reports already contain authoritative transaction id, account, row amounts,
    and row serial/account values. Thumbnail images on the report must not be treated
    as separate physical instruments.
    """
    t = norm_text(text).upper()
    return bool(
        re.search(r"\bDEPOSIT\s+DETAIL\s+REPORT\b", t)
        or (
            re.search(r"\bDEPOSIT\s+DETAIL\s+FOR\s+DEPOSIT\s+ID\b", t)
            and re.search(r"\bTRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION\s+ID\b", t)
        )
        or (
            re.search(r"\bDEPOSIT\s+CONTROL\s+INFORMATION\b", t)
            and re.search(r"\bTRANSACTION\s+CONTROL\s+INFORMATION\b", t)
        )
    )


def parse_deposit_detail_report_header(text: str) -> Dict[str, Any]:
    """Parse Deposit Detail Report header/summary fields.

    This report type has one logical deposit spread across multiple pages.
    Per-page item rows must not be aggregated as separate deposit slips.
    """
    if not is_deposit_detail_report(text):
        return {}
    t = norm_text(text)
    out: Dict[str, Any] = {
        "source_system": "Deposit Detail Report",
        "bank_name": "JPMorgan Chase Bank",
    }

    if m := re.search(r"Deposit\s+Detail\s+for\s+Deposit\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["deposit_id"] = m.group(1)
    if m := re.search(r"Transaction\s+Detail\s+for\s+Transaction\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["deposit_transaction"] = m.group(1)
    if m := re.search(r"Batch\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["batch_number"] = m.group(1)
    if m := re.search(r"Processing\s+Date\s*:?\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})", t, flags=re.IGNORECASE):
        out["deposit_date"] = parse_date(m.group(1))
        out["deposited_date"] = out["deposit_date"]
        out["printed_on"] = out["deposit_date"]
    if m := re.search(r"Account\s+Name\s*:?\s*([^\n]+?)(?=\s+Location\s+ID\b|\s+Transaction\s+Detail\b|$)", t, flags=re.IGNORECASE):
        name = _clean_property_name(m.group(1))
        if name:
            out["account_name"] = name
            out["property_name"] = name
            out["property_aliases"] = build_property_aliases(name)
    if m := re.search(r"Deposit\s+Account\s*:?\s*(\d{4,})(?:\s*-\s*([^\n]+?))?(?=\s+Partnership\b|\s+AUX|$)", t, flags=re.IGNORECASE):
        out["deposit_account"] = m.group(1)
        out["account_number"] = m.group(1)
        if m.group(2):
            name = _clean_property_name(m.group(2))
            if name:
                out.setdefault("account_name", name)
                out.setdefault("property_name", name)
    for key, pat in (
        ("deposit_total", r"Deposit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("check_total", r"Checks?\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("credit_total", r"Credit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("debit_total", r"Debit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_amount", r"Deposit\s+Amount\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
    ):
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = parse_money(m.group(1))
            if val is not None:
                out[key] = val
    if out.get("deposit_total") is not None:
        out.setdefault("deposit_amount", out["deposit_total"])
        out.setdefault("check_total", out["deposit_total"])
    for key, pat in (
        ("credit_items", r"Credit\s+Items\s*:?\s*(\d{1,4})"),
        ("debit_items", r"Debit\s+Items\s*:?\s*(\d{1,4})"),
        ("report_item_count", r"Item\s+Count\s*:?\s*(\d{1,4})"),
    ):
        if m := re.search(pat, t, flags=re.IGNORECASE):
            try:
                out[key] = int(m.group(1))
            except ValueError:
                pass
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _clean_property_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = _one_line(value)
    s = re.sub(r"\bAccount\s+Currency\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNumber\s+of\s+Deposits\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTotal\s+of\s+Deposits\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"/\s*\d{4,}\b.*$", "", s)
    s = s.strip(" -:/")
    if not s or re.fullmatch(r"REGIONS|CHASE|JPMORGAN|BANK", s, flags=re.IGNORECASE):
        return None
    return s


def build_property_aliases(property_name: Optional[str]) -> List[str]:
    """Return match aliases for property/payee validation."""
    if not property_name:
        return []
    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        value = _clean_property_name(value)
        if not value:
            return
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        seen = {re.sub(r"[^A-Z0-9]", "", x.upper()) for x in candidates}
        if compact and compact not in seen:
            candidates.append(value)

    add(property_name)
    m = re.search(r"\bd\s*/?\s*b\s*/?\s*a\b\s+(.+)$|\bdba\b\s+(.+)$", property_name, flags=re.IGNORECASE)
    alias_base = None
    if m:
        alias_base = m.group(1) or m.group(2)
        add(alias_base)
    else:
        alias_base = property_name

    if alias_base:
        alias_base = _clean_property_name(alias_base) or alias_base
        no_suffix = re.sub(r"\b(APARTMENTS?|APTS?|VILLAGE|LP|LLC|LTD|LIMITED\s+PARTNERSHIP)\b", "", alias_base, flags=re.IGNORECASE)
        add(no_suffix)
        for sep in (" in ", " at ", " dba "):
            if sep in alias_base.lower():
                add(re.split(sep, alias_base, flags=re.IGNORECASE)[0])
        words = re.findall(r"[A-Za-z0-9]+", alias_base)
        if len(words) >= 2:
            add(" ".join(words[:2]))
        if words and len(words[0]) >= 5:
            add(words[0])
    return candidates


def is_regions_deposit_report(text: str) -> bool:
    """Detect Regions "Details of Deposits by Account" reports.

    These pages look like a payment register but are not payment instruments. If
    they are misrouted to vision extraction, routing/account/check values become
    fake instruments.  Keep this detector deterministic and conservative.
    """
    t = norm_text(text).upper()
    signals = [
        r"\bDETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT\b",
        r"\bTOTAL\s+OF\s+DEPOSITS\s+SUBMITTED\b",
        r"\bTOTAL\s+NUMBER\s+OF\s+ITEMS\b",
        r"\bACCOUNT\s+NAME/NUMBER\b",
        r"\bCAPTURE\s+SEQ\b",
        r"\bPOST\s+AMOUNT\b",
        r"\bCREDIT\s+AMOUNT\b",
        r"\bDEPOSIT\s+NUMBER\b",
    ]
    return sum(1 for pat in signals if re.search(pat, t, flags=re.IGNORECASE)) >= 2


def _clean_regions_name(value: str) -> Optional[str]:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip(" -:/")
    value = re.split(
        r"\b(?:NUMBER\s+OF\s+DEPOSITS|TOTAL\s+OF\s+DEPOSITS|TOTAL\s+NUMBER|ACCOUNT\s+CURRENCY|USD|DEPOSIT\s+NUMBER)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" -:/")
    return value or None


def parse_regions_deposit_header(text: str) -> Dict[str, Any]:
    """Parse Regions bank deposit-register header fields.

    Example page text contains:
      Account Name/Number: Raja Bata LP dba Arella Forest in Woodland/0293323581
      Total of Deposits Submitted: 25,213.12
      Total Number of Items: 17
    """
    if not is_regions_deposit_report(text):
        return {}

    t = norm_text(text)
    out: Dict[str, Any] = {
        "bank_name": "Regions Bank",
        "source_system": "Regions",
    }

    # The account name can wrap before the slash/account number.
    m = re.search(
        r"Account\s+Name/Number\s*:\s*(?P<name>.*?)/\s*(?P<acct>\d{4,})",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        name = _clean_regions_name(m.group("name"))
        if name:
            out["property_name"] = name
        out["account_number"] = m.group("acct")

    if not out.get("property_name"):
        m = re.search(
            r"Details\s+of\s+Deposits\s+by\s+Account\s*-\s*(?P<name>.+?)\s*-",
            t,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            name = _clean_regions_name(m.group("name"))
            if name:
                out["property_name"] = name

    for key, pat in [
        ("batch_amount", r"Total\s+of\s+Deposits\s+Submitted\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_amount", r"Total\s+of\s+Deposits\s+Submitted\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_total", r"Deposit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
    ]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            out[key] = parse_money(m.group(1))

    m = re.search(r"Total\s+Number\s+of\s+Items\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE)
    if m:
        out["total_items"] = int(m.group(1))
        out["item_count"] = int(m.group(1))

    # Regions deposit number appears in the summary row below the header.
    m = re.search(
        r"Deposit\s+Number\s+Item\s+Count.*?\n\s*(\d{4,})\s+\d+\s+[\d,]+\.\d{2}",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["deposit_transaction"] = m.group(1)

    # Prefer the actual deposit date row if present; otherwise any report date is acceptable.
    m = re.search(r"Deposit\s+Date.*?(\d{1,2}/\d{1,2}/20\d{2})", t, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"\b(\d{1,2}/\d{1,2}/20\d{2})\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\b", t, flags=re.IGNORECASE)
    if m:
        out["deposited_date"] = parse_date(m.group(1))
        out.setdefault("printed_on", parse_date(m.group(1)))

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def parse_regions_deposit_items(text: str) -> List[Dict[str, Any]]:
    """Parse Regions table rows into authoritative register items."""
    if not is_regions_deposit_report(text):
        return []

    items: List[Dict[str, Any]] = []
    row_re = re.compile(
        r"^\s*(?P<capture_seq>\d{6})\s+"
        r"(?P<routing_number>\d{9})\s+"
        r"(?P<account_number>\d{4,20})\s+"
        r"(?P<check_number>\d{1,12})\s+"
        r"(?P<post_amount>[\d,]+\.\d{2})\s+"
        r"(?P<credit_amount>[\d,]+\.\d{2})"
        r"(?:\s+(?P<adjustment>[\d,]+\.\d{2}))?\s*$",
        flags=re.IGNORECASE,
    )
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        m = row_re.match(line)
        if not m:
            continue
        raw_check = m.group("check_number")
        serial = raw_check if len(raw_check) <= 5 else (raw_check.lstrip("0") or raw_check)
        amount = parse_money(m.group("post_amount"))
        payment_description = "Payment-MoneyOrder" if len(serial) >= 6 else "Payment-Check"
        items.append(
            {
                "item_no": int(m.group("capture_seq")),
                "serial_number": serial,
                "check_number": raw_check,
                "routing_number": m.group("routing_number"),
                "drawee_account_number": m.group("account_number"),
                "amount_numeric": amount,
                "credit_amount": parse_money(m.group("credit_amount")),
                "adjustment": parse_money(m.group("adjustment")) if m.group("adjustment") else 0.0,
                "payment_description": payment_description,
                "source": "regions_deposit_report",
            }
        )
    return items


def parse_transaction_detail_items(text: str) -> List[Dict[str, Any]]:
    t = norm_text(text)
    if not re.search(r"Transaction\s+Detail\s+for\s+Transaction|Deposit\s+Control\s+Information", t, flags=re.IGNORECASE):
        return []
    items: List[Dict[str, Any]] = []
    row_re = re.compile(
        r"^\s*(?:(?P<aux>[A-Z0-9-]{4,14})\s+)?"
        r"(?P<routing>\d{9})\s+"
        r"(?P<account>\d{4,20})"
        r"(?:\s+(?P<check>\d{1,12}))?\s+"
        r"\$?\s*(?P<amount>[\d,]+\.\d{2})\s+"
        r"(?P<item_type>\d{3,4}|Credit|Debit)?\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    item_no = 0
    for m in row_re.finditer(t):
        line = m.group(0)
        if re.search(r"\bCredit\b", line, flags=re.IGNORECASE):
            continue
        amount = parse_money(m.group("amount"))
        if amount is None or amount <= 0 or amount > 5000:
            continue
        serial = normalize_serial(m.group("aux") or m.group("check"))
        acct = m.group("account")
        if not serial:
            # Western Union/Kroger report account values often look like
            # 40198336200716: 40 + printed 19XXXXXXXXX serial + check digit.
            # Keep the readable 19/22 serial when present instead of the full MICR tail.
            serial_match = re.search(r"(?:40)?((?:19|22)\d{7,10})\d?$", acct or "")
            if serial_match:
                serial = normalize_serial(serial_match.group(1))
            else:
                serial = normalize_serial(acct[-10:]) if len(acct) >= 9 else None
        item_no += 1
        items.append(
            {
                "item_no": item_no,
                "routing_number": m.group("routing"),
                "account_number": m.group("account"),
                "check_number": m.group("check"),
                "serial_number": serial,
                "amount_numeric": amount,
                "payment_description": "Payment-MoneyOrder",
                "instrument_type": "MoneyOrder",
                "source": "transaction_detail_report",
                "source_system": "Deposit Detail Report",
            }
        )
    return items


def parse_batch_header(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    result: Dict[str, Any] = {}
    _regions = parse_regions_deposit_header(t)
    if _regions:
        result.update(_regions)

    deposit_detail = parse_deposit_detail_report_header(t)
    if deposit_detail:
        # Keep deposit-detail fields but do not let generic header OCR overwrite
        # key values with labels like Worktype as the property.
        result.update({k: v for k, v in deposit_detail.items() if k not in {"deposit_amount", "deposit_total", "check_total", "credit_total", "debit_total", "report_item_count"}})
        if deposit_detail.get("deposit_total") is not None:
            result["batch_amount"] = deposit_detail["deposit_total"]

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
    if bank and not result.get("bank_name"):
        result["bank_name"] = normalize_bank_name(bank)

    acct = after_label(r"Account\s*#|Account\s+Number|Deposit\s+Account")
    if acct and not result.get("account_number"):
        nums = re.findall(r"\d{4,}", acct)
        if nums:
            result["account_number"] = nums[-1]

    total = None
    for pat in [
        r"Actual\s+Items\s*:?[ \t]*(\d+)",
        r"Line\s+Items\s*:?[ \t]*(\d+)",
        r"Total\s+Items\s*:?[ \t]*(\d+)",
        r"Total\s+Number\s+of\s+Items\s*:?[ \t]*(\d+)",
        r"Totals?\s*\((\d+)\)",
    ]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            total = int(m.group(1))
            break
    if total is not None:
        result["total_items"] = total

    if not result.get("batch_amount"):
        for pat in [
            r"Batch\s+Amount\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
            r"Total\s+of\s+Deposits\s+Submitted\s*:?\s*\$?\s*([\d,]+\.\d{2})",
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

    if not result.get("property_name"):
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for i, line in enumerate(lines[:10]):
            if re.search(r"BATCH\s+DETAIL|DEPOSIT\s+BATCH|DETAILS\s+OF\s+DEPOSITS|YOTTAREAL|PRINTED\s+ON|REGIONS|CHASE|JPMORGAN", line, re.IGNORECASE):
                continue
            if re.search(r"\b[A-Z][A-Z .'-]{3,}\b", line) and not re.search(r"BANK|ACCOUNT|BATCH|PERIOD|REPORT|TOTAL", line, re.IGNORECASE):
                name = _clean_property_name(line)
                if name:
                    result.setdefault("property_name", name.title())
                    result.setdefault("property_aliases", build_property_aliases(name))
                if i + 1 < len(lines) and re.search(r"\b[A-Z]{2}\s+\d{5}\b", lines[i + 1], re.IGNORECASE):
                    result.setdefault("property_address", lines[i + 1].title())
                break

    if result.get("property_name") and not result.get("property_aliases"):
        result["property_aliases"] = build_property_aliases(result.get("property_name"))

    return {k: v for k, v in result.items() if v not in (None, "", [])}


def parse_deposit_info(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    out: Dict[str, Any] = {}

    deposit_detail = parse_deposit_detail_report_header(t)
    if deposit_detail:
        out.update(deposit_detail)

    deposit_context = bool(
        re.search(
            r"MY\s+TRANSACTION\s+SUMMARY|CHECKING\s+DEPOSIT|COMMERCIAL\s+DEPOSIT|"
            r"DEPOSIT\s+ACCOUNT|DEPOSIT\s+TOTAL|CHECKS?\s+TOTAL|ACCOUNT\s+NUMBER\s+ENDING\s+IN|"
            r"TRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION|DEPOSIT\s+CONTROL\s+INFORMATION|"
            r"DEPOSIT\s+TICKET|TOTAL\s+ITEMS|FOR\s+CASH\s+DEPOSIT|"
            r"CHECKS\s+AND\s+OTHER\s+ITEMS\s+ARE\s+RECEIVED\s+FOR\s+DEPOSIT|"
            r"DETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT|ACCOUNT\s+NAME/NUMBER",
            t,
            flags=re.IGNORECASE,
        )
    )

    if deposit_context and re.search(r"CHASE|JPMORGAN", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "JPMorgan Chase Bank")
        out.setdefault("source_system", "Chase")

    # Chase ATM/teller receipt parsing.
    if re.search(r"MY\s+TRANSACTION\s+SUMMARY|CHECKING\s+DEPOSIT|COMMERCIAL\s+DEPOSIT", t, flags=re.IGNORECASE):
        for pat in (
            r"(?:CHECKING|SAVINGS|COMMERCIAL)?\s*DEPOSIT\s*\$?\s*([\d,]+\.\d{2})",
            r"DEPOSIT\s+AMOUNT\s*:?\s*\$?\s*([\d,]+\.\d{2})",
        ):
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                amount = parse_money(m.group(1))
                if amount is not None:
                    out.setdefault("deposit_amount", amount)
                    out.setdefault("deposit_total", amount)
                    out.setdefault("check_total", amount)
                break
        if m := re.search(r"ACCOUNT\s+NUMBER\s+ENDING\s+IN\s*:?\s*(\d{2,6})", t, flags=re.IGNORECASE):
            out.setdefault("account_last4", m.group(1)[-4:])
            out.setdefault("deposit_account", m.group(1)[-4:])
        if m := re.search(r"TRANSACTION\s*#\s*:?\s*(\d{1,12})", t, flags=re.IGNORECASE):
            out.setdefault("deposit_transaction", m.group(1))
        if m := re.search(r"BUSINESS\s+DATE\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})", t, flags=re.IGNORECASE):
            out.setdefault("deposit_date", parse_date(m.group(1)))

    # Deposit tickets often carry the property/legal entity near the top.
    if m := re.search(r"\bDBA\s+([A-Z0-9 .&'/-]+?)\s+OPERATING\s+ACCOUNT\b", t, flags=re.IGNORECASE):
        prop = _clean_property_name(m.group(1))
        if prop:
            out.setdefault("account_name", prop.title())

    _regions = parse_regions_deposit_header(t)
    if _regions:
        if _regions.get("batch_amount") is not None:
            out["deposit_total"] = _regions["batch_amount"]
            out["check_total"] = _regions["batch_amount"]
        if _regions.get("total_items") is not None:
            out["item_count"] = _regions["total_items"]
        for src, dst in [
            ("account_number", "deposit_account"),
            ("property_name", "account_name"),
            ("deposited_date", "deposit_date"),
            ("deposit_transaction", "deposit_transaction"),
            ("bank_name", "bank_name"),
            ("source_system", "source_system"),
        ]:
            if _regions.get(src) is not None:
                out[dst] = _regions[src]

    if m := re.search(r"Transaction\s+(?:ID|#)\s*:?[ \t]*([0-9]{2,})", t, flags=re.IGNORECASE):
        out["deposit_transaction"] = m.group(1)
    if m := re.search(r"Deposit\s+Account\s*:?[ \t]*([0-9]{4,})(?:\s*-\s*([^\n]+))?", t, flags=re.IGNORECASE):
        out["deposit_account"] = m.group(1)
        if m.group(2):
            out["account_name"] = m.group(2).strip()
    if m := re.search(r"Account\s+Number\s+Ending\s+In\s*:?[ \t]*([0-9]{3,6})", t, flags=re.IGNORECASE):
        out.setdefault("deposit_account", m.group(1))
        out["account_last4"] = m.group(1)[-4:]
    if deposit_context and re.search(r"\bCHASE\b|JPMORGAN\s+CHASE", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "JPMorgan Chase Bank")
        out.setdefault("source_system", "Chase")
    if deposit_context and re.search(r"\bREGIONS\b", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "Regions Bank")

    for key, pat in [
        ("deposit_amount", r"Deposit\s+Amount\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Deposit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Checking\s+Deposit\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Commercial\s+Deposit\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("check_total", r"Checks?\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("credit_total", r"Credit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("debit_total", r"Debit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("difference", r"Difference\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("cash_back", r"Cash\s+Back\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Total\s+of\s+Deposits\s+Submitted\s*:?\s*\$?([\d,]+\.\d{2})"),
    ]:
        if m := re.search(pat, t, flags=re.IGNORECASE):
            val = parse_money(m.group(1))
            if val is not None:
                out[key] = val
                if key == "deposit_total":
                    out.setdefault("check_total", val)

    if m := re.search(r"Item\s+Count\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Total\s+Number\s+of\s+Items\s*:?\s*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Total\s+Items\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))

    for date_pat in [
        r"Business\s+Date\s*:?[ \t]*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
        r"Report\s+Time\s*:?[ \t]*(20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
        r"\bDate\s*:?[ \t]*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
    ]:
        if m := re.search(date_pat, t, flags=re.IGNORECASE):
            parsed = parse_date(m.group(1))
            if parsed:
                out["deposit_date"] = parsed
                break

    clean = {k: v for k, v in out.items() if v not in (None, "", [])}
    if not deposit_context and not _regions:
        return {}
    return clean


def parse_batch_line_items(text: str) -> List[Dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    items: List[Dict[str, Any]] = []

    items.extend(parse_regions_deposit_items(text))

    existing_keys = {(i.get("source"), i.get("serial_number"), i.get("amount_numeric")) for i in items}
    for item in parse_transaction_detail_items(text):
        key = (item.get("source"), item.get("serial_number"), item.get("amount_numeric"))
        if key not in existing_keys:
            items.append(item)
            existing_keys.add(key)

    row_pat = re.compile(
        r"^(?P<unit>[A-Z]?\d{1,5}[A-Z]?)\s+"
        r"(?P<resident>.+?)\s+"
        r"(?P<desc>Payment[- ](?:MoneyOrder|Check)|Escrow[^\s]*)\s+"
        r"(?P<serial>[A-Z0-9-]{4,18})\s+"
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
        desc = m.group("desc").replace(" ", "-")
        item = {
            "item_no": len(items) + 1,
            "unit": m.group("unit"),
            "resident_name": m.group("resident").strip(),
            "payment_description": desc,
            "instrument_type": "Check" if "Check" in desc else "MoneyOrder",
            "serial_number": serial,
            "posted_date": parse_date(m.group("date")),
            "amount_numeric": parse_money(m.group("amount")),
            "posted_by": (m.group("posted_by") or "").strip() or None,
            "source": "yottareal_batch_detail",
        }
        key = (item.get("source"), item.get("serial_number"), item.get("amount_numeric"))
        if key not in existing_keys:
            items.append(item)
            existing_keys.add(key)
    return items
