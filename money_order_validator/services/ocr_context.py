from __future__ import annotations

import re
from typing import Iterable, List

from money_order_validator.settings import settings


IMPORTANT_PATTERNS = [
    r"PAY\s+EXACTLY",
    r"PAY\s+TO\s+THE\s+ORDER",
    r"PAYABLE\s+TO",
    r"PURCHASER|REMITTER|DRAWER|SENDER",
    r"MONEY\s*ORDER|CASHIER|CHECK\s+NO|SERIAL",
    r"WESTERN\s+UNION|MONEYGRAM|INTERMEX|FIDELITY|BARRI|DOLEX|PLS|CHASE|JPMORGAN|WELLS\s+FARGO|PROSPERITY",
    r"\$\s*\d|\*+\s*\$?\d",
    r"\b(?:19|22)[-\s]?\d{7,10}\b|\b\d{9,12}\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b20\d{2}-\d{2}-\d{2}\b",
    r"APT|APARTMENT|UNIT|SUITE|#\s*\d",
    r"BATCH\s*#|BATCH\s+AMOUNT|ACTUAL\s+ITEMS|TOTALS?\b|DEPOSIT\s+ACCOUNT|TRANSACTION\s+ID",
    r"MICR|ROUTING|ACCOUNT",
]


def _clean_line(line: str) -> str:
    line = re.sub(r"[ \t]+", " ", line.strip())
    return line


def dedupe_keep_order(lines: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        key = re.sub(r"\W+", "", line).upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def compact_ocr_context(text: str, max_chars: int | None = None) -> str:
    if not text:
        return ""
    max_chars = max_chars or settings.ocr_context_max_chars
    raw_lines = [_clean_line(x) for x in text.splitlines()]
    raw_lines = [x for x in raw_lines if len(x) >= 2]

    keep: List[str] = []
    combined_patterns = [re.compile(p, re.IGNORECASE) for p in IMPORTANT_PATTERNS]
    for idx, line in enumerate(raw_lines):
        if any(p.search(line) for p in combined_patterns):
            # Add neighbor lines because OCR splits values across lines/columns.
            if idx > 0:
                keep.append(raw_lines[idx - 1])
            keep.append(line)
            if idx + 1 < len(raw_lines):
                keep.append(raw_lines[idx + 1])

    if not keep:
        keep = raw_lines[:35]

    keep = dedupe_keep_order(keep)
    text_out = "\n".join(keep)
    if len(text_out) <= max_chars:
        return text_out

    # Preserve beginning and last numeric-heavy lines.
    head = []
    total = 0
    for line in keep:
        if total + len(line) + 1 > max_chars:
            break
        head.append(line)
        total += len(line) + 1
    return "\n".join(head)
