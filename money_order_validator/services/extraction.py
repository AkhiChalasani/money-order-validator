from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from money_order_validator.clients.azure_openai import llm_client
from money_order_validator.prompts import (
    BATCH_HEADER_PROMPT,
    DEPOSIT_SLIP_PROMPT,
    DEPOSIT_TICKET_ITEMS_PROMPT,
    INSTRUMENT_EXTRACTION_PROMPT,
    LLM_SYSTEM_MSG,
    REGISTER_ITEMS_PROMPT,
)
from money_order_validator.schemas import TokenUsage
from money_order_validator.services.image_utils import crop_to_content
from money_order_validator.services.ocr_context import compact_ocr_context
from money_order_validator.services.page_classifier import PageKind
from money_order_validator.services.regex_parsers import (
    parse_batch_header,
    parse_basic_instrument_from_ocr,
    parse_batch_line_items,
    parse_deposit_info,
    sanitize_instrument,
)
from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


class VisionExtractor:
    async def extract_instruments(
        self,
        image: Image.Image,
        ocr_text: str,
        page_kind: PageKind,
    ) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Return instruments, token usage, and whether LLM was used."""
        ocr_context = compact_ocr_context(ocr_text)

        # OCR-only fallback when LLM is unavailable or disabled.
        if not llm_client.available:
            fallback = parse_basic_instrument_from_ocr(ocr_text)
            return ([fallback] if fallback else []), TokenUsage(), False

        if not settings.force_vision_for_instruments:
            fallback = parse_basic_instrument_from_ocr(ocr_text)
            if fallback and fallback.get("serial_number") and fallback.get("amount_numeric"):
                return [fallback], TokenUsage(), False

        prompt = INSTRUMENT_EXTRACTION_PROMPT.replace("{ocr_context}", ocr_context or "(no OCR text available)")
        width = settings.report_image_width if page_kind == PageKind.REPORT_WITH_INSTRUMENTS else settings.max_image_width
        img = crop_to_content(image)
        raw, usage = await llm_client.json_vision(
            system_prompt=LLM_SYSTEM_MSG,
            user_prompt=prompt,
            image=img,
            max_width=width,
            detail="high",
            max_completion_tokens=4000,
        )
        instruments_raw = raw.get("instruments") if isinstance(raw, dict) else None
        if isinstance(instruments_raw, dict):
            instruments_raw = [instruments_raw]
        if instruments_raw is None and isinstance(raw, dict) and any(raw.get(k) for k in ("serial_number", "amount_numeric", "payee_raw")):
            instruments_raw = [raw]
        if not isinstance(instruments_raw, list):
            instruments_raw = []

        instruments: List[Dict[str, Any]] = []
        for item in instruments_raw:
            if not isinstance(item, dict):
                continue
            sanitized = sanitize_instrument(item, ocr_text=ocr_text)
            # Drop obvious empties.
            if not any(sanitized.get(k) for k in ("serial_number", "amount_numeric", "payee_raw", "issuer", "micr_line")):
                continue
            instruments.append(sanitized)

        # Patch from OCR when vision missed machine-printed details.
        ocr_patch = parse_basic_instrument_from_ocr(ocr_text)
        if ocr_patch and len(instruments) == 1:
            for key in ("issuer", "issuer_agent", "serial_number", "amount_numeric", "issue_date", "micr_line"):
                if not instruments[0].get(key) and ocr_patch.get(key):
                    instruments[0][key] = ocr_patch[key]
        elif ocr_patch and not instruments:
            instruments.append(ocr_patch)

        return instruments, usage, True

    async def extract_batch_header(self, image: Image.Image, ocr_text: str) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        parsed = parse_batch_header(ocr_text)
        enough = bool(parsed.get("batch_number") or parsed.get("batch_amount") or parsed.get("total_items"))
        if enough or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = BATCH_HEADER_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
        raw, usage = await llm_client.json_vision(
            system_prompt=LLM_SYSTEM_MSG,
            user_prompt=prompt,
            image=crop_to_content(image),
            max_width=settings.report_image_width,
            detail="high",
            max_completion_tokens=1200,
        )
        merged = {**parsed}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if v not in (None, "", []):
                    merged[k] = v
        return parse_batch_header("\n".join([ocr_text, str(merged)])) | merged, usage, True

    async def extract_deposit(self, image: Image.Image, ocr_text: str) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        parsed = parse_deposit_info(ocr_text)
        enough = bool(parsed.get("deposit_amount") or parsed.get("deposit_total") or parsed.get("check_total"))
        if enough or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = DEPOSIT_SLIP_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
        raw, usage = await llm_client.json_vision(
            system_prompt=LLM_SYSTEM_MSG,
            user_prompt=prompt,
            image=crop_to_content(image),
            max_width=settings.report_image_width,
            detail="high",
            max_completion_tokens=1200,
        )
        merged = {**parsed}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if v not in (None, "", []):
                    merged[k] = v
        return merged, usage, True

    async def extract_deposit_ticket_items(
        self,
        image: Image.Image,
        ocr_text: str,
        expected_count: Optional[int] = None,
        expected_total: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Extract handwritten deposit-ticket row amounts.

        Chase deposit tickets are frequently scanned sideways or upside down. A
        single vision pass can read the table total but misread several line
        amounts. This method tries the native orientation first, then rotated
        copies when the row count/total does not look trustworthy, and selects
        the most internally consistent candidate.
        """
        if not llm_client.available:
            return [], TokenUsage(), False

        prompt = DEPOSIT_TICKET_ITEMS_PROMPT.replace(
            "{ocr_context}", compact_ocr_context(ocr_text, max_chars=3000) or "(no OCR text available)"
        )

        def _parse_items(raw: Dict[str, Any], orientation: int) -> List[Dict[str, Any]]:
            items_raw = raw.get("items") if isinstance(raw, dict) else None
            if not isinstance(items_raw, list):
                return []
            items: List[Dict[str, Any]] = []
            for idx, item in enumerate(items_raw, start=1):
                if not isinstance(item, dict):
                    continue
                amount = item.get("amount_numeric")
                try:
                    amount_f = round(float(str(amount).replace(",", "").replace("$", "")), 2)
                except (TypeError, ValueError):
                    continue
                if amount_f <= 0:
                    continue
                # Deposit-ticket row amounts are individual payment amounts. Drop obvious
                # copied totals or OCR garbage outside the normal MO/check range.
                if amount_f > 5000:
                    continue
                row = {
                    "item_no": item.get("item_no") or idx,
                    "unit": item.get("unit"),
                    "amount_numeric": amount_f,
                    "source": "deposit_ticket_sequence",
                    "payment_description": "Payment-MoneyOrder",
                    "instrument_type": "MoneyOrder",
                    "orientation_degrees": orientation,
                }
                items.append(row)
            return items

        def _candidate_score(items: List[Dict[str, Any]]) -> float:
            if not items:
                return -1_000_000.0
            row_sum = round(sum(float(i.get("amount_numeric") or 0.0) for i in items), 2)
            score = float(len(items)) * 100.0
            if expected_count:
                score -= abs(len(items) - int(expected_count)) * 500.0
                if len(items) == int(expected_count):
                    score += 1000.0
            # Use the slip total only as a weak signal. Handwritten totals are often OCR'd
            # with a missing leading digit, so a wrong expected_total must not force a bad
            # line-item candidate to win.
            if expected_total:
                diff = abs(row_sum - float(expected_total))
                if diff <= 1.0:
                    score += 400.0
                elif diff <= 250.0:
                    score += 75.0
                else:
                    score -= min(diff, 1000.0) / 25.0
            # If candidates have the same row count, prefer the larger positive sum. In these
            # tickets the common OCR failure is dropping a leading digit or reading 442 as 400,
            # not inventing extra dollars.
            score += row_sum / 1000.0
            return score

        total_usage = TokenUsage()
        candidates: List[Tuple[float, List[Dict[str, Any]]]] = []

        async def _run_for_orientation(degrees: int) -> None:
            img = image.rotate(degrees, expand=True) if degrees else image
            raw, usage = await llm_client.json_vision(
                system_prompt=LLM_SYSTEM_MSG,
                user_prompt=prompt,
                image=crop_to_content(img),
                max_width=settings.report_image_width,
                detail="high",
                max_completion_tokens=1800,
            )
            total_usage.prompt_tokens += usage.prompt_tokens
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.total_tokens += usage.total_tokens
            items = _parse_items(raw, degrees)
            candidates.append((_candidate_score(items), items))

        await _run_for_orientation(0)
        best_score, best_items = max(candidates, key=lambda c: c[0])
        best_sum = round(sum(float(i.get("amount_numeric") or 0.0) for i in best_items), 2)
        count_ok = bool(expected_count and len(best_items) == int(expected_count))
        total_ok = bool(expected_total and abs(best_sum - float(expected_total)) <= 1.0)

        # If native orientation does not match the ticket count/total, try rotated copies.
        # 180 degrees fixes upside-down deposit tickets; 90/270 cover sideways camera scans.
        if not (count_ok and (total_ok or not expected_total)):
            for degrees in (180, 90, 270):
                await _run_for_orientation(degrees)
            best_score, best_items = max(candidates, key=lambda c: c[0])

        return best_items, total_usage, True

    async def extract_register_items(self, image: Image.Image, ocr_text: str) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Extract bank/property register rows from a deposit report page.

        Focused fallback for when Azure OCR does not preserve table rows well enough for
        regex parsing. Intentionally separate from instrument vision.
        """
        parsed = parse_batch_line_items(ocr_text)
        if parsed or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = REGISTER_ITEMS_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
        raw, usage = await llm_client.json_vision(
            system_prompt=LLM_SYSTEM_MSG,
            user_prompt=prompt,
            image=crop_to_content(image),
            max_width=settings.report_image_width,
            detail="high",
            max_completion_tokens=2500,
        )
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items_raw, list):
            return [], usage, True
        items: List[Dict[str, Any]] = []
        for idx, item in enumerate(items_raw, start=1):
            if not isinstance(item, dict):
                continue
            amount = item.get("amount_numeric")
            serial = item.get("serial_number") or item.get("check_number")
            if amount in (None, "") or serial in (None, ""):
                continue
            inst_type = item.get("instrument_type") or ("MoneyOrder" if len(str(serial).lstrip("0")) >= 7 else "Check")
            if inst_type not in {"Check", "MoneyOrder", "CashiersCheck", "Escrow"}:
                inst_type = "MoneyOrder" if len(str(serial).lstrip("0")) >= 7 else "Check"
            row = {
                "item_no": item.get("item_no") or idx,
                "routing_number": item.get("routing_number"),
                "account_number": item.get("account_number"),
                "check_number": item.get("check_number") or serial,
                "serial_number": serial,
                "amount_numeric": amount,
                "instrument_type": inst_type,
                "payment_description": item.get("payment_description") or ("Payment-MoneyOrder" if inst_type == "MoneyOrder" else "Payment-Check"),
                "source": "vision_register_items",
            }
            items.append(row)
        return items, usage, True


vision_extractor = VisionExtractor()
