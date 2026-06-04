from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from PIL import Image

from money_order_validator.clients.azure_openai import llm_client
from money_order_validator.prompts import (
    BATCH_HEADER_PROMPT,
    DEPOSIT_SLIP_PROMPT,
    INSTRUMENT_EXTRACTION_PROMPT,
    LLM_SYSTEM_MSG,
)
from money_order_validator.schemas import TokenUsage
from money_order_validator.services.image_utils import crop_to_content
from money_order_validator.services.ocr_context import compact_ocr_context
from money_order_validator.services.page_classifier import PageKind
from money_order_validator.services.regex_parsers import (
    parse_batch_header,
    parse_basic_instrument_from_ocr,
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

        prompt = INSTRUMENT_EXTRACTION_PROMPT.format(ocr_context=ocr_context or "(no OCR text available)")
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
        prompt = BATCH_HEADER_PROMPT.format(ocr_context=compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
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
        prompt = DEPOSIT_SLIP_PROMPT.format(ocr_context=compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
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


vision_extractor = VisionExtractor()
