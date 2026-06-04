from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
except Exception:  # Optional dependency until requirements are installed.
    AsyncAzureOpenAI = None
    AsyncOpenAI = None
from PIL import Image

from money_order_validator.schemas import TokenUsage
from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


def _resize_image(image: Image.Image, max_width: int) -> Image.Image:
    img = image.convert("RGB")
    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    return img


def image_to_data_url(image: Image.Image, max_width: int, quality: int = 88) -> str:
    img = _resize_image(image, max_width=max_width)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class LLMClient:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(max(1, settings.openai_concurrency))
        self.mode = "none"
        self.model = ""
        self.client: Optional[Any] = None

        if settings.azure_openai_ready and AsyncAzureOpenAI is not None:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint.rstrip("/"),
                api_version=settings.azure_openai_api_version,
                timeout=settings.openai_timeout_seconds,
                max_retries=2,
            )
            self.model = settings.azure_openai_deployment_name
            self.mode = "azure"
        elif settings.openai_api_key and AsyncOpenAI is not None:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=2,
            )
            self.model = settings.openai_model_name
            self.mode = "openai"

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.model)

    async def json_vision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image,
        max_width: int,
        detail: str = "high",
        max_completion_tokens: int = 2500,
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        if not self.available:
            return {}, TokenUsage()

        data_url = image_to_data_url(image, max_width=max_width)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                ],
            },
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0,
        }
        async with self._semaphore:
            response = await self._create_with_fallback(kwargs)
        content = response.choices[0].message.content or "{}"
        return parse_json_object(content), TokenUsage.from_openai_usage(getattr(response, "usage", None))

    async def json_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 1000,
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        if not self.available:
            return {}, TokenUsage()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0,
        }
        async with self._semaphore:
            response = await self._create_with_fallback(kwargs)
        content = response.choices[0].message.content or "{}"
        return parse_json_object(content), TokenUsage.from_openai_usage(getattr(response, "usage", None))

    async def _create_with_fallback(self, kwargs: Dict[str, Any]) -> Any:
        assert self.client is not None
        attempts = []

        async def _call(k: Dict[str, Any]) -> Any:
            return await self.client.chat.completions.create(**k)

        variants = [dict(kwargs)]

        no_temp = dict(kwargs)
        no_temp.pop("temperature", None)
        variants.append(no_temp)

        no_json = dict(no_temp)
        no_json.pop("response_format", None)
        variants.append(no_json)

        max_tokens_variant = dict(no_json)
        if "max_completion_tokens" in max_tokens_variant:
            max_tokens_variant["max_tokens"] = max_tokens_variant.pop("max_completion_tokens")
        variants.append(max_tokens_variant)

        last_exc: Optional[Exception] = None
        for variant in variants:
            signature = tuple(sorted(variant.keys()))
            if signature in attempts:
                continue
            attempts.append(signature)
            try:
                return await _call(variant)
            except Exception as exc:  # Azure deployments vary on supported params.
                last_exc = exc
                msg = str(exc).lower()
                if not any(s in msg for s in ("temperature", "response_format", "max_completion_tokens", "max_tokens", "unsupported", "unknown parameter")):
                    logger.warning("OpenAI call failed: %s", exc)
                    raise
                logger.info("Retrying OpenAI call with reduced parameters after: %s", exc)
        assert last_exc is not None
        raise last_exc


llm_client = LLMClient()
