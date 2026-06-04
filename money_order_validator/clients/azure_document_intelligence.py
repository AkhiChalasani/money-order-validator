from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

try:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except Exception:  # Optional dependency until requirements are installed.
    DocumentIntelligenceClient = None
    AnalyzeDocumentRequest = None
    AzureKeyCredential = None

from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OcrPage:
    page_number: int
    text: str
    angle: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class AzureDocumentIntelligenceReader:
    def __init__(self) -> None:
        self.endpoint = settings.azure_document_intelligence_endpoint
        self.key = settings.azure_document_intelligence_key

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.key and DocumentIntelligenceClient and AnalyzeDocumentRequest and AzureKeyCredential)

    async def analyze_pdf(self, content: bytes) -> List[OcrPage]:
        if not self.available:
            return []
        try:
            async with DocumentIntelligenceClient(
                endpoint=self.endpoint.rstrip("/"),
                credential=AzureKeyCredential(self.key),
            ) as client:
                poller = await client.begin_analyze_document(
                    "prebuilt-read",
                    AnalyzeDocumentRequest(bytes_source=content),
                )
                result = await poller.result()
        except Exception as exc:
            logger.warning("Azure Document Intelligence failed; continuing without OCR: %s", exc)
            return []

        pages: List[OcrPage] = []
        for page in sorted(result.pages or [], key=lambda p: p.page_number):
            lines = page.lines or []
            text = "\n".join(line.content for line in lines if getattr(line, "content", None))
            pages.append(
                OcrPage(
                    page_number=page.page_number,
                    text=text,
                    angle=getattr(page, "angle", None),
                    width=getattr(page, "width", None),
                    height=getattr(page, "height", None),
                )
            )
        logger.info("Azure DI extracted OCR for %d page(s)", len(pages))
        return pages


adi_reader = AzureDocumentIntelligenceReader()
