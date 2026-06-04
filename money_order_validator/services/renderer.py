from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from money_order_validator.settings import settings


class PdfRenderer:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def render(self, pdf_content: bytes, dpi: int | None = None) -> List[Image.Image]:
        dpi = dpi or settings.pdf_render_dpi
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._render_sync, pdf_content, dpi)

    @staticmethod
    def _render_sync(pdf_content: bytes, dpi: int) -> List[Image.Image]:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images: List[Image.Image] = []
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                images.append(img)
        finally:
            doc.close()
        return images


pdf_renderer = PdfRenderer()
