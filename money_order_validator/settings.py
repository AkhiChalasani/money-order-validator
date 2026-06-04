from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


class Settings(BaseSettings):
    app_name: str = "Check & Money Order Validator API"
    version: str = "2.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = "2025-04-01-preview"

    # Optional standard OpenAI fallback. Azure is preferred if Azure values exist.
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o-mini"

    azure_document_intelligence_endpoint: Optional[str] = None
    azure_document_intelligence_key: Optional[str] = None

    max_file_size_mb: int = 120
    max_files_per_batch: int = 10
    processing_timeout_seconds: int = 900
    result_retention_minutes: int = 1440

    openai_concurrency: int = 2
    openai_timeout_seconds: int = 180

    pdf_render_dpi: int = 180
    max_image_width: int = 1280
    report_image_width: int = 1800
    ocr_context_max_chars: int = 2600

    # Accuracy/cost switches.
    # force_vision_for_instruments=true gives ChatGPT-site-like extraction because every likely
    # front page is sent as image. Set false only if you trust OCR-only parsing for clean batches.
    force_vision_for_instruments: bool = True
    vision_on_unknown_pages: bool = True
    return_debug_pages: bool = False
    include_register_only_items: bool = True

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.azure_openai_api_key = _first_env(
            "AZURE_OPENAI_API_KEY_GPT5_4",
            "AZURE_OPENAI_KEY_GPT5_4",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_KEY",
            default=self.azure_openai_api_key,
        )
        self.azure_openai_endpoint = _first_env(
            "AZURE_OPENAI_ENDPOINT_GPT5_4",
            "AZURE_OPENAI_ENDPOINT",
            default=self.azure_openai_endpoint,
        )
        self.azure_openai_deployment_name = _first_env(
            "AZURE_OPENAI_DEPLOYMENT_NAME_GPT5_4",
            "AZURE_OPENAI_DEPLOYMENT_GPT5_4",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_DEPLOYMENT",
            default=self.azure_openai_deployment_name,
        ) or ""
        self.azure_openai_api_version = _first_env(
            "AZURE_OPENAI_API_VERSION_GPT5_4",
            "AZURE_OPENAI_API_VERSION",
            default=self.azure_openai_api_version,
        ) or self.azure_openai_api_version
        self.azure_document_intelligence_endpoint = _first_env(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "DOCUMENT_INTELLIGENCE_ENDPOINT",
            default=self.azure_document_intelligence_endpoint,
        )
        self.azure_document_intelligence_key = _first_env(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY",
            "DOCUMENT_INTELLIGENCE_KEY",
            default=self.azure_document_intelligence_key,
        )

    @property
    def azure_openai_ready(self) -> bool:
        return bool(
            self.azure_openai_api_key
            and self.azure_openai_endpoint
            and self.azure_openai_deployment_name
        )

    @property
    def document_intelligence_ready(self) -> bool:
        return bool(self.azure_document_intelligence_endpoint and self.azure_document_intelligence_key)


settings = Settings()
