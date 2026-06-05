from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BatchContext(BaseModel):
    batch_id: str
    batch_number: Optional[str] = None
    batch_type: Optional[str] = None
    batch_status: Optional[str] = None
    pay_period: Optional[str] = None
    bank_name: Optional[str] = None
    account_number: Optional[str] = None
    property_name: Optional[str] = None
    property_aliases: List[str] = Field(default_factory=list)
    property_address: Optional[str] = None
    deposited_date: Optional[str] = None
    deposit_transaction: Optional[str] = None
    total_items: Optional[int] = None
    batch_amount: Optional[float] = None
    printed_on: Optional[str] = None
    source_system: Optional[str] = "YottaReal"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gl_summary: List[Dict[str, Any]] = Field(default_factory=list)
    overall_decision: Optional[Literal["ACCEPT", "REVIEW", "REJECT"]] = None
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    risk_summary: Dict[str, Any] = Field(default_factory=dict)


class Instrument(BaseModel):
    item_no: int
    instrument_id: str
    batch_number: Optional[str] = None
    unit: Optional[str] = None
    resident_name: Optional[str] = None
    instrument_type: str = "MoneyOrder"
    payment_description: str = "Payment-MoneyOrder"
    issuer: Optional[str] = None
    issuer_agent: Optional[str] = None
    serial_number: Optional[str] = None
    micr_line: Optional[str] = None
    issue_date: Optional[str] = None
    amount_numeric: Optional[float] = None
    amount_words: Optional[str] = None
    payee_raw: Optional[str] = None
    payee_normalized: Optional[str] = None
    payer_name: Optional[str] = None
    payer_address: Optional[str] = None
    payer_signature: bool = False
    payment_for_acct: Optional[str] = None
    mobile_deposit_prohibited: bool = False
    watermark_present: bool = False
    posted_by: Optional[str] = None
    posted_date: Optional[str] = None
    ocr_confidence: float = 0.0
    processing_tier: int = 3
    llm_used: bool = True
    missing_from_scan: bool = False
    page_number: Optional[int] = None
    source_file: Optional[str] = None
    image_quality: Optional[str] = None
    validation: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    file_name: Optional[str] = None
    batch: BatchContext
    instruments: List[Instrument]
    # Legacy aggregate object. When a PDF contains multiple physical slips/receipts, this
    # contains summed totals and also includes a nested deposit_slips array.
    deposit_slip: Optional[Dict[str, Any]] = None
    # Convenience top-level list for clients that need each physical slip separately.
    deposit_slips: Optional[List[Dict[str, Any]]] = None
    pages: Optional[List[Dict[str, Any]]] = None


class JobStatus(BaseModel):
    status: Literal["processing", "done", "failed"]
    job_id: str
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai_usage(cls, usage: Any) -> "TokenUsage":
        if not usage:
            return cls()
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
