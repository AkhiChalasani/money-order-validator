from __future__ import annotations

import asyncio
import logging
from typing import List, Literal, Tuple

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from money_order_validator.clients.azure_document_intelligence import adi_reader
from money_order_validator.clients.azure_openai import llm_client
from money_order_validator.schemas import JobStatus, ValidationResult
from money_order_validator.services.jobs import job_store
from money_order_validator.services.pipeline import document_processor
from money_order_validator.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Batch check and money-order extraction/validation API. "
        "Uses Azure Document Intelligence for OCR/page routing and Azure OpenAI vision for exact field extraction."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASYNC_SIZE_BYTES = 80 * 1024 * 1024


async def _read_uploads(files: List[UploadFile]) -> List[Tuple[str, bytes]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
    if len(files) > settings.max_files_per_batch:
        raise HTTPException(status_code=400, detail=f"Too many files. Max {settings.max_files_per_batch}.")
    payloads: List[Tuple[str, bytes]] = []
    for f in files:
        name = f.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {name}")
        content = await f.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large: {name}")
        payloads.append((name, content))
    return payloads


async def _run_job(job_id: str, payloads: List[Tuple[str, bytes]]) -> None:
    try:
        job_store.update_progress(job_id, {"files": len(payloads), "stage": "processing"})
        results = await document_processor.process_batch(payloads)
        body = _format_results(results)
        job_store.complete(job_id, body)
    except Exception as exc:
        logger.exception("Async job failed")
        job_store.fail(job_id, str(exc))


def _format_results(results: List[ValidationResult]) -> dict:
    if len(results) == 1:
        return results[0].model_dump(mode="json", exclude_none=True)
    return {"results": [r.model_dump(mode="json", exclude_none=True) for r in results]}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if llm_client.available else "degraded",
        "service": settings.app_name,
        "version": settings.version,
        "azure_openai": "ok" if settings.azure_openai_ready else "unconfigured",
        "azure_document_intelligence": "ok" if adi_reader.available else "unconfigured",
        "llm_provider": llm_client.mode,
        "llm_model_or_deployment": llm_client.model,
    }


@app.get("/models")
async def models() -> dict:
    return {
        "provider": llm_client.mode,
        "model_or_deployment": llm_client.model,
        "azure_openai_api_version": settings.azure_openai_api_version,
        "vision_extractor": "chat.completions multimodal JSON",
        "document_ocr": "azure-document-intelligence/prebuilt-read" if adi_reader.available else None,
    }


_FILE_UPLOAD_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["files"],
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                        }
                    },
                }
            }
        },
    }
}


@app.post("/validate-batch", openapi_extra=_FILE_UPLOAD_SCHEMA)
async def validate_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    mode: Literal["auto", "sync", "async"] = Query("auto"),
) -> dict:
    payloads = await _read_uploads(files)
    total_size = sum(len(data) for _, data in payloads)
    use_async = mode == "async" or (mode == "auto" and (len(payloads) > 1 or total_size > ASYNC_SIZE_BYTES))
    if use_async:
        job_id = job_store.create({"files": len(payloads), "stage": "queued"})
        background_tasks.add_task(_run_job, job_id, payloads)
        return {
            "status": "processing",
            "job_id": job_id,
            "poll_url": f"/v1/jobs/{job_id}",
            "result_url": f"/v1/jobs/{job_id}/result",
            "message": f"Processing {len(payloads)} file(s). Poll result_url when status is done.",
        }

    try:
        results = await asyncio.wait_for(
            document_processor.process_batch(payloads),
            timeout=settings.processing_timeout_seconds,
        )
        return _format_results(results)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Processing timeout. Retry with mode=async.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Synchronous validation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/validate-batch-async", status_code=status.HTTP_202_ACCEPTED, openapi_extra=_FILE_UPLOAD_SCHEMA)
async def validate_batch_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
) -> dict:
    payloads = await _read_uploads(files)
    job_id = job_store.create({"files": len(payloads), "stage": "queued"})
    background_tasks.add_task(_run_job, job_id, payloads)
    return {
        "status": "processing",
        "job_id": job_id,
        "poll_url": f"/v1/jobs/{job_id}",
        "result_url": f"/v1/jobs/{job_id}/result",
        "message": "Processing started. Poll the result_url when status is done.",
    }


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        status=job["status"],
        job_id=job_id,
        progress=job.get("progress"),
        error=job.get("error"),
    )


@app.get("/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "Job failed")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job is still processing")
    return job["result"]


# Backward-compatible aliases used by older test scripts.
@app.get("/job/{job_id}")
async def get_job_legacy(job_id: str) -> dict:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "done":
        result = job.get("result") or {}
        return {"status": "done", "job_id": job_id, **result}
    return {"status": job["status"], "job_id": job_id, "progress": job.get("progress"), "error": job.get("error")}


@app.get("/test-sample")
async def test_sample() -> dict:
    return {
        "file_name": "sample.pdf",
        "batch": {
            "batch_id": "sample",
            "batch_number": "060320001",
            "batch_type": "Check/MO",
            "property_name": "Sample Apartments",
            "total_items": 1,
            "batch_amount": 1000.00,
            "overall_decision": "ACCEPT",
        },
        "instruments": [
            {
                "item_no": 1,
                "instrument_id": "INS-060320001-001",
                "batch_number": "060320001",
                "instrument_type": "MoneyOrder",
                "payment_description": "Payment-MoneyOrder",
                "issuer": "Western Union",
                "serial_number": "22124699790",
                "issue_date": "2026-04-24",
                "amount_numeric": 1000.00,
                "payee_raw": "Sample Apartments",
                "unit": "1218",
                "validation": {"overall_status": "VALID", "risk_score": 0.0, "flags": []},
            }
        ],
    }
