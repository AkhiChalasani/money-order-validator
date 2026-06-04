# Check & Money Order Validator API - Refactored

This folder is a complete refactor of the batch validator into a modular FastAPI project.
It is built for the sample PDFs that contain YottaReal/Deposit Batch reports, deposit receipts,
front/back money orders, cashier's checks, and pages with multiple instruments.

## What changed

- Azure Document Intelligence is used first for OCR and page routing.
- Back pages, receipts, blank pages, and deposit-only pages are skipped before OpenAI vision calls.
- OCR text sent to the model is compressed to only important lines instead of sending full-page OCR.
- The vision prompt is shorter and schema-focused.
- `/validate-batch-async` and legacy `/job/{job_id}` are included for compatibility.
- Single-file `/validate-batch` returns the direct JSON result: `batch`, `instruments`, `deposit_slip`.
- Multi-file or large-file runs return a job and then `results` from `/v1/jobs/{job_id}/result`.

## Folder structure

```text
money_order_validator_refactor/
  .env
  .env.example
  app.py
  run.py
  requirements.txt
  test_api.py
  money_order_validator/
    main.py
    settings.py
    schemas.py
    prompts.py
    clients/
      azure_openai.py
      azure_document_intelligence.py
    services/
      renderer.py
      page_classifier.py
      ocr_context.py
      regex_parsers.py
      extraction.py
      validation.py
      pipeline.py
      jobs.py
```

## Setup

```bash
cd money_order_validator_refactor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `.env` and fill in:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-AZURE-OPENAI-RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4
AZURE_OPENAI_API_VERSION=2025-04-01-preview

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://YOUR-DOC-INTELLIGENCE-RESOURCE.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

Then run:

```bash
python run.py
```

API will start on `http://localhost:8000`.

## Test

```bash
python test_api.py
python test_api.py "../MergedBatchChecks (36).pdf"
python test_api.py "../MergedBatchChecks (35).pdf" --async-mode
```

## Main endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Sync validation

```bash
curl -X POST "http://localhost:8000/validate-batch?mode=sync" \
  -F "files=@MergedBatchChecks.pdf"
```

### Async validation

```bash
curl -X POST http://localhost:8000/validate-batch-async \
  -F "files=@MergedBatchChecks.pdf"
```

Poll:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/jobs/<job_id>/result
```

## Notes on matching ChatGPT website behavior

The API sends the page image to the configured Azure OpenAI deployment with deterministic JSON extraction.
The ChatGPT website may use internal OCR/routing and a different exact model build, so byte-for-byte identical
results cannot be guaranteed. For closest behavior, keep:

```bash
FORCE_VISION_FOR_INSTRUMENTS=true
OPENAI_CONCURRENCY=2
MAX_IMAGE_WIDTH=1280
REPORT_IMAGE_WIDTH=1800
OCR_CONTEXT_MAX_CHARS=2600
```

## Token controls

To reduce tokens further after accuracy testing:

```bash
FORCE_VISION_FOR_INSTRUMENTS=false
VISION_ON_UNKNOWN_PAGES=false
OCR_CONTEXT_MAX_CHARS=1600
MAX_IMAGE_WIDTH=1024
```

Use `batch.processing_stats` in the response to see token usage by phase.


## Regions deposit report support

For Regions `Details of Deposits by Account` PDFs, page 1/2 are bank register pages, not checks. The parser now:

- classifies those pages as `deposit_report`, so they are not sent to instrument vision;
- extracts `property_name`, `account_number`, `batch_amount`, `total_items`, deposit date, and deposit number from the report header;
- extracts each `Capture Seq.` row as a register item;
- reconciles visible instruments against the register by check/serial/MICR suffix;
- corrects amount OCR/cents errors from the authoritative register amount;
- emits unmatched register rows with `missing_from_scan=true`.
