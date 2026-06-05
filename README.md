# Check & Money Order Validator API

A modular FastAPI service that validates YottaReal/Chase/Regions deposit batches containing money orders, checks, cashier's checks, and mixed-format PDFs. It uses Azure Document Intelligence for OCR and page routing, then Azure OpenAI vision for instrument extraction and deposit-slip reading.

---

## Response schema

Every `POST /validate-batch` call returns:

```json
{
  "file_name": "...",
  "batch": {
    "batch_id": "uuid",
    "batch_number": "...",
    "batch_type": "Check/MO",
    "bank_name": "JPMorgan Chase Bank",
    "account_number": "...",
    "property_name": "...",
    "property_aliases": ["..."],
    "deposited_date": "YYYY-MM-DD",
    "total_items": 7,
    "batch_amount": 3935.34,
    "source_system": "YottaReal | Chase | Regions | Deposit Detail Report",
    "overall_decision": "ACCEPT | REVIEW | REJECT",
    "processing_stats": { ... },
    "reconciliation": {
      "instrument_sum": 3935.34,
      "deposit_total": 3935.34,
      "difference": 0.0,
      "amounts_match": true,
      "instrument_count": 7,
      "expected_item_count": 7,
      "item_count_match": true,
      "decision": "PASS | PASS_WITH_REVIEW | FAIL",
      "flags": ["amounts_reconciled", "item_count_reconciled"]
    },
    "risk_summary": { ... }
  },
  "instruments": [ ... ],
  "deposit_slip": { ... },
  "deposit_slips": [ ... ]
}
```

---

## Folder structure

```text
money_order_validator_refactor/
  .env
  .env.example
  run.py
  requirements.txt
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
      image_utils.py
      ocr_context.py
      regex_parsers.py
      extraction.py
      validation.py
      pipeline.py
      jobs.py
```

---

## Setup

```bash
cd money_order_validator_refactor
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fill in `.env`:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4
AZURE_OPENAI_API_VERSION=2025-04-01-preview

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://YOUR-DOC-INTEL.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

Then run:

```bash
python run.py
# API live at http://localhost:8000
```

---

## Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

### Sync validation (returns JSON directly)

```bash
curl -X POST "http://localhost:8000/validate-batch?mode=sync" \
  -F "files=@MergedBatchChecks.pdf"
```

### Async validation

```bash
# Submit
curl -X POST http://localhost:8000/validate-batch-async \
  -F "files=@MergedBatchChecks.pdf"

# Poll
curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/jobs/<job_id>/result
```

---

## .env reference

| Variable | Default | Purpose |
|---|---|---|
| `FORCE_VISION_FOR_INSTRUMENTS` | `true` | Always use vision for instrument pages |
| `VISION_ON_UNKNOWN_PAGES` | `true` | Try vision on unclassified pages |
| `INCLUDE_REGISTER_ONLY_ITEMS` | `true` | Emit register rows with no matching scan as `missing_from_scan=true` |
| `RETURN_DEBUG_PAGES` | `false` | Include per-page debug info in response |
| `OPENAI_CONCURRENCY` | `2` | Max parallel LLM calls |
| `MAX_IMAGE_WIDTH` | `1280` | Instrument page image width |
| `REPORT_IMAGE_WIDTH` | `1800` | Report/deposit page image width |
| `OCR_CONTEXT_MAX_CHARS` | `2600` | Max OCR chars sent to LLM |
| `PDF_RENDER_DPI` | `180` | PDF render resolution |

Use `batch.processing_stats` in the response to see token usage by phase and LLM call count.

---

## Supported document types

### YottaReal batch detail reports
Regex-only header parsing. Extracts `batch_number`, `batch_amount`, `total_items`, `property_name`, GL table, and per-instrument register rows. Zero LLM cost for the report page.

### Chase deposit tickets (handwritten)
- Detects deposit-ticket pages by signal scoring.
- Reads handwritten row amounts using `extract_deposit_ticket_items` with multi-orientation retry (0°, 180°, 90°, 270°) — handles upside-down and sideways scans.
- Row amounts are applied to following instruments as a cents-only fill (never overwrites a model-read amount that has written-amount evidence).
- Ticket totals are corrected against the following instrument sum when counts match.

### Chase Deposit Detail Reports
- Detects "Deposit Detail Report" pages by header pattern.
- Reads the printed black-header table rows using `extract_deposit_detail_report_items`:
  - Full-page vision pass (catches most rows).
  - Per-row-crop vision pass (thick black header bars cropped individually — more reliable for small/blurred text).
  - OCR regex pass (`parse_transaction_detail_items`).
  - All three sources merged and deduplicated by serial+amount.
- If the row extractor misses a row but the control total is internally balanced, a `gap_item` placeholder is inserted tagged `unread_deposit_detail_report_row`.
- Instrument vision is blocked for these pages — report rows are the authoritative instruments.
- `credit_items` / `debit_items` parsed from the control block for precise physical row count.

### Regions "Details of Deposits by Account"
- Classifies pages as `deposit_report`, not `unknown`.
- Parses `Account Name/Number`, `Total of Deposits Submitted`, `Total Number of Items`, deposit date, and account number without an LLM call.
- Parses the Regions item table using `Capture Seq.`, `R/T`, `Account Number`, `Check Number`, `Post Amount`.
- Reconciles vision-extracted instruments against the register; overrides small OCR/cents errors from the register.
- Unmatched register rows emitted as `missing_from_scan=true`.
- DBA property aliases (e.g. `Raja Bata LP dba Arella Forest in Woodland` → `Arella Forest`) reduce false `payee_mismatch` flags.

### Mixed instrument pages
- Page classifier routes pages to instrument vision, deposit-slip vision, register extraction, or skip.
- Back pages (`SERVICE CHARGE`, `LOAD THIS DIRECTION`, etc.) are hard-skipped before vision.
- Deposit-ticket artifacts (pre-printed MICR rows hallucinated as checks) are filtered before reconciliation.
- Back-page artifacts (low-evidence rows on the page immediately after a real instrument) are dropped.
- Azure DI page angle used to rotate inverted pages (80–100°, 170–190°, 260–280°) before vision.

---

## Batch reconciliation

After all instruments are validated, `apply_batch_reconciliation` compares:

| Check | ACCEPT condition |
|---|---|
| `instrument_sum` vs `deposit_total` | Difference ≤ $0.01 |
| `instrument_count` vs `expected_item_count` | Exact match (or no expected count) |
| Item-level flags | No INVALID items, no manual-review flags |

| Decision | `overall_decision` |
|---|---|
| All checks pass | `ACCEPT` |
| Amounts/counts match but items flagged | `REVIEW` |
| Amount or count mismatch | `REJECT` |

---

## Image quality flags

Instruments extracted from unclear pages are tagged `image_quality: "unclear"` and get three validation flags:

- `unclear_instrument_image`
- `low_confidence_extraction`
- `manual_review_required`

This adds +0.18 risk score and forces `REVIEW` status without deleting the instrument.

When vision extracts more rows than the authoritative item count, the weakest rows are suppressed and a `batch_review_notes` annotation is added to the kept rows.

---

## Property alias matching

All payee matching uses `build_property_aliases` which expands DBA names, strips legal suffixes, and generates partial-name variants. Example:

```
"Raja Bata LP dba Arella Forest in Woodland"
→ ["Raja Bata LP dba Arella Forest in Woodland", "Arella Forest in Woodland",
   "Arella Forest", "Arella", "Raja Bata"]
```

A payee that matches any alias scores ≥ 0.85 and does not trigger `payee_mismatch`.

---

## Token controls

For accuracy-first settings (recommended):

```bash
FORCE_VISION_FOR_INSTRUMENTS=true
INCLUDE_REGISTER_ONLY_ITEMS=true
```

To reduce tokens after accuracy testing:

```bash
FORCE_VISION_FOR_INSTRUMENTS=false
VISION_ON_UNKNOWN_PAGES=false
OCR_CONTEXT_MAX_CHARS=1600
MAX_IMAGE_WIDTH=1024
```
