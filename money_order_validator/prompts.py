LLM_SYSTEM_MSG = (
    "You are a precise bank document extraction engine. Return only valid JSON. "
    "Use null for unreadable fields. Never guess digits."
)

INSTRUMENT_EXTRACTION_PROMPT = """
Extract all visible FRONT-SIDE payment instruments from the image.
The image can be rotated, sideways, cropped, or can contain multiple instruments on one page.

Return JSON only:
{
  "instruments": [
    {
      "instrument_type": "MoneyOrder|Check|CashiersCheck|Escrow",
      "payment_description": "Payment-MoneyOrder|Payment-Check|Escrow Deposit Paid In",
      "issuer": "Western Union|MoneyGram|PLS|DolEx|Intermex|Fidelity Express|JPMorgan Chase|Wells Fargo|Comerica Bank|null",
      "issuer_agent": string|null,
      "serial_number": string|null,
      "issue_date": "YYYY-MM-DD"|null,
      "amount_numeric": number|null,
      "amount_words": string|null,
      "payee_raw": string|null,
      "unit": string|null,
      "payer_name": string|null,
      "payer_address": string|null,
      "payer_signature": boolean,
      "payment_for_acct": string|null,
      "micr_line": string|null,
      "mobile_deposit_prohibited": boolean,
      "watermark_present": boolean
    }
  ]
}

Skip and do not include:
- Back pages: service charge text, load-this-direction arrows, endorsement panels, "for deposit only" stamps,
  purchaser agreement text, security-feature/legal disclosure pages.
- Deposit tickets/forms, deposit slips, deposit receipts, Chase ATM receipts, batch/register pages, blank pages.
- Pre-printed deposit-ticket forms at the top of a mixed page are NOT checks; do not create an instrument for them even if they have a MICR line or total box.
- Regions "Details of Deposits by Account" pages with Capture Seq/R/T/Post Amount/Credit Amount tables.
- Bleed-through text from the other side of a money order.

Important extraction rules:
- Extract only fields physically visible on the page image.
- A page can contain one, two, three, or more instruments. Return one object per front-side instrument.
- Amount must come from a labeled amount box or written amount words. MICR/routing/account numbers are never amounts.
- Preserve cents exactly. If the amount box visually shows separated cents like "$356 56", return 356.56, not 356.00.
- If written amount words clearly disagree with numeric amount, trust the written amount words.
- Western Union serials often start with 19 or 22 and may print with a dash. Return the actual digits, not examples.
- MoneyGram serial is usually top-right/check-number area. Do not confuse vertical form numbers with the serial.
- For cashier/personal checks, serial_number is the check number, usually top-right and/or final MICR group.
- payee_raw is the property/community name only. If the payee line also has an apartment/unit number, put that number in unit.
- unit is an apartment/unit identifier, usually 3-5 digits, from payee line, purchaser address, memo, payment-for/account field, or handwritten notes.
- payer_name is the purchaser/remitter/drawer name, not labels like "purchaser", "remitter", or "address".
- Return null instead of hallucinating unreadable names or digits.

Compressed OCR context from Azure Document Intelligence, if available:
{ocr_context}
""".strip()

BATCH_HEADER_PROMPT = """
Extract batch/header fields from this property-management batch/deposit page.
Return JSON only with keys:
{
  "property_name": string|null,
  "property_address": string|null,
  "batch_number": string|null,
  "batch_type": string|null,
  "batch_status": string|null,
  "pay_period": string|null,
  "bank_name": string|null,
  "account_number": string|null,
  "total_items": integer|null,
  "batch_amount": number|null,
  "printed_on": "YYYY-MM-DD"|null,
  "deposit_transaction": string|null
}
Rules:
- Batch number must come from a visible Batch # label when present.
- Do not use money-order serials, transaction IDs, routing numbers, or account numbers as batch_number.
- Account number is the deposit/bank account, not the 9-digit routing number.
- Normalize JPMorgan/CHASE to JPMorgan Chase Bank when clear.

OCR context:
{ocr_context}
""".strip()

DEPOSIT_SLIP_PROMPT = """
Extract deposit slip or deposit report information from this page.
Return JSON only:
{
  "bank_name": string|null,
  "deposit_account": string|null,
  "deposit_transaction": string|null,
  "deposit_date": "YYYY-MM-DD"|null,
  "deposit_amount": number|null,
  "check_total": number|null,
  "cash_back": number|null,
  "item_count": integer|null,
  "credit_total": number|null,
  "debit_total": number|null,
  "difference": number|null
}
Do not extract individual money orders here. For Chase ATM/teller receipts, extract the Checking Deposit/Commercial Deposit total and Business Date.
OCR context:
{ocr_context}
""".strip()

DEPOSIT_TICKET_ITEMS_PROMPT = """
Extract handwritten line items from a bank deposit ticket/slip page.
Return JSON only:
{
  "items": [
    {
      "item_no": integer|null,
      "unit": string|null,
      "amount_numeric": number|null
    }
  ]
}
Rules:
- Use the numbered rows in the deposit ticket table only.
- The row amount is split across DOLLARS and CENTS columns. Combine them exactly.
  Example: DOLLARS=610 and CENTS=50 -> amount_numeric=610.50.
- If the total or page is upside down, rotate mentally and read the rows upright.
- Do not extract the preprinted MICR/account number as an item.
- Do not extract the deposit total as an item.
- Stop at the last filled row. Empty rows must not be returned.
- Return [] if no row-level item amounts are visible.

OCR context:
{ocr_context}
""".strip()


DEPOSIT_DETAIL_REPORT_ITEMS_PROMPT = """
Extract the authoritative transaction rows from this Deposit Detail Report page.

Return JSON only:
{
  "items": [
    {
      "item_no": integer|null,
      "aux_serial": string|null,
      "routing_number": string|null,
      "account_number": string|null,
      "check_number": string|null,
      "amount_numeric": number|null,
      "item_type": string|null,
      "is_deposit_total": boolean
    }
  ]
}

Rules:
- Read the printed black-header table rows, not the thumbnail image text.
- Table headers usually include AUX/Serial, RIC, RT, WAUX/FLD4, Account, Check, Amount, Item Type, Item Status.
- The first ELECTRONIC/Credit row is the aggregate deposit total. Mark it is_deposit_total=true.
- Do NOT return the aggregate credit row as a physical payment item. It can be included only if is_deposit_total=true.
- All later rows with Item Type like 0003 or Debit are physical deposited payment items.
- Use the printed row amount above each thumbnail as amount_numeric. Do not use handwritten amount in the thumbnail when the row amount is visible.
- Preserve cents exactly. $294.36 must remain 294.36, not 294.56.
- If AUX/Serial is blank, derive serial_number later from account_number; leave aux_serial null.
- Return each visible printed row once. Do not duplicate rows from thumbnail front/back images.

OCR context:
{ocr_context}
""".strip()


REGISTER_ITEMS_PROMPT = """
Extract the item/register rows from this bank deposit report or property-management batch table.
Return JSON only:
{
  "items": [
    {
      "item_no": integer|null,
      "routing_number": string|null,
      "account_number": string|null,
      "check_number": string|null,
      "serial_number": string|null,
      "instrument_type": "Check|MoneyOrder|null",
      "payment_description": "Payment-Check|Payment-MoneyOrder|null",
      "amount_numeric": number|null
    }
  ]
}
Rules:
- Extract only real table rows, not header totals.
- For Regions reports, rows are under columns like Capture Seq., R/T, Account Number, Check Number, Post Amount, Credit Amount.
- Use Post Amount as amount_numeric.
- For personal/business checks, serial_number is the check number.
- For money orders, serial_number is the long check/MO number when shown.
- Do not duplicate rows from continued summary pages that do not show item rows.
- Return [] if no individual item rows are visible.

OCR context:
{ocr_context}
""".strip()
