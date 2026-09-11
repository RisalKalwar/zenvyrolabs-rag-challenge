# AI-Powered CV Screening & Automated Recruitment System

Combines a criteria-based CV evaluation engine (Python + Groq LLM) with an
n8n automation workflow that routes each candidate to a personalized
selection or rejection email.

---

## 1. Architecture

```
Bulk CV Upload (multiple PDFs + criteria JSON)
        |
        v
FastAPI endpoint: POST /api/screen-cvs
        |
        v
For each CV:
   1. Extract full text (PyMuPDF)
   2. Extract candidate email (regex) -> flag if none found
   3. Build evaluation prompt from admin-supplied criteria
   4. Call Groq LLM -> { candidate_name, score, category, reasoning, feedback }
        |
        v
POST all results to n8n webhook
        |
        v
n8n workflow (hosted on n8n Cloud):
   - Split into one item per candidate (Code node)
   - Flagged (no email)?  -> log to "Manual Review" sheet, skip email
   - Otherwise, route on category (IF node):
       Selected -> selection email (Gmail), with the pre-generated feedback
       Rejected -> rejection email (Gmail), with the pre-generated feedback
   - Log every outcome to a "ScreeningLog" sheet
```

### Architecture decisions worth calling out

**No vector search / chunking for CV evaluation.** The original starter
project (a book-tutor RAG system) uses ChromaDB + chunking because
textbooks are hundreds of pages and no single LLM call can hold that much
context. CVs are the opposite case: 1-3 pages, short enough to pass in
full. Passing the full extracted text directly to the LLM is simpler and
more accurate for this document size than retrieving fragments.

**Feedback is generated once, in Python — not in n8n.** The personalized
feedback is produced by the same LLM call that scores the candidate (one
call per CV), so n8n's job is simplified to routing and templating.

**Email extraction is regex-only, not LLM-based.** CVs almost universally
list an email address in plain, unambiguous text — a regex is faster,
free, and deterministic for this specific sub-task.

**Split Out uses a Code node, not n8n's built-in Split Out node.** The
built-in node repeatedly failed to extract `body.candidates` correctly
in testing (returned "No output data" with no error, across both a
local instance and a fresh n8n Cloud instance). A two-line Code node
(`return $input.first().json.body.candidates.map(c => ({json: c}))`)
replaces it reliably.

**Filter nodes replaced with IF nodes for branching.** n8n's Filter node
only has a single output (items either pass through or get silently
dropped) — it cannot route to two different downstream paths. Both
branching decisions (email-flagged vs not, Selected vs Rejected)
needed IF nodes instead, which have real true/false outputs.

---

## 2. Setup

### Backend

```bash
pip install -r requirements.txt
pip install requests  # not in the original requirements.txt — needed to POST to n8n
```

Add to your `.env` (never commit this file):

```
GROQ_API_KEY=your_groq_api_key_here
N8N_WEBHOOK_URL=https://your-instance.app.n8n.cloud/webhook/cv-screening
```

Mount the new router in `main.py`:

```python
from cv_api import router as cv_router
app.include_router(cv_router)
```

Run as usual:

```bash
python main.py
```

### n8n — Cloud, not local

**Recommendation: use n8n Cloud (n8n.io), not a local `npx n8n` install.**
During development, running n8n locally on Windows surfaced a
browser-extension interference bug that caused nodes to silently return
"No output data" despite correct configuration and no thrown errors —
confirmed by the same workflow running correctly in an Incognito window
and, separately, on n8n Cloud. n8n Cloud sidesteps this entirely and is
the setup this project was ultimately built and tested against.

1. Sign up at n8n.io (free trial).
2. Import `n8n_cv_screening_workflow_final.json` (Workflows → Import
   from File).
3. Connect a Gmail credential on both "Send Selection Email" and "Send
   Rejection Email" nodes (n8n Cloud offers a simple Google sign-in
   flow for this).
4. Create a Google Sheet with two tabs: `ManualReview` and
   `ScreeningLog`, and connect that same Google account to both
   Google Sheets nodes. Set each node's Document/Sheet fields
   accordingly, and map each column to its corresponding field
   (e.g. `{{ $json.candidate_name }}`) — note that nodes positioned
   after a Gmail "send" node must reference the upstream node directly
   (e.g. `{{ $('Route: Selected vs Rejected').item.json.candidate_name }}`),
   since the Gmail node's own output overwrites `$json` with its send
   confirmation rather than passing the original data through.
5. Click **Publish** (this both saves a version and activates the
   workflow in this n8n Cloud version — no separate "Activate" toggle
   was found; publishing was confirmed sufficient by testing that the
   Production URL responded without manually re-arming a test listener).
6. Copy the **Production URL** from the Webhook node into
   `N8N_WEBHOOK_URL` in your `.env`.

---

## 3. Workflow — how a screening run actually works

1. Admin calls `POST /api/screen-cvs` with:
   - `files`: multiple CV PDFs (multipart form)
   - `criteria`: a JSON string, e.g.
     ```json
     {
       "required_skills": ["Python", "FastAPI", "SQL"],
       "education": "Bachelor's in Computer Science or related field",
       "experience": "1+ years, internships count",
       "projects": "At least one deployed/working project",
       "certifications": ["AWS", "Google"],
       "selection_threshold": 70
     }
     ```
   Every field is optional except a default threshold (70) is used if
   `selection_threshold` is omitted — nothing about the criteria is
   hardcoded in the code itself.
2. Each CV is evaluated independently and scored 0-100 against exactly
   the criteria provided in that request.
3. Results are returned in the API response immediately AND forwarded
   to n8n for email automation.
4. n8n sends the right email to each candidate automatically, using the
   feedback text the LLM already wrote.

Test scripts included for manual verification:
- `test_api.py` — sends a real CV through the FastAPI endpoint directly.
- `test_webhook.py` — sends a mock evaluation payload straight to the
  n8n webhook, bypassing the API/LLM (useful for testing the n8n side
  in isolation).

---

## 4. Limitations

- **Sequential processing.** CVs are evaluated one at a time — deliberate,
  since Groq's free tier has tight tokens-per-minute limits. For
  "hundreds of CVs" at production scale, this should move to a
  background job queue rather than a single blocking HTTP request.
- **No OCR fallback for scanned/image-only CVs.** A scanned CV with no
  text layer will return an empty extraction and be auto-rejected with
  a score of 0 — this should be reviewed manually rather than trusted
  as a real rejection.
- **Single email per candidate, no retry queue.** If a Gmail send fails,
  n8n's `continueErrorOutput` prevents the whole batch from stopping,
  but there's no automatic retry — failed sends should be checked in
  n8n's execution history.
- **Criteria matching quality depends on prompt/LLM judgment**, not a
  deterministic rules engine — intentional, but means identical CVs and
  criteria could occasionally produce slightly different scores across
  runs (temperature is set low at 0.2 to minimize this).
- **No authentication on `/api/screen-cvs`.** Should not be exposed on
  a public endpoint without adding one.
- **n8n Cloud free trial has execution limits** (1000 executions on the
  trial tier used here) — a production deployment would need a paid
  plan or a self-hosted instance with the local setup issues above
  resolved.
