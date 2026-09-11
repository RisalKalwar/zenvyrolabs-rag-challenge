"""
cv_screening_engine.py

Core evaluation logic for the CV screening pipeline. Reuses the same LLM
(Groq) and PDF-loading stack as rag_engine.py, but deliberately skips
chunking/embedding/vector search — CVs are short (1-3 pages), so passing
the full extracted text directly to the LLM is more reliable than
retrieval-based RAG would be here. This trade-off is documented in the
README under "Architecture Decisions".
"""

import os
import re
import json
import time
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Create a .env file with "
        "GROQ_API_KEY=your_key_here (get one free at console.groq.com)"
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    timeout=30,
)

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # e.g. http://localhost:5678/webhook/cv-screening

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


# ---------------------------------------------------------------------------
# Step 1: Extract full text from a CV PDF
# ---------------------------------------------------------------------------
def extract_cv_text(file_path: str) -> str:
    """Extracts all text from a CV PDF using PyMuPDF directly (no chunking —
    see module docstring for why)."""
    doc = fitz.open(file_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    full_text = "\n".join(pages_text).strip()
    return full_text


# ---------------------------------------------------------------------------
# Step 2: Extract candidate email (regex first, LLM fallback)
# ---------------------------------------------------------------------------
def extract_email(cv_text: str) -> Optional[str]:
    """Returns the first valid-looking email address found in the CV text,
    or None if none is found. Regex-only by design — fast, deterministic,
    and CVs almost always list an email in plain text near the top."""
    match = EMAIL_REGEX.search(cv_text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Step 3: Build the evaluation prompt from configurable criteria
# ---------------------------------------------------------------------------
def _build_criteria_block(criteria: Dict[str, Any]) -> str:
    """Turns the admin-provided criteria dict into a readable block for the
    prompt. Every field is optional — only include what's actually provided,
    so the model isn't told to weigh criteria the admin didn't set."""
    lines = []
    if criteria.get("required_skills"):
        lines.append(f"- Required skills: {', '.join(criteria['required_skills'])}")
    if criteria.get("education"):
        lines.append(f"- Education requirement: {criteria['education']}")
    if criteria.get("experience"):
        lines.append(f"- Experience requirement: {criteria['experience']}")
    if criteria.get("projects"):
        lines.append(f"- Desired project experience: {criteria['projects']}")
    if criteria.get("certifications"):
        lines.append(f"- Preferred certifications: {', '.join(criteria['certifications'])}")
    if not lines:
        lines.append("- No specific criteria provided — evaluate general employability and role fit.")
    return "\n".join(lines)


EVAL_PROMPT_TEMPLATE = """You are an AI recruitment screener. Evaluate the candidate CV below
strictly against the criteria provided. Be fair, specific, and evidence-based —
cite concrete details from the CV to support your score, not generic praise.

Criteria:
{criteria_block}

Selection threshold: a CV scoring {threshold} or above should be categorized "Selected";
below that, "Rejected".

Candidate CV:
{cv_text}

Respond with ONLY a valid JSON object (no markdown fences, no extra text) matching this
exact shape:
{{
  "candidate_name": "<best-guess full name from the CV, or 'Unknown' if not findable>",
  "score": <integer 0-100>,
  "category": "<'Selected' or 'Rejected'>",
  "reasoning": "<2-4 sentences explaining the score, citing specific CV details>",
  "feedback": "<3-6 sentences of personalized, constructive feedback written directly to the candidate. Reference specific things from THEIR CV — actual skills, projects, or gaps. Never use generic filler like 'we were impressed by many candidates' or 'unfortunately we cannot move forward at this time' with no specifics.>"
}}
"""


def _invoke_with_backoff(prompt_text: str, max_retries: int = 5, initial_delay: int = 8) -> str:
    """Same retry/backoff pattern as rag_engine.py's _invoke_with_backoff —
    Groq's free tier rate-limits aggressively under bulk processing."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt_text)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "rate_limit" in msg.lower() or "429" in msg or "413" in msg
            if attempt == max_retries - 1 or not is_rate_limit:
                raise
            print(f"    ...rate limited, waiting {delay}s before retry {attempt + 2}/{max_retries}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Exceeded retry attempts")


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses the JSON object from the LLM's response, tolerating
    minor formatting slip-ups (e.g. accidental markdown fences)."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: pull out the first {...} block found anywhere in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------------------
# Step 4: Evaluate a single CV against criteria
# ---------------------------------------------------------------------------
def evaluate_cv(file_path: str, criteria: Dict[str, Any], filename: str = "") -> Dict[str, Any]:
    """
    Full per-candidate pipeline: extract text -> extract email -> LLM
    evaluation (score, category, reasoning, feedback). Returns a dict ready
    to be sent to n8n or returned via the API.
    """
    cv_text = extract_cv_text(file_path)

    if not cv_text:
        return {
            "filename": filename,
            "candidate_name": "Unknown",
            "email": None,
            "email_flagged": True,
            "score": 0,
            "category": "Rejected",
            "reasoning": "No extractable text found in the uploaded file — likely a scanned/image-only PDF.",
            "feedback": "",
            "error": "empty_extraction",
        }

    email = extract_email(cv_text)
    threshold = criteria.get("selection_threshold", 70)

    prompt = EVAL_PROMPT_TEMPLATE.format(
        criteria_block=_build_criteria_block(criteria),
        threshold=threshold,
        cv_text=cv_text[:12000],  # safety cap against unusually long CVs
    )

    raw_response = _invoke_with_backoff(prompt)

    try:
        result = _parse_json_response(raw_response)
    except (json.JSONDecodeError, AttributeError):
        return {
            "filename": filename,
            "candidate_name": "Unknown",
            "email": email,
            "email_flagged": email is None,
            "score": 0,
            "category": "Rejected",
            "reasoning": "AI evaluation failed to return parseable output.",
            "feedback": "",
            "error": "llm_parse_failure",
            "raw_response": raw_response[:500],
        }

    result["filename"] = filename
    result["email"] = email
    result["email_flagged"] = email is None  # requirement 5: flag for manual review if no email found
    return result


# ---------------------------------------------------------------------------
# Step 5: Bulk evaluation over multiple CVs
# ---------------------------------------------------------------------------
def evaluate_cvs_bulk(file_paths: List[str], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs evaluate_cv() sequentially over every uploaded CV.

    Sequential by design for this version — see README "Limitations" for
    why (Groq free-tier rate limits make naive parallelism risky without
    a queue/worker setup)."""
    results = []
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"[cv screening] evaluating {filename}...")
        result = evaluate_cv(path, criteria, filename=filename)
        results.append(result)
        time.sleep(1)  # light pacing between candidates, same rationale as rag_engine.py
    return results


# ---------------------------------------------------------------------------
# Step 6: Send results to n8n for email routing
# ---------------------------------------------------------------------------
def send_to_n8n(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    POSTs the evaluation results to the configured n8n webhook. n8n is
    responsible for routing each candidate to a Selected or Rejected email
    template (see n8n_cv_screening_workflow.json) — the feedback text is
    already fully generated here, so n8n only needs to place it into the
    right template and send.
    """
    if not N8N_WEBHOOK_URL:
        return {"sent": False, "reason": "N8N_WEBHOOK_URL not configured in .env"}

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json={"candidates": results},
            timeout=15,
        )
        return {
            "sent": True,
            "status_code": response.status_code,
            "n8n_response": response.text[:500],
        }
    except requests.RequestException as e:
        return {"sent": False, "reason": str(e)}
