"""
cv_api.py

FastAPI router for the CV screening pipeline. Mount this in main.py
alongside the existing /api/upload and /api/chat routes:

    from cv_api import router as cv_router
    app.include_router(cv_router)

Kept as a separate router (rather than editing main.py's endpoints
directly) so the original book-tutor RAG functionality stays untouched
and testable independently.
"""

import os
import json
import shutil
from typing import List

from fastapi import APIRouter, UploadFile, File, Form

from cv_screening_engine import evaluate_cvs_bulk, send_to_n8n

router = APIRouter()

TEMP_CV_DIR = "./temp_cvs"
os.makedirs(TEMP_CV_DIR, exist_ok=True)


@router.post("/api/screen-cvs")
async def screen_cvs(
    files: List[UploadFile] = File(...),
    criteria: str = Form(...),  # JSON string — see criteria shape below
):
    """
    Bulk CV screening endpoint.

    `files`: multiple CV PDFs uploaded in one request (requirement 1).
    `criteria`: a JSON string in the request form data, e.g.:

        {
          "required_skills": ["Python", "FastAPI", "SQL"],
          "education": "Bachelor's in Computer Science or related field",
          "experience": "1+ years, internships count",
          "projects": "At least one deployed/working project, not just coursework",
          "certifications": ["AWS", "Google"],
          "selection_threshold": 70
        }

    Every field is optional except selection_threshold (defaults to 70 if
    omitted) — this is what satisfies requirement 2 (configurable, not
    hardcoded): the admin sets these per screening run via the request,
    nothing is baked into the code.
    """
    try:
        criteria_dict = json.loads(criteria)
    except json.JSONDecodeError:
        return {"error": "criteria must be a valid JSON string"}

    saved_paths = []
    try:
        for file in files:
            file_path = os.path.join(TEMP_CV_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)

        results = evaluate_cvs_bulk(saved_paths, criteria_dict)

        n8n_status = send_to_n8n(results)

        return {
            "candidates_processed": len(results),
            "results": results,
            "n8n_dispatch": n8n_status,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up temp files regardless of success/failure
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)


@router.get("/api/screen-cvs/health")
async def health_check():
    """Quick check that the CV screening router is mounted and the
    N8N_WEBHOOK_URL env var is set, without processing anything."""
    return {
        "status": "ok",
        "n8n_configured": bool(os.getenv("N8N_WEBHOOK_URL")),
    }
