"""
FastAPI application — Founder Briefing Agent.

Routes:
  POST /run                        → run agent, return Brief
  GET  /briefs                     → list recent briefs
  GET  /briefs/{brief_id}          → fetch single brief
  POST /drafts/{draft_id}/approve  → approve (optionally edit) a draft
  POST /drafts/{draft_id}/reject   → reject a draft
  GET  /health                     → liveness probe

Security:
  - slowapi rate limiter: 10 requests / hour per IP   (LLM10)
  - global daily spend cap checked before each run    (LLM10)
  - input sanitised before hitting the agent          (LLM01)
  - CORS restricted to same-origin in production
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import run_agent
from app.schemas import ApproveRequest, Brief, RejectRequest, RunRequest
from app.security import sanitise
from app.store import (
    budget_exceeded,
    get_brief,
    get_draft,
    init_db,
    list_briefs,
    log_tokens,
    save_brief,
    update_draft_status,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (LLM10)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["10/hour"])

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Founder Briefing Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Global catch-all — always return JSON so the UI can show the message
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal server error"},
    )

# CORS — lock down in prod; allow localhost for local dev
_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Static files (index.html lives in /static)
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup() -> None:
    init_db()
    logger.info("DB initialised.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Founder Briefing Agent API"})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=Brief, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
async def run(request: Request, body: RunRequest) -> Brief:
    """
    Main entry point.  Accepts pasted text (emails, notes, CSV) and
    returns a prioritised Brief with drafts and tasks.

    Rate limited: 10 requests / hour per IP.
    Blocked when the global daily token spend cap is reached.
    """
    if budget_exceeded():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily token budget exhausted. Try again tomorrow.",
        )

    # Sanitise input before passing to agent (LLM01)
    clean_input = sanitise(body.text, max_chars=50_000)

    request_id = str(uuid.uuid4())[:8]
    logger.info("run [%s] input_len=%d ip=%s", request_id, len(clean_input), get_remote_address(request))

    try:
        brief = run_agent(clean_input, request_id=request_id)
    except Exception as exc:
        logger.exception("[%s] agent failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    saved = save_brief(brief)

    # Log token spend (approximate — real count in agent.py)
    log_tokens(ip=get_remote_address(request), tokens=len(clean_input) // 4)

    return saved


@app.get("/briefs", response_model=List[dict])
async def get_briefs() -> list:
    """Return the 50 most recent brief summaries."""
    return list_briefs()


@app.get("/briefs/{brief_id}", response_model=Brief)
async def get_brief_by_id(brief_id: str) -> Brief:
    brief = get_brief(brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found.")
    return brief


@app.post("/drafts/{draft_id}/approve")
async def approve_draft(draft_id: str, body: ApproveRequest) -> dict:
    """
    Human-approval endpoint.  Updates the draft status to 'approved'
    and optionally saves an edited body.
    This is the ONLY way a draft transitions out of 'pending'.
    No email is ever sent automatically.   (LLM06)
    """
    draft = get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found.")
    if draft.status == "approved":
        raise HTTPException(status_code=409, detail="Draft already approved.")

    updated = update_draft_status(
        draft_id,
        status="approved",
        body=body.edited_body,
    )
    logger.info("draft approved: %s", draft_id)
    return {"status": "approved", "draft": updated}


@app.post("/drafts/{draft_id}/reject")
async def reject_draft(draft_id: str, body: RejectRequest) -> dict:
    """Mark a draft as rejected.  Reason is logged but not stored."""
    draft = get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found.")

    updated = update_draft_status(draft_id, status="rejected")
    logger.info("draft rejected: %s reason=%s", draft_id, body.reason or "none")
    return {"status": "rejected", "draft": updated}
