"""
Agent tools — plain functions with typed signatures.
Each tool is also described as a Gemini function declaration in TOOL_DECLARATIONS.

Security notes:
  - classify_and_summarise: input is already sanitised + wrapped by security.py
  - draft_reply: writes ONLY with status='pending'; no auto-send path exists
  - add_task: writes task row; no external side effects
  - search_web: validates URL (https + no private IPs), wraps result as untrusted
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from app.schemas import Draft, Item, Task, Urgency
from app.security import (
    detect_injection,
    is_safe_url,
    sanitise_draft_body,
    wrap_web_result,
)

# ---------------------------------------------------------------------------
# In-process state shared with agent.py
# Drafts and tasks are accumulated here during a run, then persisted by main.py
# ---------------------------------------------------------------------------

_run_items: List[Item] = []
_run_drafts: List[Draft] = []
_run_tasks: List[Task] = []


def reset_run_state() -> None:
    global _run_items, _run_drafts, _run_tasks
    _run_items = []
    _run_drafts = []
    _run_tasks = []


def get_run_state() -> Dict[str, Any]:
    return {"items": _run_items, "drafts": _run_drafts, "tasks": _run_tasks}


# ---------------------------------------------------------------------------
# Tool: classify_and_summarise
# ---------------------------------------------------------------------------

def classify_and_summarise(text: str) -> List[Dict[str, Any]]:
    """
    Parse raw text (already sanitised) into a list of Item dicts.
    In the agent loop this is called with the model's extracted structured data,
    so here we just validate and store the items.

    Args:
        text: JSON string produced by the model representing a list of items.

    Returns:
        List of item dicts (serialised Items).
    """
    import json

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"classify_and_summarise: invalid JSON from model — {exc}")

    if not isinstance(raw, list):
        raw = [raw]

    result = []
    for entry in raw:
        # Detect injection in the summary the model produced
        suspicious, _ = detect_injection(entry.get("summary", ""))
        entry["suspicious"] = suspicious or entry.get("suspicious", False)
        entry.setdefault("id", str(uuid.uuid4()))

        item = Item(**entry)
        _run_items.append(item)
        result.append(item.model_dump())

    return result


# ---------------------------------------------------------------------------
# Tool: draft_reply
# ---------------------------------------------------------------------------

def draft_reply(item_id: str, intent: str = "", tone: str = "formal",
                # Aliases the model sometimes uses
                body: Optional[str] = None,
                content: Optional[str] = None,
                text: Optional[str] = None,
                **kwargs: Any) -> Dict[str, Any]:
    """
    Create a reply draft (status='pending') for the given item.
    The model provides the body text; we sanitise before storing.

    Args:
        item_id: ID of the Item this reply is for.
        intent: The body of the email to draft (plain text).
        tone: One of formal | friendly | assertive | empathetic.

    Returns:
        Draft dict with status='pending'.
    """
    # Accept common model aliases for the body field
    resolved_intent = intent or body or content or text or ""

    # Validate tone
    valid_tones = {"formal", "friendly", "assertive", "empathetic"}
    if tone not in valid_tones:
        tone = "formal"

    # Find subject from already-classified items
    subject = "Re: (your message)"
    for item in _run_items:
        if item.id == item_id:
            subject = f"Re: {item.subject}" if item.subject else "Re: (your message)"
            break

    safe_body = sanitise_draft_body(resolved_intent)

    draft = Draft(
        item_id=item_id,
        subject=subject,
        body=safe_body,
        tone=tone,  # type: ignore[arg-type]
        status="pending",
    )
    _run_drafts.append(draft)
    return draft.model_dump()


# ---------------------------------------------------------------------------
# Tool: add_task
# ---------------------------------------------------------------------------

def add_task(title: str = "", due: Optional[str] = None, item_id: Optional[str] = None,
             # Aliases the model sometimes uses
             description: Optional[str] = None,
             due_date: Optional[str] = None,
             task: Optional[str] = None,
             name: Optional[str] = None,
             **kwargs: Any) -> Dict[str, Any]:
    """
    Create a follow-up task.

    Args:
        title: Short action description (max 256 chars).
        due: Optional ISO date string (YYYY-MM-DD).
        item_id: Optional reference to an Item.

    Returns:
        Task dict.
    """
    from datetime import date

    # Accept common model hallucinations for the title field
    resolved_title = title or description or task or name or "Follow-up task"
    # Accept common model hallucinations for the due field
    resolved_due = due or due_date

    due_obj = None
    if resolved_due:
        try:
            due_obj = date.fromisoformat(str(resolved_due))
        except ValueError:
            due_obj = None

    t = Task(
        title=resolved_title[:256],
        due=due_obj,
        item_id=item_id,
    )
    _run_tasks.append(t)
    return t.model_dump()


# ---------------------------------------------------------------------------
# Tool: search_web  (optional — skip if no SERPER_API_KEY)
# ---------------------------------------------------------------------------

_SEARCH_TIMEOUT = 5  # seconds
_MAX_RESULT_CHARS = 4_000
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")


def search_web(query: str) -> Dict[str, Any]:
    """
    Optional web search using Serper.dev (Google Search API).
    Results are wrapped as untrusted to prevent indirect injection.

    Args:
        query: The search query string.

    Returns:
        Dict with 'results' key containing the wrapped snippet text.
    """
    if not SERPER_API_KEY:
        return {"results": "Search unavailable (no SERPER_API_KEY configured).", "query": query}

    endpoint = "https://google.serper.dev/search"
    if not is_safe_url(endpoint):
        return {"results": "Blocked: unsafe endpoint.", "query": query}

    try:
        resp = httpx.post(
            endpoint,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 3},
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", [])
        snippets = "\n\n".join(
            f"{r.get('title','')}: {r.get('snippet','')}" for r in organic[:3]
        )
    except Exception as exc:
        snippets = f"Search error: {exc}"

    wrapped = wrap_web_result(snippets[:_MAX_RESULT_CHARS], query)
    return {"results": wrapped, "query": query}


# ---------------------------------------------------------------------------
# OpenAI-format tool declarations (Groq uses the same schema)
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "classify_and_summarise",
            "description": (
                "Parse raw text into structured Items. "
                "Call this first with a JSON array of item objects you have extracted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "JSON array of item objects with keys: source, sender, subject, "
                            "summary, urgency (1-5), category, needs_reply (bool)."
                        ),
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_reply",
            "description": (
                "Write a reply draft for an item that needs_reply=true. "
                "Status is always 'pending'; nothing is sent automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "ID of the Item to reply to."},
                    "intent":  {"type": "string", "description": "Full plain-text email body to draft."},
                    "tone": {
                        "type": "string",
                        "enum": ["formal", "friendly", "assertive", "empathetic"],
                        "description": "Tone for the reply.",
                    },
                },
                "required": ["item_id", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Record a follow-up task related to an item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Short action description."},
                    "due":     {"type": "string", "description": "ISO date YYYY-MM-DD, optional."},
                    "item_id": {"type": "string", "description": "Item ID this task relates to, optional."},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for background context. Use sparingly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                },
                "required": ["query"],
            },
        },
    },
]

# Dispatch map used by the agent loop
TOOL_DISPATCH: Dict[str, Any] = {
    "classify_and_summarise": classify_and_summarise,
    "draft_reply": draft_reply,
    "add_task": add_task,
    "search_web": search_web,
}
