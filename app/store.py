"""
SQLite persistence layer.
All writes go through this module.  The DB path is read from
the BRIEF_DB_PATH env var (default: ./data/briefs.db).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

from app.schemas import Brief, Draft, Item, Task

_DB_PATH = os.getenv("BRIEF_DB_PATH", str(Path(__file__).parent.parent / "data" / "briefs.db"))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    _ensure_dir()
    with _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS briefs (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                top_3       TEXT NOT NULL,   -- JSON array of item IDs
                raw_json    TEXT NOT NULL    -- full Brief JSON for fast reload
            );

            CREATE TABLE IF NOT EXISTS items (
                id          TEXT PRIMARY KEY,
                brief_id    TEXT NOT NULL REFERENCES briefs(id),
                source      TEXT NOT NULL,
                sender      TEXT NOT NULL,
                subject     TEXT NOT NULL,
                summary     TEXT NOT NULL,
                urgency     INTEGER NOT NULL,
                category    TEXT NOT NULL,
                needs_reply INTEGER NOT NULL,
                suspicious  INTEGER NOT NULL DEFAULT 0,
                raw_snippet TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS drafts (
                id          TEXT PRIMARY KEY,
                brief_id    TEXT NOT NULL REFERENCES briefs(id),
                item_id     TEXT NOT NULL,
                subject     TEXT NOT NULL,
                body        TEXT NOT NULL,
                tone        TEXT NOT NULL DEFAULT 'formal',
                status      TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                brief_id    TEXT NOT NULL REFERENCES briefs(id),
                item_id     TEXT,
                title       TEXT NOT NULL,
                due         TEXT,
                done        INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS spend_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                ip          TEXT NOT NULL,
                tokens      INTEGER NOT NULL
            );
        """)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_dir()
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Brief persistence
# ---------------------------------------------------------------------------

def save_brief(brief: Brief) -> Brief:
    """Persist a full Brief and all its children.  Returns the Brief with
    created_at filled in."""
    now = datetime.now(timezone.utc).isoformat()
    brief.created_at = now

    with _conn() as db:
        db.execute(
            "INSERT INTO briefs (id, created_at, top_3, raw_json) VALUES (?,?,?,?)",
            (brief.id, now, json.dumps(brief.top_3), brief.model_dump_json()),
        )
        for item in brief.items:
            _upsert_item(db, brief.id, item)
        for draft in brief.drafts:
            _upsert_draft(db, brief.id, draft)
        for task in brief.tasks:
            _upsert_task(db, brief.id, task)

    return brief


def get_brief(brief_id: str) -> Optional[Brief]:
    with _conn() as db:
        row = db.execute("SELECT raw_json FROM briefs WHERE id=?", (brief_id,)).fetchone()
    if not row:
        return None
    brief = Brief.model_validate_json(row["raw_json"])
    # Reload live draft statuses from the drafts table
    brief.drafts = list_drafts(brief_id)
    return brief


def list_briefs() -> List[dict]:
    """Return lightweight list (id, created_at, top_3) for the sidebar."""
    with _conn() as db:
        rows = db.execute(
            "SELECT id, created_at, top_3 FROM briefs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "top_3": json.loads(r["top_3"])} for r in rows]


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------

def _upsert_item(db: sqlite3.Connection, brief_id: str, item: Item) -> None:
    db.execute(
        """INSERT OR REPLACE INTO items
           (id, brief_id, source, sender, subject, summary, urgency, category,
            needs_reply, suspicious, raw_snippet)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (item.id, brief_id, item.source, item.sender, item.subject,
         item.summary, int(item.urgency), item.category,
         int(item.needs_reply), int(item.suspicious), item.raw_snippet),
    )


# ---------------------------------------------------------------------------
# Draft helpers
# ---------------------------------------------------------------------------

def _upsert_draft(db: sqlite3.Connection, brief_id: str, draft: Draft) -> None:
    db.execute(
        """INSERT OR REPLACE INTO drafts
           (id, brief_id, item_id, subject, body, tone, status)
           VALUES (?,?,?,?,?,?,?)""",
        (draft.id, brief_id, draft.item_id, draft.subject,
         draft.body, draft.tone, draft.status),
    )


def list_drafts(brief_id: str) -> List[Draft]:
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM drafts WHERE brief_id=?", (brief_id,)
        ).fetchall()
    return [Draft(**dict(r)) for r in rows]


def get_draft(draft_id: str) -> Optional[Draft]:
    with _conn() as db:
        row = db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
    return Draft(**dict(row)) if row else None


def update_draft_status(draft_id: str, status: str, body: Optional[str] = None) -> Optional[Draft]:
    with _conn() as db:
        if body is not None:
            db.execute(
                "UPDATE drafts SET status=?, body=? WHERE id=?",
                (status, body, draft_id),
            )
        else:
            db.execute("UPDATE drafts SET status=? WHERE id=?", (status, draft_id))
        row = db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
    return Draft(**dict(row)) if row else None


def add_draft(brief_id: str, draft: Draft) -> Draft:
    with _conn() as db:
        _upsert_draft(db, brief_id, draft)
    return draft


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _upsert_task(db: sqlite3.Connection, brief_id: str, task: Task) -> None:
    db.execute(
        """INSERT OR REPLACE INTO tasks
           (id, brief_id, item_id, title, due, done)
           VALUES (?,?,?,?,?,?)""",
        (task.id, brief_id, task.item_id, task.title,
         task.due.isoformat() if task.due else None, int(task.done)),
    )


def list_tasks(brief_id: str) -> List[Task]:
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM tasks WHERE brief_id=?", (brief_id,)
        ).fetchall()
    return [_row_to_task(r) for r in rows]


def add_task(brief_id: str, task: Task) -> Task:
    with _conn() as db:
        _upsert_task(db, brief_id, task)
    return task


def _row_to_task(row: sqlite3.Row) -> Task:
    d = dict(row)
    d["done"] = bool(d["done"])
    return Task(**d)


# ---------------------------------------------------------------------------
# Spend tracking  (LLM10 — unbounded consumption)
# ---------------------------------------------------------------------------

DAILY_TOKEN_CAP = int(os.getenv("DAILY_TOKEN_CAP", "500000"))


def log_tokens(ip: str, tokens: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as db:
        db.execute(
            "INSERT INTO spend_log (ts, ip, tokens) VALUES (?,?,?)",
            (now, ip, tokens),
        )


def daily_tokens_used() -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with _conn() as db:
        row = db.execute(
            "SELECT COALESCE(SUM(tokens),0) AS total FROM spend_log WHERE ts >= ?",
            (today,),
        ).fetchone()
    return int(row["total"])


def budget_exceeded() -> bool:
    return daily_tokens_used() >= DAILY_TOKEN_CAP
