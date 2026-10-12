"""
Pydantic schemas for the Founder Briefing Agent.
All IDs are UUIDs generated at creation time.
"""

from __future__ import annotations

import uuid
from datetime import date
from enum import IntEnum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class Urgency(IntEnum):
    LOW = 1
    BELOW_AVERAGE = 2
    AVERAGE = 3
    HIGH = 4
    CRITICAL = 5


CATEGORIES = Literal[
    "investor",
    "customer",
    "operations",
    "legal",
    "hr",
    "partnership",
    "spam",
    "scheduling",
    "complaint",
    "other",
]

TONES = Literal["formal", "friendly", "assertive", "empathetic"]

DRAFT_STATUS = Literal["pending", "approved", "rejected"]


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Item(BaseModel):
    """A single classified item extracted from the raw input."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: Literal["email", "meeting_note", "lead_csv"] = "email"
    sender: str = Field(..., max_length=256)
    subject: str = Field(default="", max_length=512)
    summary: str = Field(..., max_length=1024)
    urgency: Urgency
    category: CATEGORIES  # type: ignore[valid-type]
    needs_reply: bool
    suspicious: bool = False          # set True if injection detected
    raw_snippet: str = Field(default="", max_length=512)  # first 512 chars only

    @field_validator("sender")
    @classmethod
    def strip_sender(cls, v: str) -> str:
        return v.strip()


class Draft(BaseModel):
    """A reply draft for a given Item.  Never auto-sent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_id: str
    subject: str = Field(..., max_length=512)
    body: str = Field(..., max_length=4096)
    tone: TONES = "formal"  # type: ignore[valid-type]
    status: DRAFT_STATUS = "pending"  # type: ignore[valid-type]


class Task(BaseModel):
    """A follow-up action item linked to an Item."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., max_length=256)
    due: Optional[date] = None
    item_id: Optional[str] = None
    done: bool = False


class Brief(BaseModel):
    """The final structured output returned by the agent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: List[Item]
    drafts: List[Draft]
    tasks: List[Task]
    top_3: List[str]          # item IDs of the three highest-priority items
    created_at: str = ""      # ISO timestamp filled by the store


# ---------------------------------------------------------------------------
# API request / response wrappers
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Payload posted to POST /run."""

    text: str = Field(..., min_length=1, max_length=50_000)


class ApproveRequest(BaseModel):
    """Payload posted to POST /drafts/{draft_id}/approve."""

    edited_body: Optional[str] = Field(default=None, max_length=4096)


class RejectRequest(BaseModel):
    """Payload posted to POST /drafts/{draft_id}/reject."""

    reason: Optional[str] = Field(default=None, max_length=256)
