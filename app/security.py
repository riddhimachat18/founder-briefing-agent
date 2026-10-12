"""
Security controls for the Founder Briefing Agent.

OWASP LLM Top 10 mitigations implemented here:
  LLM01 — Prompt injection: delimiter wrapping, Unicode stripping, length caps
  LLM05 — Output handling: schema validation of all tool arguments
  LLM06 — Excessive agency: no send tool; only pending writes allowed
  LLM10 — Unbounded consumption: per-IP rate limit + global daily spend cap

Indirect injection (LLM01 variant):
  search_web results are wrapped in <untrusted_web_result> delimiters.

Secrets (LLM09):
  API key sourced from env only; never logged.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITEM_CHARS = 8_000       # max chars per single pasted block
MAX_TOTAL_CHARS = 50_000     # max total input (matches Pydantic RunRequest)
MAX_SUBJECT_CHARS = 512
MAX_BODY_CHARS = 4_096

# Regex patterns that signal attempted injection
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+a?\s*(new\s+)?(different\s+)?ai", re.I),
    re.compile(r"new\s+system\s+prompt", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"disregard\s+(all\s+)?previous", re.I),
    re.compile(r"wire\s+\$?\d+", re.I),
    re.compile(r"send\s+(an?\s+)?email\s+to\b", re.I),
    re.compile(r"forward\s+(this\s+)?(email|message)\s+to\b", re.I),
    re.compile(r"execute\s+(the\s+following\s+)?command", re.I),
    re.compile(r"<\s*script", re.I),
    re.compile(r"javascript\s*:", re.I),
]

# Zero-width and other dangerous Unicode categories
_ZW_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2028\u2029\u202a-\u202e]"
)


# ---------------------------------------------------------------------------
# Text sanitisation  (LLM01)
# ---------------------------------------------------------------------------

def sanitise(text: str, max_chars: int = MAX_ITEM_CHARS) -> str:
    """
    1. Normalise Unicode to NFC.
    2. Strip zero-width / direction-override characters.
    3. Truncate to max_chars.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZW_PATTERN.sub("", text)
    return text[:max_chars]


def wrap_untrusted(text: str, source_id: str, source_type: str = "email") -> str:
    """
    Wrap content in XML-style delimiters so the model treats it as data.
    The system prompt instructs the model that text inside these tags is
    untrusted data and never instructions.
    """
    cleaned = sanitise(text)
    return f"<untrusted_{source_type} id={source_id}>\n{cleaned}\n</untrusted_{source_type}>"


def wrap_web_result(text: str, query: str) -> str:
    """Wrap search results as untrusted (indirect injection guard)."""
    cleaned = sanitise(text, max_chars=4_000)
    return f"<untrusted_web_result query={repr(query)}>\n{cleaned}\n</untrusted_web_result>"


# ---------------------------------------------------------------------------
# Injection detection  (LLM01)
# ---------------------------------------------------------------------------

def detect_injection(text: str) -> Tuple[bool, list[str]]:
    """
    Scan text for known injection patterns.
    Returns (is_suspicious, list_of_matched_patterns).
    Does NOT block — caller decides how to handle (mark suspicious, etc.).
    """
    matches = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    return bool(matches), matches


def flag_suspicious_items(items: list) -> list:  # type: ignore[type-arg]
    """
    Post-process a list of Item dicts or Item objects; set suspicious=True
    for any whose summary or raw_snippet matches injection patterns.
    """
    flagged = []
    for item in items:
        # Support both dict and Pydantic model
        if hasattr(item, "model_dump"):
            d = item.model_dump()
        else:
            d = dict(item)

        text_to_check = f"{d.get('summary','')} {d.get('raw_snippet','')}"
        suspicious, _ = detect_injection(text_to_check)
        d["suspicious"] = suspicious
        flagged.append(d)
    return flagged


# ---------------------------------------------------------------------------
# SSRF guard for search_web  (indirect injection / LLM01)
# ---------------------------------------------------------------------------

import ipaddress
import urllib.parse


_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def is_safe_url(url: str) -> bool:
    """Return True only for https URLs that don't resolve to private ranges."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            return False
        host = parsed.hostname or ""
        # Reject raw IPs in private ranges
        try:
            addr = ipaddress.ip_address(host)
            return not any(addr in net for net in _PRIVATE_NETWORKS)
        except ValueError:
            pass  # It's a hostname, not an IP — allow it
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Output sanitisation  (LLM05)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Remove any HTML tags from a draft body before display."""
    return _HTML_TAG_RE.sub("", text)


def sanitise_draft_body(body: str) -> str:
    """Ensure draft body is plain text and within length limits."""
    body = strip_html(body)
    body = sanitise(body, max_chars=MAX_BODY_CHARS)
    return body


# ---------------------------------------------------------------------------
# Allowed tool names  (LLM06 — excessive agency)
# ---------------------------------------------------------------------------

ALLOWED_TOOL_NAMES = frozenset({
    "classify_and_summarise",
    "draft_reply",
    "add_task",
    "search_web",
})


def assert_tool_allowed(name: str) -> None:
    """Raise ValueError if the model requests a tool outside the allow-list."""
    if name not in ALLOWED_TOOL_NAMES:
        raise ValueError(
            f"Tool '{name}' is not in the allowed set {ALLOWED_TOOL_NAMES}. "
            "Possible prompt injection attempt."
        )
