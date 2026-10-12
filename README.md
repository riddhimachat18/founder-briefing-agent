# Founder Briefing Agent

**Tool-calling LLM agent (Gemini, FastAPI) that triages emails and notes into a prioritised brief with human-approved reply drafts. Defences against prompt injection, rate limiting, and a spend cap, with an injection test suite and 10-case eval.**

🚀 **[Live demo →](https://your-cloudrun-url.run.app)** *(deploy and update this link)*

---

## What it does

Paste raw emails, meeting notes, or a leads CSV. The agent classifies and prioritises every item, drafts a reply for anything that needs one, and creates follow-up tasks — all in one brief. You review, edit, approve or reject each draft. Nothing is ever sent automatically.

---

## Architecture

```
Browser (single page)
        │  POST /run  (text or CSV)
        ▼
FastAPI app/main.py
  ├─ slowapi rate limiter (10 req/hr per IP)        ← LLM10
  ├─ input sanitiser + injection check               ← LLM01
  ├─ spend cap (DAILY_TOKEN_CAP tokens/day)          ← LLM10
  │
  ▼
app/agent.py  — agent loop (max 6 iterations)
  │
  ├─ Gemini (gemini-1.5-flash, function calling)
  │        ↕  tool calls
  ├─ tools.py
  │     classify_and_summarise(text) → List[Item]
  │     draft_reply(item_id, intent) → Draft (status=pending)
  │     add_task(title, due, item_id) → Task
  │     search_web(query)             → wrapped result
  │
  ▼
app/store.py — SQLite (WAL mode)
  tables: briefs · items · drafts · tasks · spend_log

static/index.html — single-page UI
  Top-3 panel · urgency-sorted items
  Edit / Approve / Reject per draft
  Copy · Open in email app (mailto) on approval
```

---

## Security controls — OWASP LLM Top 10

| Risk | OWASP ID | Control |
|---|---|---|
| Prompt injection | LLM01 | Every email/note wrapped in `<untrusted_* id=N>` delimiters. System prompt declares content inside is *data, never instructions*. Zero-width and Bidi-override Unicode stripped. Per-item length capped at 8 000 chars. |
| Indirect injection | LLM01 | `search_web` results wrapped in `<untrusted_web_result>`. HTTPS-only, private IP ranges blocked (SSRF guard), 5 s timeout, 4 000 char limit. |
| Excessive agency | LLM06 | No `send_email` tool exists. `draft_reply` and `add_task` write **pending** rows only. Approval is a separate HTTP endpoint requiring a human click. Tool allow-list enforced in agent loop. |
| Output handling | LLM05 | Draft bodies stripped of HTML before storage. All tool arguments validated against Pydantic schemas before execution. |
| Unbounded consumption | LLM10 | `slowapi` rate limit (10 req/hr per IP). Max input 50 000 chars. Agent loop capped at 6 iterations + 32 000 token budget. Global `DAILY_TOKEN_CAP` (default 500 000 tokens); agent refuses once exceeded. |
| Secrets | LLM09 | API key in env var / Secret Manager only. `.env` in `.gitignore`. Run `gitleaks` in CI. |
| Logging | — | Tool calls and decisions logged; full email bodies never logged. |

---

## Eval results

Run `GEMINI_API_KEY=<key> python eval/eval.py` to reproduce.

| Metric | Score |
|---|---|
| Priority order (top-3 overlap) | **[run eval]** |
| `needs_reply` F1 | **[run eval]** |
| Injection defence (pass / fail) | **[run eval]** |
| **Overall** | **[run eval]** |

*Fill in real numbers after running the eval script.*

---

## Injection test suite

```bash
pytest tests/test_injection.py -v
```

Tests cover:
- Known attack phrases (`ignore previous instructions`, `reveal system prompt`, `wire $50k`, `send email to`, `<script>` injection)
- Zero-width character stripping
- Tool allow-list enforcement (blocks `send_email`, `wire_funds`, `execute_command`)
- Injected emails correctly flagged `suspicious=True`
- HTML stripped from draft bodies

---

## Local setup

```bash
# 1. Clone
git clone https://github.com/your-org/founder-briefing-agent
cd founder-briefing-agent

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Open
open http://localhost:8000
```

---

## Run tests

```bash
# Unit + security tests (no API key needed)
pytest tests/ -v

# Full eval (requires GEMINI_API_KEY)
GEMINI_API_KEY=<key> python eval/eval.py
```

---

## Docker

```bash
docker build -t founder-brief .
docker run -p 8080:8080 -e GEMINI_API_KEY=<key> founder-brief
```

---

## Deploy to Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/founder-brief

# Deploy
gcloud run deploy founder-brief \
  --image gcr.io/YOUR_PROJECT/founder-brief \
  --platform managed \
  --region us-central1 \
  --max-instances 2 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --allow-unauthenticated
```

Set a GCP billing budget alert (Console → Billing → Budgets) before enabling public traffic.

---

## Limitations

- **No real-time sending.** By design — all drafts require human approval and copy/mailto only.
- **Single-user SQLite.** Fine for a personal demo; swap for Postgres for multi-user production.
- **No email-client integration.** The mailto link opens your default mail client; credentials are never touched.
- **Eval is heuristic.** Ground-truth labels in `eval/eval.py` are manually authored against the sample set; real inboxes will vary.
- **Gemini function-calling API may change.** Tested against `google-generativeai==0.8.3`.

---

## Repo layout

```
app/
  main.py        FastAPI app, routes, rate limiter
  agent.py       Agent loop (~40 lines of logic)
  tools.py       classify_and_summarise, draft_reply, add_task, search_web
  schemas.py     Pydantic models: Item, Draft, Task, Brief
  security.py    Sanitiser, injection detection, SSRF guard, allow-list
  store.py       SQLite persistence, spend tracking
static/
  index.html     Single-page UI
samples/
  emails.txt     10 fake emails (inc. injection attack)
  leads.csv      20-row fake leads CSV
tests/
  test_injection.py   Injection + security tests
  test_agent.py       Tool unit tests
eval/
  eval.py        Priority, needs_reply, injection scoring
Dockerfile
requirements.txt
.env.example
README.md
```
