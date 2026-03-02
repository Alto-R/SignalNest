# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

SignalNest is a personal AI daily digest service. It aggregates content from GitHub Trending, YouTube channels, and RSS feeds, runs it through a two-stage AI filtering pipeline, and dispatches the result via email, Feishu, or WeCom on a cron schedule (managed by Docker + supercronic).

## Development Commands

### Run locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in credentials
cp docker/.env.example docker/.env

# Run once (dry-run preview, no notifications sent)
python -m src.main --schedule-name "早间日报" --dry-run

# Run once for real
python -m src.main --schedule-name "早间日报"

# Run with the first schedule (omit --schedule-name)
python -m src.main
```

Environment variables are loaded from `docker/.env` (via `python-dotenv`) when running locally. In Docker, env vars are injected by docker-compose.

### Docker commands

```bash
cd docker/

# First run / after code changes
docker compose up -d --build

# After editing config.yaml or personal/*.yaml only
docker compose restart

# After changing .env
docker compose up -d --force-recreate

# Trigger immediately without waiting for cron
# Set IMMEDIATE_RUN=true and SCHEDULE_NAME=早间日报 in docker/.env first
docker compose up -d --force-recreate

# View logs
docker logs -f signalnest

# Stop
docker compose down
```

There are no automated tests in this project.

## Architecture

### Data flow

```
main.py::run()
  │
  ├─ personal/  (if content_blocks includes "schedule"/"todos")
  │   ├── schedule_reader.py  → reads config/personal/schedule.yaml
  │   └── todo_reader.py      → reads config/personal/todos.yaml
  │
  ├─ collectors/  (if content_blocks includes "news")
  │   ├── github_collector.py  → scrapes github.com/trending
  │   ├── youtube_collector.py → YouTube Data API v3
  │   └── rss_collector.py     → feedparser
  │
  ├─ ai/summarizer.py
  │   ├─ Stage 1: batch title filtering (1 LiteLLM call)
  │   ├─ [YouTube only] fetch transcripts for shortlisted videos
  │   ├─ [RSS only] cap per-feed candidates
  │   └─ Stage 2: per-item score + summary (N LiteLLM calls)
  │
  ├─ ai/feedback.py  → SQLite feedback.db, taste examples for few-shot
  │
  └─ notifications/dispatcher.py
      ├── email_sender.py   → SMTP HTML email
      ├── feishu_sender.py  → Feishu webhook
      └── wework_sender.py  → WeCom webhook
```

### Configuration layering

- `config/config.yaml` — all non-secret settings (schedules, collector params, AI params, notification toggles)
- `docker/.env` — secrets only: `AI_API_KEY`, `AI_MODEL`, `AI_API_BASE`, `EMAIL_FROM`, `EMAIL_PASSWORD`, `GITHUB_TOKEN`, `YOUTUBE_API_KEY`, `FEISHU_WEBHOOK_URL`, `WEWORK_WEBHOOK_URL`
- Environment variables always take precedence over `config.yaml` fallback values (see `summarizer.py` for the pattern: `os.environ.get("AI_MODEL") or ai_cfg.get("model", ...)`)

### Key config paths

| Config key | Description |
|---|---|
| `schedules[].content` | List from `[news, schedule, todos]` — controls which modules run |
| `schedules[].sources` | List from `[github, youtube, rss]` — which collectors are called |
| `schedules[].focus` | Free-text topic direction fed to AI scoring prompt |
| `ai.min_relevance_score` | Items below this (1-10) are filtered out |
| `ai.max_items_per_digest` | Max items in final output |
| `ai.min_items_per_source` | Per-source floor enforced at two stages: title filtering + final output |

### AI model configuration

Uses LiteLLM — supports any OpenAI-compatible API. Model format is `provider/model_name` (e.g. `openai/gpt-4o`, `gemini/gemini-2.0-flash`). Set `AI_API_BASE` for proxy/relay endpoints.

### Feedback / preference learning loop

Each run saves `data/last_digest.json`. Users can fill in `user_score` (1-5) on any item; the next run reads these scores, writes them to `data/feedback.db` (SQLite via `ai/feedback.py`), and uses the top-scored historical items as few-shot examples in the AI system prompt.

### Privacy routing in email

`schedule` and `todos` content (personal data) is only sent to the `EMAIL_FROM` address (the sender/owner). Other recipients in `EMAIL_TO` only receive the news section. This logic lives in `email_sender.py`.

### Docker entrypoint behavior

`docker/entrypoint.sh` supports two modes via `RUN_MODE` env var:
- `cron` (default): reads `config.yaml`, generates a crontab dynamically, starts supercronic
- `once`: runs `src.main` immediately and exits

`IMMEDIATE_RUN=true` in `cron` mode causes one immediate execution before handing off to supercronic.

## Important Files

| File | Role |
|---|---|
| `src/main.py` | Top-level orchestrator — the entry point |
| `src/config_loader.py` | Merges `config.yaml` + `.env` into a single config dict |
| `src/ai/summarizer.py` | Two-stage AI filtering engine (core logic, most complex file) |
| `src/ai/feedback.py` | SQLite feedback store + taste example loader |
| `src/notifications/dispatcher.py` | Routes payload to enabled notification channels |
| `config/config.yaml` | Primary user-facing configuration |
| `docker/.env` | Secrets (never commit) |
| `docker/entrypoint.sh` | Container startup — generates crontab from config.yaml |
