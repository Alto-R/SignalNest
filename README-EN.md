<div align="center">

# SignalNest 📡

A self-hosted personal AI digest system — scheduled aggregation from GitHub Trending / YouTube / RSS, two-stage AI filtering and summarization, delivered via Email / Feishu / WeCom.

[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-quick-start)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Scheduler](https://img.shields.io/badge/Scheduler-supercronic-orange?style=flat-square)](https://github.com/aptible/supercronic)
[![Mode](https://img.shields.io/badge/Executor-agent%20(default)-00A86B?style=flat-square)](#-agent-mode)

**[中文](README.md)** | **[English](README-EN.md)**

</div>

<br>

## 📑 Quick Navigation

<div align="center">

| | | |
|:---:|:---:|:---:|
| [🚀 Quick Start](#-quick-start) | [⚙️ Configuration](#️-configuration) | [🤖 Agent Mode](#-agent-mode) |
| [💻 Local Development](#-local-development-without-docker) | [🗂️ Data & Preference Learning](#️-data--preference-learning) | [❓ FAQ](#-faq) |

</div>

<br>

## ✨ Key Features

- Multi-source ingestion: compose `github` / `youtube` / `rss` per schedule.
- Two-stage AI pipeline: batch title filtering first, then per-item scoring and summaries.
- History deduplication: inject recent 7-day titles from `data/history/` to reduce repeats.
- Per-source minimums: `ai.min_items_per_source` (default GitHub>=5, YouTube>=2).
- Dual-track YouTube: subscribed channels + optional keyword search derived from `focus`.
- Personal assistant blocks: parse `config/personal/schedule.md` and `projects.md`, with optional per-recipient files `schedule-<name>.md` / `projects-<name>.md`.
- Preference learning: `user_score` in `data/last_digest.json` is persisted into `feedback.db`.
- Multi-channel delivery: HTML email + Feishu webhook + WeCom webhook.
- Privacy split: `EMAIL_FROM` keeps default personal blocks; other recipients get personal blocks only when matching named files exist, otherwise news-only.
- Agent-only execution path: local agent scheduling is the only runtime flow.

## 🏗️ Architecture Overview

```text
Docker entrypoint (supercronic)
  -> python -m src.main --schedule-name <name>

main.py (agent-only)
  -> agent.kernel            (LLM planning + tool calls + session persistence)
  -> agent.tools             (collect/summarize/payload/dispatch tools)
  -> ai.feedback + data/history + data/last_digest.json
```

## 🚀 Quick Start

### 1) Configure Docker environment variables

```bash
cd SignalNest/docker
cp .env.example .env
```

Edit `docker/.env` with at least:

- `AI_API_KEY`, `AI_MODEL` (when `AI_BACKEND=litellm`)
- `EMAIL_FROM`, `EMAIL_PASSWORD`, `EMAIL_TO` (use `name:email` format in `EMAIL_TO` to enable per-recipient personalization)

Common optional fields:

- `GITHUB_TOKEN`
- `YOUTUBE_API_KEY`
- `FEISHU_WEBHOOK_URL`
- `WEWORK_WEBHOOK_URL`
- `EMAIL_OPENING_AI_NAMES` (opening-line enabled names, default `yy`)
- `EMAIL_OPENING_YY` (manual opening line for `yy`; manual text takes priority, AI is fallback)

### 2) Prepare personal files (optional)

```bash
cd ..
cp config/personal/schedule_example.md config/personal/schedule.md
cp config/personal/projects_example.md config/personal/projects.md

# Optional: recipient-specific files (name must exactly match EMAIL_TO name)
cp config/personal/schedule_example.md config/personal/schedule-yy.md
cp config/personal/projects_example.md config/personal/projects-yy.md
```

`schedule.md` and `projects.md` can be free-form Markdown. Structure extraction is handled by AI.  
If `EMAIL_TO` includes `yy:foo@example.com`, the system will try `schedule-yy.md` and `projects-yy.md` and include whichever personal blocks are available in yy's email.

### 3) Configure schedules

Edit `config/config.yaml`. Key fields:

- `schedules[].name`
- `schedules[].cron`
- `schedules[].content`: `news` / `schedule` / `todos`
- `schedules[].sources`: `github` / `youtube` / `rss`
- `schedules[].focus`: priority direction for AI scoring
- `schedules[].subject_prefix`: notification title prefix

### 4) Launch

```bash
cd docker
docker compose up -d --build
```

Check logs:

```bash
docker logs -f signalnest
```

### 5) Trigger one immediate run (optional)

Set in `docker/.env`:

```dotenv
IMMEDIATE_RUN=true
SCHEDULE_NAME=Morning Digest
```

Then run:

```bash
docker compose up -d --force-recreate
```

## 💻 Local Development (without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare local env file:

```bash
cp docker/.env.example .env
```

For local runs, `src/config_loader.py` reads repository-root `.env` (not `docker/.env`).

### Agent-only flow

```bash
python -m src.main --schedule-name "Morning Digest" --dry-run
python -m src.main --schedule-name "Morning Digest"
```

## ⚙️ Configuration

### Core config blocks

| Config Path | Purpose |
| --- | --- |
| `app.timezone` / `app.language` | Global timezone and output language |
| `schedules[]` | Cron plans and content composition |
| `collectors.github` | GitHub Trending collection parameters |
| `collectors.youtube` | Subscribed-channel + keyword-search parameters |
| `collectors.rss` | RSS window and per-feed limits |
| `ai.*` | Backend/model/threshold/concurrency/source minimums |
| `agent.*` | Agent step limits and policy (including `max_steps_hard_limit`) |
| `notifications.*` | Delivery channel toggles |
| `storage.*` | Data directory and todo lookahead window |

### AI backends

| Backend | Description | API Key Required |
| --- | --- | :---: |
| `litellm` (default) | Calls OpenAI-compatible cloud APIs | Yes (`AI_API_KEY`) |
| `claude-cli` | Uses local `claude --print` | No |
| `codex-cli` | Uses local `codex -q` | No |

Environment variables override `config.yaml`:

- `AI_BACKEND`
- `AI_MODEL`
- `AI_API_BASE`
- `AI_API_KEY`

### Docker execution modes

`docker/entrypoint.sh` supports:

- `RUN_MODE=cron` (default): parse `config.yaml`, generate crontab, keep running with supercronic.
- `RUN_MODE=once`: execute one schedule and exit.

Scheduler command is fixed to: `python -m src.main --schedule-name <name>` (agent-only).

## 🤖 Agent Mode

Agent core lives in `src/agent/`, following “LLM planning -> tool calls -> state persistence”.

### Built-in tools

- `collect_github`
- `collect_rss`
- `collect_youtube`
- `collect_all_news`
- `summarize_news`
- `read_today_schedule`
- `read_active_projects`
- `build_digest_payload`
- `dispatch_notifications` (side-effect tool)

### Policy and persistence

- Tool policy: allowlist / denylist / `allow_side_effects` (driven only by `config.agent.policy`)
- Step config: `agent.max_steps` / `agent.schedule_max_steps`, uniformly capped by `agent.max_steps_hard_limit`
- Schedule side-effect switch: `agent.schedule_allow_side_effects` (controls real sends in `--schedule-name`)
- Context lookback window: `agent.recent_turns_context_limit` (inject last N turns into planner prompt)
- Session DB: `data/agent_sessions.db`
- Scheduled runs are always persisted (fixed behavior, no user-facing toggle)
- Config convergence: agent runtime parameters are centralized in `config.agent` with strict startup validation

## 🗂️ Data & Preference Learning

Key runtime files:

- `data/last_digest.json`: latest items with editable `user_score`
- `data/history/digest_*.json`: archived run snapshots
- `data/feedback.db`: persisted user feedback
- `data/agent_sessions.db`: agent sessions and tool traces

Feedback loop:

1. Edit `data/last_digest.json` and set `user_score` to `1-5` for preferred items.
2. Next run imports those scores into `feedback.db` automatically.
3. Future summarization uses high-score records as taste examples.

## ❓ FAQ

### Email failed with 535 authentication error

Use SMTP app password / authorization code, not your mailbox login password.

### Local run cannot read environment variables

Make sure `.env` is in repository root. Local mode does not read `docker/.env`.

### YouTube collector returned no items

Check:

- `collectors.youtube.enabled=true`
- `YOUTUBE_API_KEY` is set
- `days_lookback` / `search_days_lookback` are not too short

### Keyword search does not work with `claude-cli` / `codex-cli`

`youtube_collector._ai_generate_keywords()` currently hardcodes LiteLLM + `AI_API_KEY`. Without `AI_API_KEY`, keyword search is skipped, while subscribed-channel collection still works.

### Notifications enabled but run still fails

`dispatch()` requires at least one enabled channel to succeed by default. Verify SMTP/webhook credentials and connectivity.

## 🔐 Security Notes

- Never commit `docker/.env` or repository-root `.env`.
- `config/personal/schedule.md`, `projects.md`, and `schedule-*.md` / `projects-*.md` may contain private data and should remain local.
- Rotate keys/passwords immediately if leaked.

## 📚 Credits

Inspired by:

- [TrendRadar](https://github.com/sansan0/TrendRadar)
- [obsidian-daily-digest](https://github.com/Lantern567/obsidian-daily-digest)

