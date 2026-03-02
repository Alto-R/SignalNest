<div align="center">

# SignalNest 📡

A self-hosted personal AI digest service — aggregates GitHub / YouTube / RSS, filters with AI, delivered straight to your inbox on a schedule.

[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-docker-deployment)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![supercronic](https://img.shields.io/badge/scheduler-supercronic-orange?style=flat-square)](https://github.com/aptible/supercronic)

[![Email](https://img.shields.io/badge/Email-HTML_Rich_Text-00D4AA?style=flat-square)](#)
[![Feishu](https://img.shields.io/badge/Feishu-Webhook-00D4AA?style=flat-square)](https://www.feishu.cn/)
[![WeCom](https://img.shields.io/badge/WeCom-Webhook-00D4AA?style=flat-square)](https://work.weixin.qq.com/)

**[中文](README.md)** | **[English](README-EN.md)**

</div>

<br>

## 📑 Quick Navigation

<div align="center">

| | | |
|:---:|:---:|:---:|
| [🚀 Quick Start](#-quick-start) | [⚙️ Configuration](#️-configuration) | [🐳 Docker Deployment](#-docker-deployment) |
| [🎯 Features](#-features) | [🧠 Preference Learning](#-preference-learning) | [❓ FAQ](#-faq) |

</div>

<br>

## 🎯 Features

- **Three source types**: GitHub trending repos / YouTube curated videos / RSS feeds — mix and match
- **Focus-based filtering**: each schedule has a `focus` field; the AI prioritizes content that matches it
- **Two-stage AI pipeline**: batch title filtering (low token cost) → deep read scoring (only for selected items)
- **Per-source minimums**: configurable minimum item counts per source (default GitHub≥5, YouTube≥2) to reduce source imbalance
- **YouTube dual-track**: subscribed channels support views/newest ordering + AI auto-generates search keywords from `focus` to discover other channels
- **Preference learning**: rate items to teach the AI your taste — recommendations improve over time
- **Personal assistant**: morning schedule reminder + TODO due-date checker (overdue / today / upcoming)
- **Multi-channel delivery**: Email (HTML) + Feishu + WeCom, with per-recipient content splitting
- **Multi-schedule**: define any number of cron triggers in `config.yaml`, push different content at different times
- **One-command Docker deploy**: powered by supercronic, stable in-container scheduling

<br>

## 🚀 Quick Start

### Step 1: Configure environment variables

```bash
cd SignalNest/docker/
cp .env.example .env
```

Edit `docker/.env` with required fields:

```dotenv
# AI (required)
AI_API_KEY=your_api_key_here
AI_MODEL=openai/gpt-4o          # LiteLLM format: provider/model_name
AI_API_BASE=                    # Custom endpoint for proxy services; leave blank for official APIs

# Email (required)
EMAIL_FROM=your_email@example.com
EMAIL_PASSWORD=your_smtp_password   # Use app password / auth code, NOT your login password
EMAIL_TO=recipient@example.com      # Comma-separated for multiple recipients
```

<details>
<summary>Optional: GitHub / YouTube / Feishu / WeCom</summary>

```dotenv
# GitHub Token (without this, API rate limit is 60 req/hour)
GITHUB_TOKEN=ghp_xxxxx

# YouTube Data API v3 (skip YouTube collection if not set)
YOUTUBE_API_KEY=AIzaSy_xxxxx

# Feishu group bot
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxx

# WeCom group bot
WEWORK_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxx
```

</details>

> **Gmail / SMTP**: Use an app password generated from your account security settings, not your login password.

---

### Step 2: Configure schedules and sources

Edit `config/config.yaml`:

```yaml
schedules:
  - name: "Morning Digest"
    cron: "0 8 * * *"
    content: [schedule, todos, news]   # schedule + todos + news
    sources: [github, youtube, rss]
    focus: "AI agents, LLM engineering, and open-source ecosystem updates"
    subject_prefix: "Good Morning | SignalNest"

  - name: "Evening Digest"
    cron: "0 21 * * *"
    content: [news]
    sources: [github, youtube, rss]
    focus: "Today's tech and AI industry news, product launches, and research breakthroughs"
    subject_prefix: "Evening Picks | SignalNest"
```

`content` options:

| Value | Description |
|---|---|
| `news` | Collect sources + AI digest |
| `schedule` | Today's schedule from `personal/schedule.yaml` |
| `todos` | Due / overdue TODOs from `personal/todos.yaml` |

`focus` field: the AI uses this as the primary scoring signal for each schedule run. Leave blank to rely solely on learned preferences.

---

### Step 3: Set up personal assistant (optional)

<details>
<summary><code>config/personal/schedule.yaml</code> — Weekly schedule</summary>

```yaml
daily:
  - time: "07:30"
    title: "Morning workout"

weekly:
  mon:
    - time: "09:00"
      title: "Team meeting"
      location: "Room 201"
  tue:
    - time: "10:00"
      title: "1-on-1 with advisor"
```

</details>

<details>
<summary><code>config/personal/todos.yaml</code> — TODO list</summary>

```yaml
todos:
  - id: "r001"
    title: "Submit paper draft"
    due: "2026-03-10"
    priority: "high"    # high / medium / low
    done: false
```

TODOs are grouped automatically in the digest:

- ⚠ **Overdue** (due < today)
- ★ **Due today**
- ○ **Upcoming** (due within `lookahead_days` days)

</details>

---

### Step 4: Launch

```bash
cd SignalNest/docker/
docker compose up -d
```

Check logs:

```bash
docker logs -f signalnest
```

<br>

## 🐳 Docker Deployment

### Common commands

```bash
# Start (background, triggers automatically by cron)
docker compose up -d

# Rebuild after source code changes
docker compose up -d --build

# Restart only (after editing config.yaml or personal/)
docker compose restart

# Recreate container (after changing .env)
docker compose up -d --force-recreate

# Stop
docker compose down
```

### Trigger immediately for testing

In `docker/.env`:

```dotenv
IMMEDIATE_RUN=true            # Run once immediately on startup
SCHEDULE_NAME=Morning Digest  # Leave blank to use the first schedule
```

Then: `docker compose up -d --force-recreate`

### Data persistence

`data/` is mounted as a Docker volume to the host. `feedback.db` (preference history) survives container rebuilds.
Each run (morning/evening/weekly) is also archived to `data/history/` by run time, so historical outputs are preserved instead of overwritten.

<br>

## 🧠 Preference Learning

After each digest run, `data/last_digest.json` is generated automatically:

```json
{
  "date": "2026-03-02",
  "source": "github",
  "title": "vllm-project/vllm",
  "ai_score": 9,
  "ai_summary": "High-performance LLM inference engine...",
  "user_score": null,
  "user_notes": ""
}
```

Set `user_score` to an integer 1–5 for items you care about. **On the next run, scores are applied automatically** — the AI references your high-rated history when filtering new content.

> `data/` is volume-mounted to the host, so you can edit the file directly without entering the container.

<br>

## ⚙️ Configuration

### AI settings

```yaml
ai:
  model: "openai/gpt-5.2"       # LiteLLM format; env AI_MODEL takes priority
  api_base: ""                  # Custom endpoint; env AI_API_BASE takes priority
  min_relevance_score: 5        # Filter items below this score (1–10)
  max_items_per_digest: 20      # Max items shown per digest
  min_items_per_source:         # Optional per-source minimums
    github: 5
    youtube: 2
  max_tokens: 2048              # Max tokens per summary
```

`min_items_per_source` is enforced in both stages (title selection + final output).
If high-score items are insufficient, lower-score candidates from that source may be used as fallback. If collection itself returns too few items, the final count can still be lower than the target.

### GitHub collector

Scrapes `github.com/trending`; the AI filters by `focus` — no manual keyword lists needed.

```yaml
collectors:
  github:
    enabled: true
    trending_since: "daily"      # daily / weekly / monthly
    trending_languages: []       # Leave empty for all languages, or e.g. ["python", "typescript"]
    max_repos: 25                # Max repos to fetch
```

### YouTube collector

Two parallel tracks. Transcripts are fetched **after** title-based AI filtering, so only selected videos incur transcript requests.

```yaml
collectors:
  youtube:
    enabled: true                # Requires YOUTUBE_API_KEY
    # ── Track 1: Subscribed channels ─────────────────────────
    channel_ids:
      - "UCnUYZLuoy1rq1aVMwx4aTzw"   # Lex Fridman Podcast
      - "UCcefcZRL2oaA_uBNeo5UOWg"   # Y Combinator
    max_results_per_channel: 3   # Final videos kept per channel (based on sort_by)
    days_lookback: 7             # Only fetch videos from the last N days
    sort_by: "views"             # "views" (popularity) / "date" (newest first)
    # ── Track 2: AI keyword search (other channels) ───────────
    enable_keyword_search: true  # Adds one AI call + YouTube Search API quota
    search_sort_by: "views"      # Track 2 ordering: "views" (popularity) / "date" (newest first)
    max_search_results: 3        # Max videos per keyword
    search_days_lookback: 3      # Time window for keyword search (independent of channel window)
```

When `enable_keyword_search` is enabled, the AI derives 3–5 English search phrases from the current schedule's `focus` and queries the YouTube Search API to surface content beyond your subscribed channels.
In Track 2, both `search_sort_by: "views"` and `search_sort_by: "date"` first fetch `max_search_results * 3` candidates, then apply local selection and keep the top `max_search_results` items.

### RSS feeds

Two-phase fetch: each feed initially pulls `max_items_per_feed_initial` titles for batch AI filtering, then only the selected items proceed to deep-read scoring.

```yaml
collectors:
  rss:
    enabled: true
    days_lookback: 2
    max_items_per_feed_initial: 10  # Titles fetched per feed for batch filtering
    max_items_per_feed: 3           # Max articles per feed that reach deep-read scoring
    feeds:
      - id: "hacker-news"
        name: "Hacker News"
        url: "https://hnrss.org/frontpage"
      # Add more feeds here...
```

Changes to `config.yaml` take effect after `docker compose restart` — no rebuild needed.

### Notification channels

```yaml
notifications:
  email:  { enabled: true }
  feishu: { enabled: true }   # Also set FEISHU_WEBHOOK_URL in .env
  wework: { enabled: true }   # Also set WEWORK_WEBHOOK_URL in .env
```

> **Privacy**: `schedule` and `todos` are personal content — only sent to `EMAIL_FROM` (the sender). Other recipients receive only the news section.

## ❓ FAQ

### Q: Email sending fails with 535 authentication error

Use an app password / SMTP auth code, not your account login password.

### Q: GitHub collection is slow or hitting rate limits

Without `GITHUB_TOKEN`, you're limited to 60 API requests/hour. Generate a token at GitHub Settings → Developer Settings → Personal Access Tokens (no permissions needed).

### Q: YouTube returns 403 Forbidden

Enable YouTube Data API v3 in Google Cloud Console, and make sure the API key has no HTTP referrer restrictions.

### Q: Enabling `enable_keyword_search` increases my YouTube API quota usage

Keyword search makes one extra AI call (to generate keywords) plus several YouTube Search API requests. The YouTube Data API v3 free tier is 10,000 units/day; each Search call costs ~100 units while channel playlist fetches cost ~1 unit. Keep `max_search_results` low to stay within quota.

### Q: Why can final output still be below `github: 5` / `youtube: 2`?

Minimums can only be enforced when candidates exist. If a source returns too few items during collection (for example, no recent YouTube uploads within `days_lookback`, or temporary API failures), final counts may still be below target. Increase `days_lookback` / `search_days_lookback`, add more `channel_ids`, and check runtime logs for API errors.

### Q: How to add more RSS feeds

Edit `collectors.rss.feeds` in `config/config.yaml`, add `{id, name, url}`, then `docker compose restart`.

### Q: How to run only one schedule manually

Set `IMMEDIATE_RUN=true` and `SCHEDULE_NAME=Morning Digest` in `docker/.env`, then recreate the container.

## 📚 Credits

Inspired by [TrendRadar](https://github.com/sansan0/TrendRadar) and [obsidian-daily-digest](https://github.com/iamseeley/obsidian-daily-digest)

