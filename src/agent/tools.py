"""
Tool registry for SignalNest local agent.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.ai.summarizer import generate_digest_summary, summarize_items
from src.collectors.github_collector import collect_github
from src.collectors.rss_collector import collect_rss
from src.collectors.youtube_collector import collect_youtube
from src.notifications.dispatcher import dispatch
from src.personal.ai_reader import read_active_projects, read_today_schedule

ToolHandler = Callable[[dict[str, Any], "ToolRuntime"], dict[str, Any]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: bool
    handler: ToolHandler


@dataclass
class ToolRuntime:
    config: dict
    state: dict[str, Any]
    dry_run: bool
    now: datetime

    @property
    def today(self) -> date:
        return self.now.date()

    def tz(self) -> ZoneInfo:
        tz_name = self.config.get("app", {}).get("timezone", "Asia/Shanghai")
        return ZoneInfo(tz_name)


def _item_key(item: dict[str, Any]) -> str:
    source = str(item.get("source", "")).strip().lower()
    url = str(item.get("url", "")).strip().lower()
    title = str(item.get("title", "")).strip().lower()
    if url:
        return f"{source}:{url}"
    return f"{source}:{title}"


def _merge_items(existing: list[dict], added: list[dict]) -> list[dict]:
    merged = list(existing)
    seen = {_item_key(i) for i in merged}
    for item in added:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _compact_news_preview(items: list[dict], limit: int = 5) -> list[str]:
    return [str(item.get("title", "")) for item in items[:limit]]


def _tool_collect_github(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    cfg = copy.deepcopy(rt.config)
    gh_cfg = cfg.setdefault("collectors", {}).setdefault("github", {})
    if "since" in args:
        gh_cfg["trending_since"] = args["since"]
    if "languages" in args:
        gh_cfg["trending_languages"] = args["languages"]

    items = collect_github(cfg, max_repos=args.get("max_repos"))
    if args.get("replace_state", False):
        rt.state["raw_items"] = items
    else:
        rt.state["raw_items"] = _merge_items(rt.state.get("raw_items", []), items)

    return {
        "source": "github",
        "fetched_count": len(items),
        "raw_items_total": len(rt.state["raw_items"]),
        "sample_titles": _compact_news_preview(items),
    }


def _tool_collect_rss(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    cfg = copy.deepcopy(rt.config)
    rss_cfg = cfg.setdefault("collectors", {}).setdefault("rss", {})
    if "days_back" in args:
        rss_cfg["days_lookback"] = args["days_back"]

    items = collect_rss(cfg, max_total=args.get("max_total"))
    if args.get("replace_state", False):
        rt.state["raw_items"] = items
    else:
        rt.state["raw_items"] = _merge_items(rt.state.get("raw_items", []), items)

    return {
        "source": "rss",
        "fetched_count": len(items),
        "raw_items_total": len(rt.state["raw_items"]),
        "sample_titles": _compact_news_preview(items),
    }


def _tool_collect_youtube(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    items = collect_youtube(
        rt.config,
        focus=args.get("focus", ""),
        max_total=args.get("max_total"),
    )
    if args.get("replace_state", False):
        rt.state["raw_items"] = items
    else:
        rt.state["raw_items"] = _merge_items(rt.state.get("raw_items", []), items)

    return {
        "source": "youtube",
        "fetched_count": len(items),
        "raw_items_total": len(rt.state["raw_items"]),
        "sample_titles": _compact_news_preview(items),
    }


def _tool_collect_all_news(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    sources = args.get("sources", ["github", "youtube", "rss"])
    collected: list[dict] = []
    per_source: dict[str, int] = {}

    if "github" in sources:
        gh_items = collect_github(rt.config, max_repos=args.get("github_max_repos"))
        collected.extend(gh_items)
        per_source["github"] = len(gh_items)

    if "youtube" in sources:
        yt_items = collect_youtube(
            rt.config,
            focus=args.get("focus", ""),
            max_total=args.get("youtube_max_total"),
        )
        collected.extend(yt_items)
        per_source["youtube"] = len(yt_items)

    if "rss" in sources:
        rss_items = collect_rss(rt.config, max_total=args.get("rss_max_total"))
        collected.extend(rss_items)
        per_source["rss"] = len(rss_items)

    deduped = _merge_items([], collected)
    if args.get("replace_state", True):
        rt.state["raw_items"] = deduped
    else:
        rt.state["raw_items"] = _merge_items(rt.state.get("raw_items", []), deduped)

    return {
        "sources": sources,
        "per_source": per_source,
        "fetched_count": len(deduped),
        "raw_items_total": len(rt.state["raw_items"]),
        "sample_titles": _compact_news_preview(deduped),
    }


def _tool_summarize_news(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    raw_items = rt.state.get("raw_items", [])
    if not raw_items:
        raise ValueError("state.raw_items is empty; run a collect tool first")

    ai_cfg = rt.config.get("ai", {})
    raw_cap = ai_cfg.get("max_items_per_digest", 15)
    try:
        config_cap = int(raw_cap)
        if config_cap <= 0:
            raise ValueError("non-positive cap")
    except Exception:
        config_cap = 15
        logger.warning("summarize_news: invalid ai.max_items_per_digest=%r, fallback=15", raw_cap)

    focus = args.get("focus", "")

    news_items = summarize_items(
        raw_items,
        rt.config,
        min_score=args.get("min_score"),
        max_output=None,
        focus=focus,
    )
    if len(news_items) > config_cap:
        logger.warning(
            "summarize_news: defensive clamp triggered %s -> %s by config cap",
            len(news_items),
            config_cap,
        )
        news_items = news_items[:config_cap]

    digest_summary = generate_digest_summary(news_items, rt.config, focus=focus)

    rt.state["news_items"] = news_items
    rt.state["digest_summary"] = digest_summary

    return {
        "news_count": len(news_items),
        "top_titles": _compact_news_preview(news_items),
        "digest_summary": digest_summary,
    }


def _tool_read_today_schedule(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    personal_dir = Path(rt.config.get("_personal_dir", ""))
    schedule_path = personal_dir / "schedule.md"
    entries = read_today_schedule(str(schedule_path), rt.today, rt.config)
    rt.state["schedule_entries"] = entries

    return {
        "date": str(rt.today),
        "entry_count": len(entries),
        "sample": entries[:5],
    }


def _tool_read_active_projects(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    personal_dir = Path(rt.config.get("_personal_dir", ""))
    projects_path = personal_dir / "projects.md"
    lookahead_days = args.get(
        "lookahead_days",
        rt.config.get("storage", {}).get("todo_lookahead_days", 3),
    )
    projects = read_active_projects(
        str(projects_path),
        rt.today,
        rt.config,
        lookahead_days=lookahead_days,
    )
    rt.state["projects"] = projects

    return {
        "date": str(rt.today),
        "project_count": len(projects),
        "sample_project_titles": [p.get("title", "") for p in projects[:5]],
    }


def _tool_build_digest_payload(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    now = datetime.now(rt.tz())
    today = now.date()

    schedule_entries = rt.state.get("schedule_entries")
    projects = rt.state.get("projects")
    news_items = rt.state.get("news_items", [])
    digest_summary = rt.state.get("digest_summary", "")

    content_blocks: list[str] = []
    if schedule_entries is not None:
        content_blocks.append("schedule")
    if projects is not None:
        content_blocks.append("todos")
    if news_items:
        content_blocks.append("news")

    payload = {
        "schedule_name": args["schedule_name"],
        "subject_prefix": args["subject_prefix"],
        "focus": args.get("focus", ""),
        "date": today,
        "datetime": now,
        "schedule_entries": schedule_entries,
        "projects": projects,
        "news_items": news_items,
        "digest_summary": digest_summary,
        "content_blocks": content_blocks,
    }
    rt.state["payload"] = payload

    return {
        "schedule_name": payload["schedule_name"],
        "subject_prefix": payload["subject_prefix"],
        "content_blocks": content_blocks,
        "schedule_entries_count": len(schedule_entries or []),
        "projects_count": len(projects or []),
        "news_items_count": len(news_items or []),
    }


def _tool_dispatch_notifications(args: dict[str, Any], rt: ToolRuntime) -> dict[str, Any]:
    payload = rt.state.get("payload")
    if not payload:
        raise ValueError("state.payload is empty; run build_digest_payload first")

    # State restored from SQLite stores date/datetime as strings; normalize for senders.
    payload = dict(payload)
    if isinstance(payload.get("date"), str):
        payload["date"] = date.fromisoformat(payload["date"])
    if isinstance(payload.get("datetime"), str):
        payload["datetime"] = datetime.fromisoformat(payload["datetime"])

    if rt.dry_run:
        return {
            "dry_run": True,
            "dispatched": False,
            "reason": "dry_run=true, skipped real notification dispatch",
        }

    dispatch_result = dispatch(payload, rt.config)
    return {
        "dry_run": False,
        "dispatched": True,
        "dispatch_result": dispatch_result,
    }


def build_agent_tools() -> dict[str, ToolSpec]:
    tools: list[ToolSpec] = [
        ToolSpec(
            name="collect_github",
            description="Collect GitHub Trending repositories.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "max_repos": {"type": "integer", "minimum": 1, "maximum": 100},
                    "since": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                    },
                    "replace_state": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_collect_github,
        ),
        ToolSpec(
            name="collect_rss",
            description="Collect items from configured RSS feeds.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "max_total": {"type": "integer", "minimum": 1, "maximum": 500},
                    "days_back": {"type": "integer", "minimum": 1, "maximum": 30},
                    "replace_state": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_collect_rss,
        ),
        ToolSpec(
            name="collect_youtube",
            description="Collect YouTube videos from subscribed channels and keyword search.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "focus": {"type": "string"},
                    "max_total": {"type": "integer", "minimum": 1, "maximum": 200},
                    "replace_state": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_collect_youtube,
        ),
        ToolSpec(
            name="collect_all_news",
            description="Collect multi-source news in one call and save into state.raw_items.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["github", "youtube", "rss"]},
                        "minItems": 1,
                        "maxItems": 3,
                        "default": ["github", "youtube", "rss"],
                    },
                    "focus": {"type": "string", "default": ""},
                    "github_max_repos": {"type": "integer", "minimum": 1, "maximum": 100},
                    "youtube_max_total": {"type": "integer", "minimum": 1, "maximum": 200},
                    "rss_max_total": {"type": "integer", "minimum": 1, "maximum": 500},
                    "replace_state": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            handler=_tool_collect_all_news,
        ),
        ToolSpec(
            name="summarize_news",
            description="Run two-stage AI summarization on state.raw_items.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "default": ""},
                    "min_score": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "additionalProperties": False,
            },
            handler=_tool_summarize_news,
        ),
        ToolSpec(
            name="read_today_schedule",
            description="Parse today's schedule from config/personal/schedule.md.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_tool_read_today_schedule,
        ),
        ToolSpec(
            name="read_active_projects",
            description="Parse active projects/todos from config/personal/projects.md.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "lookahead_days": {"type": "integer", "minimum": 1, "maximum": 30},
                },
                "additionalProperties": False,
            },
            handler=_tool_read_active_projects,
        ),
        ToolSpec(
            name="build_digest_payload",
            description="Build notification payload from current session state.",
            side_effect=False,
            input_schema={
                "type": "object",
                "properties": {
                    "schedule_name": {"type": "string"},
                    "subject_prefix": {"type": "string"},
                    "focus": {"type": "string", "default": ""},
                },
                "required": ["schedule_name", "subject_prefix"],
                "additionalProperties": False,
            },
            handler=_tool_build_digest_payload,
        ),
        ToolSpec(
            name="dispatch_notifications",
            description="Dispatch payload to enabled channels (email/feishu/wework).",
            side_effect=True,
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_tool_dispatch_notifications,
        ),
    ]
    return {t.name: t for t in tools}
