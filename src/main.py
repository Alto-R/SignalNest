"""
main.py - SignalNest 主编排器
===================================
被 Docker entrypoint / supercronic 调用:
  python -m src.main --schedule-name "早间日报"
  python -m src.main --schedule-name "早间日报" --dry-run

执行流程:
  1. 加载配置 (config.yaml + .env)
  2. 找到匹配的 schedule entry
  3. 按 content 列表决定运行哪些模块
  4. 采集 → AI摘要 → 组装payload → 分发通知
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.config_loader import load_config

logger = logging.getLogger("signalnest")

WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 常见中文调度名称 -> 英文文件名别名
SCHEDULE_SLUG_MAP = {
    "早间日报": "morning_digest",
    "晚间日报": "evening_digest",
    "午间快讯": "midday_brief",
    "周末深度": "weekend_deep_dive",
}


def _split_csv(value: str) -> list[str] | None:
    raw = (value or "").strip()
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",")]
    normalized = [x for x in items if x]
    return normalized or None


def _resolve_schedule(schedule_name: str, config: dict) -> dict:
    """
    Resolve schedule by name, fallback to the first one.
    """
    schedule = next(
        (s for s in config.get("schedules", []) if s.get("name") == schedule_name),
        None,
    )
    if schedule is not None:
        return schedule

    schedules = config.get("schedules", [])
    if schedules:
        schedule = schedules[0]
        logger.warning(f"Schedule '{schedule_name}' 未找到，使用第一个: '{schedule['name']}'")
        return schedule

    logger.error("config.yaml 中没有定义任何 schedules")
    sys.exit(1)


def _build_agent_schedule_message(schedule: dict, *, dry_run: bool) -> str:
    content_blocks = schedule.get("content", ["news"])
    sources = schedule.get("sources", ["github", "youtube", "rss"])
    focus = schedule.get("focus", "")
    schedule_name = schedule.get("name", "")
    subject_prefix = schedule.get("subject_prefix", "SignalNest")

    dry_run_note = (
        "这是 dry-run，不要真实发送通知；可调用 dispatch_notifications 走通流程。"
        if dry_run
        else "这是正式定时任务，必须完成通知发送（dispatch_notifications）。"
    )

    return (
        "你正在执行 SignalNest 的定时调度任务，请按配置完成本次日报。\n"
        f"- schedule_name: {schedule_name}\n"
        f"- content_blocks: {content_blocks}\n"
        f"- sources: {sources}\n"
        f"- focus: {focus}\n"
        f"- subject_prefix: {subject_prefix}\n"
        f"- {dry_run_note}\n\n"
        "执行要求：\n"
        "1) 如果 content_blocks 包含 news：调用 collect_all_news 与 summarize_news。\n"
        "2) 如果包含 schedule：调用 read_today_schedule。\n"
        "3) 如果包含 todos：调用 read_active_projects。\n"
        "4) 调用 build_digest_payload 组装推送载荷（传入 schedule_name/subject_prefix/focus）。\n"
        "5) 若非 dry-run，必须调用 dispatch_notifications。\n"
        "6) 最后返回清晰的最终结果。"
    )


def run_agent_for_schedule(schedule_name: str, config: dict, dry_run: bool = False) -> dict:
    """
    Run one scheduled task through the local agent kernel.
    """
    from src.agent.kernel import AgentRunOptions, run_agent_turn

    # 与 legacy run() 对齐：先应用用户在 last_digest.json 中填写的反馈分数。
    _apply_pending_feedback(config)

    schedule = _resolve_schedule(schedule_name, config)
    message = _build_agent_schedule_message(schedule, dry_run=dry_run)

    agent_cfg = config.get("agent", {})
    schedule_max_steps = agent_cfg.get("schedule_max_steps", agent_cfg.get("max_steps"))
    schedule_allow_side_effects = bool(agent_cfg.get("schedule_allow_side_effects", True))
    result = run_agent_turn(
        message,
        config,
        options=AgentRunOptions(
            max_steps=schedule_max_steps,
            dry_run=dry_run,
            allow_side_effects=schedule_allow_side_effects,
        ),
    )

    status = str(result.get("status", ""))
    if status != "ok":
        raise RuntimeError(result.get("response", "agent schedule run failed"))

    if not dry_run and schedule_allow_side_effects:
        steps = result.get("steps", [])
        dispatched = any(
            isinstance(step, dict)
            and step.get("tool") == "dispatch_notifications"
            and "error" not in step
            for step in steps
        )
        if not dispatched:
            raise RuntimeError("agent run finished without dispatch_notifications")
    elif not dry_run and not schedule_allow_side_effects:
        logger.warning("agent.schedule_allow_side_effects=false，已跳过通知发送校验")

    # 与 legacy run() 保持一致：将本次新闻结果写入 last_digest 与 history 归档。
    try:
        from src.agent.session_store import AgentSessionStore

        data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
        session_store = AgentSessionStore(data_dir / "agent_sessions.db")
        state = session_store.load_state(result["session_id"])
        news_items = state.get("news_items", [])

        if isinstance(news_items, list) and news_items:
            tz = ZoneInfo(config.get("app", {}).get("timezone", "Asia/Shanghai"))
            now = datetime.now(tz)
            _save_last_digest(
                news_items=news_items,
                today=now.date(),
                run_dt=now,
                schedule_name=schedule.get("name", ""),
                config=config,
            )
    except Exception as e:
        logger.warning(f"agent 调度归档到 history 失败: {e}")

    return result


def run(schedule_name: str, config: dict, dry_run: bool = False):
    """
    执行一次完整的日报生成和推送流程。

    Args:
        schedule_name: 匹配 config["schedules"] 中的 name 字段
        config:        AppConfig dict
        dry_run:       True 时打印预览，不发送通知
    """
    # ── 读取上次 JSON 中用户填写的分数，写入 feedback.db ─────
    _apply_pending_feedback(config)

    # ── 找到匹配的 schedule entry ─────────────────────────────
    schedule = _resolve_schedule(schedule_name, config)

    content_blocks = schedule.get("content", ["news"])
    sources = schedule.get("sources", ["github", "youtube", "rss"])
    subject_prefix = schedule.get("subject_prefix", "SignalNest")
    focus = schedule.get("focus", "")
    tz = ZoneInfo(config.get("app", {}).get("timezone", "Asia/Shanghai"))
    now = datetime.now(tz)
    today = now.date()

    logger.info(
        f"▶ Schedule: '{schedule['name']}' | content={content_blocks} | sources={sources} | focus={focus!r}"
    )

    # 每次 schedule 推送都记录为持久会话（用于 Docker 场景回溯与排障）。
    session_store = None
    turn_ref = None
    session_id = ""
    session_status = "ok"
    session_reply = ""
    from src.agent.session_store import AgentSessionStore

    data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
    session_store = AgentSessionStore(data_dir / "agent_sessions.db")
    session_id = (
        f"schedule-{_slugify_schedule_name(schedule.get('name', ''))}-"
        f"{now.strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:8]}"
    )
    session_store.ensure_session(session_id, title=f"Scheduled Push | {schedule['name']}")
    backend = os.environ.get("AI_BACKEND") or config.get("ai", {}).get("backend", "litellm")
    model = os.environ.get("AI_MODEL") or config.get("ai", {}).get("model", "")
    turn_ref = session_store.start_turn(
        session_id,
        (
            f"Auto schedule push: name={schedule['name']} "
            f"date={today} dry_run={str(dry_run).lower()}"
        ),
        backend=backend,
        model=model,
    )
    logger.info(f"🧠 会话持久化已启用: session_id={session_id}")

    schedule_entries = None
    projects = None
    raw_items: list[dict] = []
    news_items: list[dict] = []
    digest_summary = ""
    payload: dict | None = None

    try:
        # ── Section 1: 个人助手 ───────────────────────────────────
        if "schedule" in content_blocks:
            from src.personal.ai_reader import read_today_schedule

            personal_dir = config.get("_personal_dir", "/app/config/personal")
            schedule_path = str(Path(personal_dir) / "schedule.md")
            schedule_entries = read_today_schedule(schedule_path, today, config)
            logger.info(f"  日程: {len(schedule_entries)} 条")

        if "todos" in content_blocks:
            from src.personal.ai_reader import read_active_projects

            personal_dir = config.get("_personal_dir", "/app/config/personal")
            projects_path = str(Path(personal_dir) / "projects.md")
            lookahead = config.get("storage", {}).get("todo_lookahead_days", 7)
            projects = read_active_projects(projects_path, today, config, lookahead)
            logger.info(f"  项目: {len(projects)} 个活跃项目")

        # ── Section 2: 新闻采集 + AI 摘要 ──────────────────────────
        if "news" in content_blocks:
            collectors_cfg = config.get("collectors", {})

            if "github" in sources and collectors_cfg.get("github", {}).get("enabled", True):
                logger.info("📦 抓取 GitHub...")
                try:
                    from src.collectors.github_collector import collect_github

                    items = collect_github(config)
                    raw_items.extend(items)
                    logger.info(f"   GitHub: {len(items)} 个仓库")
                except Exception as e:
                    logger.error(f"   GitHub 失败: {e}")

            if "youtube" in sources and collectors_cfg.get("youtube", {}).get("enabled", False):
                logger.info("📺 抓取 YouTube...")
                try:
                    from src.collectors.youtube_collector import collect_youtube

                    items = collect_youtube(config, focus=focus)
                    raw_items.extend(items)
                    logger.info(f"   YouTube: {len(items)} 个视频")
                except Exception as e:
                    logger.error(f"   YouTube 失败: {e}")

            if "rss" in sources and collectors_cfg.get("rss", {}).get("enabled", True):
                logger.info("📰 抓取 RSS...")
                try:
                    from src.collectors.rss_collector import collect_rss

                    items = collect_rss(config)
                    raw_items.extend(items)
                    logger.info(f"   RSS: {len(items)} 篇文章")
                except Exception as e:
                    logger.error(f"   RSS 失败: {e}")

            logger.info(f"采集完成，共 {len(raw_items)} 条原始内容")

            if raw_items:
                logger.info("🤖 AI 摘要中...")
                from src.ai.summarizer import summarize_items

                news_items = summarize_items(raw_items, config, focus=focus)
                logger.info(f"   筛选后: {len(news_items)} 条")

        # ── Section 2b: 生成今日要点总结 ─────────────────────────
        if news_items:
            logger.info("🤖 生成今日要点...")
            from src.ai.summarizer import generate_digest_summary

            digest_summary = generate_digest_summary(news_items, config, focus=focus)

        # ── Section 3: 组装 Payload ──────────────────────────────
        payload = {
            "schedule_name": schedule["name"],
            "subject_prefix": subject_prefix,
            "focus": focus,
            "date": today,
            "datetime": now,
            "schedule_entries": schedule_entries,
            "projects": projects,
            "news_items": news_items,
            "digest_summary": digest_summary,
            "content_blocks": content_blocks,
        }

        # ── Section 4: 分发通知 ──────────────────────────────────
        if dry_run:
            _print_dry_run(payload)
        else:
            from src.notifications.dispatcher import dispatch

            dispatch(payload, config)

        # ── Section 5: 保存新闻条目供反馈打分使用 ────────────────
        if news_items:
            _save_last_digest(
                news_items=news_items,
                today=today,
                run_dt=now,
                schedule_name=schedule.get("name", ""),
                config=config,
            )

        session_reply = (
            f"Schedule push completed: schedule={schedule['name']} "
            f"news_items={len(news_items)} dry_run={str(dry_run).lower()}"
        )
        logger.info("✅ 完成")
    except Exception as e:
        session_status = "error"
        session_reply = f"Schedule push failed: {e}"
        raise
    finally:
        if session_store and turn_ref:
            state_snapshot = {
                "schedule_name": schedule.get("name", ""),
                "focus": focus,
                "date": today,
                "datetime": now,
                "content_blocks": content_blocks,
                "sources": sources,
                "dry_run": dry_run,
                "schedule_entries": schedule_entries,
                "projects": projects,
                "raw_items": raw_items,
                "news_items": news_items,
                "digest_summary": digest_summary,
                "payload": payload,
            }
            session_store.save_state(session_id, state_snapshot)
            session_store.finish_turn(turn_ref.turn_id, session_reply, session_status)


def _apply_pending_feedback(config: dict):
    """
    读取 data/last_digest.json，将用户已填写 user_score（1-5）的条目
    写入 feedback.db，然后将这些条目的 user_score 清空（避免重复写入）。
    """
    import json
    from pathlib import Path
    from src.ai.feedback import save_feedback, init_db

    data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
    path = data_dir / "last_digest.json"
    if not path.exists():
        return

    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.warning(f"读取 last_digest.json 失败: {e}")
        return

    init_db(config)
    applied = 0
    for r in records:
        score = r.get("user_score")
        if score is not None and isinstance(score, int) and 1 <= score <= 5:
            save_feedback(
                config,
                date_str=r.get("date", ""),
                source=r.get("source", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                score=score,
                ai_summary=r.get("ai_summary", ""),
                notes=r.get("user_notes", ""),
            )
            r["user_score"] = None   # 清空，避免下次重复写入
            r["user_notes"] = ""
            applied += 1

    if applied:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"✨ 已将 {applied} 条用户反馈写入偏好数据库")


def _slugify_schedule_name(name: str) -> str:
    """
    将调度名转为英文/数字/下划线文件名片段。
    """
    raw = (name or "").strip()
    if not raw:
        return "schedule"
    if raw in SCHEDULE_SLUG_MAP:
        return SCHEDULE_SLUG_MAP[raw]

    ascii_text = raw.encode("ascii", errors="ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return slug or "schedule"


def _save_last_digest(
    news_items: list[dict],
    today: date,
    run_dt: datetime,
    schedule_name: str,
    config: dict,
):
    """
    将本次新闻条目保存到 data/last_digest.json。
    同时归档一份到 data/history/*.json（英文文件名）。
    每条记录预留 user_score / user_notes 字段（默认 null / ""），
    用户可直接编辑此文件填写分数，下次运行时自动写入偏好数据库。
    """
    import json
    from pathlib import Path

    data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "last_digest.json"

    records = []
    for item in news_items:
        records.append({
            "date":       str(today),
            "source":     item.get("source", ""),
            "title":      item.get("title", ""),
            "url":        item.get("url", ""),
            "ai_score":   item.get("ai_score"),
            "ai_summary": item.get("ai_summary", ""),
            # ── 在此填写你的评分后，下次运行时自动生效 ──────────
            "user_score": None,   # 填 1-5 整数，null 表示跳过
            "user_notes": "",     # 备注（可留空）
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"📋 已保存 {len(records)} 条内容到 {out_path}（填写 user_score 后下次运行自动学习偏好）")

    # 归档：每次运行保存一份历史快照（英文文件名）
    history_dir = data_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    schedule_slug = _slugify_schedule_name(schedule_name)
    timestamp = run_dt.strftime("%Y%m%d_%H%M%S_%f")
    history_path = history_dir / f"digest_{timestamp}_{schedule_slug}.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"🗂️ 已归档本次结果到 {history_path}")


def _print_dry_run(payload: dict):
    print(f"\n{'='*60}")
    print(f"DRY RUN: {payload['subject_prefix']} | {payload['date']}")
    print(f"{'='*60}")

    entries = payload.get("schedule_entries") or []
    if entries:
        print(f"\n--- 今日日程 ({len(entries)} 条) ---")
        for e in entries:
            loc = f" @ {e['location']}" if e.get("location") else ""
            print(f"  {e['time']}  {e['title']}{loc}")

    todos = payload.get("projects") or []
    if todos:
        print(f"\n--- 项目进展 ({len(todos)} 个) ---")
        for proj in todos:
            due = f" (软截止 {proj['soft_due']})" if proj.get("soft_due") else ""
            print(f"  ▶ {proj['title']}{due}")
            for t in proj.get("tasks", []):
                task_due = f" [{t['soft_due']}]" if t.get("soft_due") else ""
                print(f"    · {t['title']}{task_due}")

    news = payload.get("news_items") or []
    if news:
        print(f"\n--- 新闻精选 ({len(news)} 条) ---")
        for item in news[:5]:
            print(f"  [{item['ai_score']}/10][{item['source']}] {item['title'][:60]}")
        if len(news) > 5:
            print(f"  ... 还有 {len(news)-5} 条")


def main():
    parser = argparse.ArgumentParser(description="SignalNest - 个人 AI 日报服务")
    parser.add_argument(
        "--schedule-name",
        default="",
        help="要执行的调度名称（匹配 config.schedules[].name，空则用第一个）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="打印预览，不发送通知",
    )
    parser.add_argument(
        "--agent-message",
        default="",
        help="运行本地 agent（无 gateway），输入用户消息文本",
    )
    parser.add_argument(
        "--agent-schedule-name",
        default=None,
        help="按 schedule 配置自动生成任务并运行本地 agent（用于 cron）",
    )
    parser.add_argument(
        "--agent-session-id",
        default="",
        help="本地 agent 会话 ID（不填则自动新建）",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=None,
        help="本地 agent 最大工具步数（默认读 config.agent.max_steps，缺省 6）",
    )
    parser.add_argument(
        "--agent-allow-side-effects",
        action="store_true",
        default=None,
        help="允许本地 agent 执行副作用工具（如发送通知）",
    )
    parser.add_argument(
        "--agent-allow-tools",
        default="",
        help="本地 agent 工具白名单，逗号分隔；为空表示不设白名单",
    )
    parser.add_argument(
        "--agent-deny-tools",
        default="",
        help="本地 agent 工具黑名单，逗号分隔",
    )
    parser.add_argument(
        "--agent-json",
        action="store_true",
        help="本地 agent 结果按 JSON 输出",
    )
    parser.add_argument(
        "--agent-tools-schema",
        action="store_true",
        help="输出本地 agent tools schema（JSON）并退出",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.agent_tools_schema:
        from src.agent.tools import build_agent_tools, export_tools_schema

        schema = export_tools_schema(build_agent_tools())
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        return

    config = load_config()

    if args.agent_schedule_name is not None:
        result = run_agent_for_schedule(
            schedule_name=args.agent_schedule_name or "",
            config=config,
            dry_run=args.dry_run,
        )
        if args.agent_json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"[agent schedule] {args.agent_schedule_name or '(default)'}")
            print(f"[agent session] {result['session_id']} | turn #{result['turn_index']}")
            print(result.get("response", ""))
        return

    if args.agent_message:
        from src.agent.kernel import AgentRunOptions, run_agent_turn

        result = run_agent_turn(
            args.agent_message,
            config,
            options=AgentRunOptions(
                session_id=args.agent_session_id or None,
                max_steps=args.agent_max_steps,
                dry_run=args.dry_run,
                allow_side_effects=args.agent_allow_side_effects,
                allow_tools=_split_csv(args.agent_allow_tools),
                deny_tools=_split_csv(args.agent_deny_tools),
            ),
        )
        if args.agent_json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"[agent session] {result['session_id']} | turn #{result['turn_index']}")
            print(result.get("response", ""))
        return

    run(
        schedule_name=args.schedule_name,
        config=config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

