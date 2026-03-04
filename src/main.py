"""
main.py - SignalNest agent-only orchestrator
==========================================
Invoked by Docker entrypoint / supercronic:
  python -m src.main --schedule-name "早间日报"
  python -m src.main --schedule-name "早间日报" --dry-run
"""

import argparse
import copy
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config_loader import load_config

logger = logging.getLogger("signalnest")

# 常见中文调度名称 -> 英文文件名别名
SCHEDULE_SLUG_MAP = {
    "早间日报": "morning_digest",
    "晚间日报": "evening_digest",
    "午间快讯": "midday_brief",
    "周末深度": "weekend_deep_dive",
}


def _resolve_schedule(schedule_name: str, config: dict) -> dict:
    """Resolve schedule by name, fallback to the first one."""
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
        "这是 dry-run，不要真实发送通知；可以调用 dispatch_notifications 走通流程。"
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
        "2) 调用 summarize_news 生成新闻摘要。\n"
        "3) 如果包含 schedule：调用 read_today_schedule。\n"
        "4) 如果包含 todos：调用 read_active_projects。\n"
        "5) 必须调用 build_digest_payload，并显式传入 schedule_name/subject_prefix/focus。\n"
        "6) 若非 dry-run，必须调用 dispatch_notifications。\n"
        "7) 最后返回清晰的最终结果。"
    )


def _render_session_title(template: str, schedule_name: str) -> str:
    try:
        rendered = template.format(schedule_name=schedule_name)
        return rendered.strip() or schedule_name
    except Exception as e:
        logger.warning(
            "agent.session_title_template 渲染失败（template=%r, schedule_name=%r）: %s",
            template,
            schedule_name,
            e,
        )
        return schedule_name


def run_schedule(schedule_name: str, config: dict, dry_run: bool = False) -> dict:
    """Run one scheduled task through the local agent kernel."""
    from src.agent.kernel import AgentRunOptions, run_agent_turn
    from src.agent.session_store import AgentSessionStore

    _apply_pending_feedback(config)

    schedule = _resolve_schedule(schedule_name, config)
    message = _build_agent_schedule_message(schedule, dry_run=dry_run)

    agent_cfg = config["agent"]
    schedule_max_steps = int(agent_cfg["schedule_max_steps"])
    schedule_allow_side_effects = bool(agent_cfg["schedule_allow_side_effects"])
    require_dispatch_tool_call = bool(agent_cfg["require_dispatch_tool_call"])
    session_title = _render_session_title(
        str(agent_cfg["session_title_template"]),
        str(schedule.get("name", "")),
    )

    run_config = copy.deepcopy(config)
    run_config["agent"]["policy"]["allow_side_effects"] = schedule_allow_side_effects

    result = run_agent_turn(
        message,
        run_config,
        options=AgentRunOptions(
            max_steps=schedule_max_steps,
            dry_run=dry_run,
            session_title=session_title,
        ),
    )

    status = str(result.get("status", ""))
    if status != "ok":
        raise RuntimeError(result.get("response", "agent schedule run failed"))

    if not dry_run and schedule_allow_side_effects and require_dispatch_tool_call:
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

    # 将本次新闻结果写入 last_digest 与 history 归档。
    try:
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


def _apply_pending_feedback(config: dict):
    """
    读取 data/last_digest.json，将用户已填写 user_score（1-5）的条目
    写入 feedback.db，然后将这些条目的 user_score 清空（避免重复写入）。
    """
    from src.ai.feedback import init_db, save_feedback

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
            r["user_score"] = None  # 清空，避免下次重复写入
            r["user_notes"] = ""
            applied += 1

    if applied:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"✨ 已将 {applied} 条用户反馈写入偏好数据库")


def _slugify_schedule_name(name: str) -> str:
    """将调度名转为英文/数字/下划线文件名片段。"""
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
    data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "last_digest.json"

    records = []
    for item in news_items:
        records.append(
            {
                "date": str(today),
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "ai_score": item.get("ai_score"),
                "ai_summary": item.get("ai_summary", ""),
                "user_score": None,
                "user_notes": "",
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"📋 已保存 {len(records)} 条内容到 {out_path}（填写 user_score 后下次运行自动学习偏好）")

    history_dir = data_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    schedule_slug = _slugify_schedule_name(schedule_name)
    timestamp = run_dt.strftime("%Y%m%d_%H%M%S_%f")
    history_path = history_dir / f"digest_{timestamp}_{schedule_slug}.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"🗂️ 已归档本次结果到 {history_path}")


def main():
    parser = argparse.ArgumentParser(description="SignalNest - Agent-only 个人 AI 日报服务")
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    result = run_schedule(
        schedule_name=args.schedule_name,
        config=config,
        dry_run=args.dry_run,
    )

    print(f"[schedule] {args.schedule_name or '(default)'}")
    print(f"[agent session] {result['session_id']} | turn #{result['turn_index']}")
    print(result.get("response", ""))


if __name__ == "__main__":
    main()
