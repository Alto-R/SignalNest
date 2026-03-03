"""
feedback.py - SQLite 偏好反馈读写
====================================
负责：
  - 初始化 feedback.db
  - 保存用户反馈（score 1-5）
  - 读取高分历史作为 AI few-shot 示例
  - 读取近期历史标题用于去重
"""

import json
import re
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def get_db_path(config: dict) -> Path:
    data_dir = config.get("storage", {}).get("data_dir", "/app/data")
    return Path(data_dir) / "feedback.db"


def init_db(config: dict):
    db_path = get_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            source      TEXT NOT NULL,
            title       TEXT NOT NULL,
            url         TEXT NOT NULL,
            score       INTEGER,
            notes       TEXT,
            ai_summary  TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def load_taste_examples(config: dict, limit: int = 8) -> list[dict]:
    """读取高分历史（score >= 4），作为 AI few-shot 示例。"""
    init_db(config)
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT title, source, ai_summary, score, notes
        FROM feedback
        WHERE score >= 4
        ORDER BY score DESC, created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "title": r[0],
            "source": r[1],
            "summary": r[2] or "",
            "score": r[3],
            "notes": r[4] or "",
        }
        for r in rows
    ]


def load_recent_titles(config: dict, days: int = 30) -> list[str]:
    """读取最近 N 天历史摘要中出现过的标题，用于 Stage 1 去重提示。"""
    data_dir = Path(config.get("storage", {}).get("data_dir", "/app/data"))
    history_dir = data_dir / "history"
    if not history_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days)
    titles: list[str] = []

    for f in sorted(history_dir.glob("digest_*.json")):
        m = re.match(r"digest_(\d{8})_", f.name)
        if not m:
            continue
        try:
            file_date = datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
            for item in items:
                title = item.get("title", "").strip()
                if title:
                    titles.append(title)
        except Exception:
            continue

    # 去重并保留顺序
    seen: set[str] = set()
    unique: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def save_feedback(config: dict, date_str: str, source: str, title: str, url: str,
                  score: int, ai_summary: str = "", notes: str = ""):
    """保存一条用户反馈。"""
    init_db(config)
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO feedback (date, source, title, url, score, notes, ai_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (date_str, source, title, url, score, notes, ai_summary),
    )
    conn.commit()
    conn.close()
