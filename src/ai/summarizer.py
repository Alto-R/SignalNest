"""
summarizer.py - AI 摘要与品味过滤引擎
==========================================
主要改动：
  - 用传入的 config dict 替换 import config 模块
  - 使用 LiteLLM 替代 Anthropic SDK，支持任意 OpenAI 兼容接口
  - API 配置从 AI_API_KEY / AI_MODEL / AI_API_BASE 环境变量读取
  - 两阶段处理：第一阶段按标题批量筛选（1次调用），第二阶段仅对入选条目做完整摘要
"""

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    import litellm
    litellm.suppress_debug_info = True
except ImportError:
    litellm = None  # type: ignore

from src.ai.cli_backend import _call_ai
from src.ai.feedback import load_recent_history_records, load_taste_examples

logger = logging.getLogger(__name__)

_TRACKING_QUERY_KEYS = {
    "spm",
    "from",
    "ref",
    "source",
    "fbclid",
    "gclid",
    "si",
    "feature",
    "mc_cid",
    "mc_eid",
}


def _safe_positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_title(title: str) -> str:
    text = str(title or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    # 仅去除两端噪声符号，保留中间内容用于严格匹配
    text = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", text)
    return text


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""

    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw.lower()

    if not parsed.scheme and not parsed.netloc:
        return raw.lower()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")

    cleaned_query: list[tuple[str, str]] = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        key = str(k).strip()
        key_lower = key.lower()
        if key_lower.startswith("utm_") or key_lower in _TRACKING_QUERY_KEYS:
            continue
        cleaned_query.append((key, str(v).strip()))
    cleaned_query.sort(key=lambda x: (x[0].lower(), x[1]))
    query = urlencode(cleaned_query, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _is_strict_title_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    # 短标题要求完全一致，避免误判
    if min(len(a), len(b)) < 20:
        return False
    return _title_similarity(a, b) >= 0.97


def _parse_published_ts(value: str) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return 0.0


def _item_completeness_score(item: dict) -> int:
    score = 0
    if _normalize_url(str(item.get("url", ""))):
        score += 3
    if str(item.get("title", "")).strip():
        score += 2
    if str(item.get("published_at", "")).strip():
        score += 2
    if str(item.get("description", "")).strip() or str(item.get("content_snippet", "")).strip():
        score += 1
    if str(item.get("feed_title", "")).strip() or str(item.get("channel", "")).strip():
        score += 1
    return score


def _pick_better_item_index(items: list[dict], idx_a: int, idx_b: int) -> int:
    a = items[idx_a]
    b = items[idx_b]

    key_a = (
        _item_completeness_score(a),
        _parse_published_ts(str(a.get("published_at", ""))),
        -idx_a,
    )
    key_b = (
        _item_completeness_score(b),
        _parse_published_ts(str(b.get("published_at", ""))),
        -idx_b,
    )
    return idx_a if key_a >= key_b else idx_b


def _short_item_line(i: int, item: dict) -> str:
    source = str(item.get("source", "unknown")).upper()
    title = str(item.get("title", "")).replace("\n", " ").strip()
    url = str(item.get("url", "")).strip()
    published = str(item.get("published_at", "")).strip()
    feed_or_channel = str(item.get("feed_title", "") or item.get("channel", "")).strip()
    parts = [f"[{i}] [{source}] {title}"]
    if url:
        parts.append(f"url={url}")
    if published:
        parts.append(f"published_at={published}")
    if feed_or_channel:
        parts.append(f"meta={feed_or_channel}")
    return " | ".join(parts)


def _fallback_dedup_against_history(items: list[dict], history_records: list[dict]) -> list[int]:
    history_urls: set[str] = set()
    history_titles: list[str] = []
    history_titles_set: set[str] = set()

    for rec in history_records:
        nurl = _normalize_url(str(rec.get("url", "")))
        if nurl:
            history_urls.add(nurl)
        ntitle = _normalize_title(str(rec.get("title", "")))
        if ntitle and ntitle not in history_titles_set:
            history_titles.append(ntitle)
            history_titles_set.add(ntitle)

    kept: list[int] = []
    dropped = 0
    for idx, item in enumerate(items):
        nurl = _normalize_url(str(item.get("url", "")))
        ntitle = _normalize_title(str(item.get("title", "")))

        is_dup = False
        if nurl and nurl in history_urls:
            is_dup = True
        elif ntitle:
            if ntitle in history_titles_set:
                is_dup = True
            else:
                for h in history_titles:
                    if _is_strict_title_duplicate(ntitle, h):
                        is_dup = True
                        break

        if is_dup:
            dropped += 1
        else:
            kept.append(idx)

    logger.info(f"  历史去重（fallback）：{len(items)} → {len(kept)}（丢弃 {dropped}）")
    return kept


def _fallback_dedup_across_candidates(candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates

    url_groups: dict[str, list[int]] = {}
    for idx, item in enumerate(candidates):
        nurl = _normalize_url(str(item.get("url", "")))
        if not nurl:
            continue
        url_groups.setdefault(nurl, []).append(idx)

    kept_indices: set[int] = set(range(len(candidates)))
    for group in url_groups.values():
        if len(group) <= 1:
            continue
        keep = group[0]
        for idx in group[1:]:
            keep = _pick_better_item_index(candidates, keep, idx)
        for idx in group:
            if idx != keep:
                kept_indices.discard(idx)

    ordered = sorted(kept_indices)
    deduped_indices: list[int] = []
    for idx in ordered:
        candidate = candidates[idx]
        ntitle = _normalize_title(str(candidate.get("title", "")))
        merged = False
        for pos, existing_idx in enumerate(deduped_indices):
            existing_title = _normalize_title(str(candidates[existing_idx].get("title", "")))
            if not _is_strict_title_duplicate(ntitle, existing_title):
                continue
            better = _pick_better_item_index(candidates, existing_idx, idx)
            deduped_indices[pos] = better
            merged = True
            break
        if not merged:
            deduped_indices.append(idx)

    deduped_indices = sorted(set(deduped_indices))
    result = [candidates[i] for i in deduped_indices]
    logger.info(f"  跨源去重（fallback）：{len(candidates)} → {len(result)}")
    return result


def _parse_json_dict(raw_text: str) -> dict | None:
    json_match = re.search(r"\{[\s\S]*\}", raw_text or "")
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_system_prompt(taste_examples: list[dict], language: str = "zh", focus: str = "") -> str:
    lang_label = "中文" if language == "zh" else "English"
    base = f"""你是一个专业的内容策展助手，负责为用户筛选和摘要每日信息流。

用户的偏好语言是：{lang_label}

你的任务是对每一条内容进行：
1. **相关性评分**（1-10 分）：结合用户今日关注方向和历史喜好判断这条内容对用户的价值
2. **生成摘要**：用 2-4 句话提炼核心价值，说明"为什么值得关注"

评分标准：
- 9-10：极度相关，用户几乎肯定感兴趣
- 7-8：较相关，有一定参考价值
- 5-6：一般，勉强值得一看
- 1-4：不相关或质量低，不推荐
"""
    if focus:
        base += f"""
---
## 今日筛选方向

用户今天的关注重点是：**{focus}**

评分时请以此方向为首要参考：
- 与方向高度相关的内容优先给高分，即使历史上没有类似偏好
- 与方向完全无关的内容适当降分，即使内容本身质量不错
---
"""

    if taste_examples:
        base += "\n\n---\n## 用户历史高分内容（品味参考）\n\n"
        base += "以下是用户过去打高分（4-5/5）的内容，请参考这些来判断用户的偏好：\n\n"
        for i, ex in enumerate(taste_examples, 1):
            base += f"**示例 {i}**（用户评分 {ex['score']}/5）\n"
            base += f"- 标题: {ex['title']}\n"
            base += f"- 来源: {ex['source']}\n"
            if ex["summary"]:
                base += f"- 摘要: {ex['summary']}\n"
            if ex["notes"]:
                base += f"- 用户备注: {ex['notes']}\n"
            base += "\n"
        base += "---\n"

    return base


def _make_item_text(item: dict) -> str:
    source = item.get("source", "unknown")
    lines = [f"**来源**: {source.upper()}", f"**标题**: {item.get('title', '')}"]

    if item.get("url"):
        lines.append(f"**链接**: {item['url']}")

    if source == "github":
        if item.get("stars"):
            lines.append(f"**Stars**: {item['stars']:,}")
        if item.get("stars_gained"):
            lines.append(f"**近期新增**: {item['stars_gained']}")
        if item.get("language"):
            lines.append(f"**语言**: {item['language']}")
        if item.get("description"):
            lines.append(f"**简介**: {item['description']}")
        if item.get("readme_snippet"):
            lines.append(f"**README 片段**:\n{item['readme_snippet'][:800]}")

    elif source == "youtube":
        if item.get("channel"):
            lines.append(f"**频道**: {item['channel']}")
        if item.get("view_count"):
            lines.append(f"**播放量**: {item['view_count']:,}")
        if item.get("description"):
            lines.append(f"**视频描述**: {item['description']}")
        if item.get("transcript_snippet"):
            lines.append(f"**字幕片段**:\n{item['transcript_snippet'][:800]}")

    elif source == "rss":
        if item.get("feed_title"):
            lines.append(f"**订阅源**: {item['feed_title']}")
        if item.get("content_snippet"):
            lines.append(f"**正文片段**:\n{item['content_snippet'][:800]}")

    return "\n".join(lines)


def _score_single_item(
    item: dict,
    system_prompt: str,
    backend: str,
    call_kwargs: dict,
    min_score: int,
    idx: int,
    total: int,
) -> tuple[dict | None, dict | None]:
    """
    对单条内容调用 AI 打分+摘要。
    Returns: (high_score_item, low_score_item)，有分数的一侧非 None，失败时均为 None。
    """
    logger.info(f"  摘要进度: {idx+1}/{total} - {item.get('title', '')[:50]}")
    item_text = _make_item_text(item)
    user_message = f"""请对以下内容进行评估，并以 JSON 格式返回结果：

{item_text}

请严格按照以下 JSON 格式返回（不要包含其他文字）：
{{
  "score": <1到10的整数>,
  "summary": "<2-4句话的摘要，说明核心内容>",
  "reason": "<1-2句话说明为什么推荐或不推荐>"
}}
"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        raw_text = _call_ai(messages, backend, call_kwargs)
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            logger.warning(f"AI 返回格式异常: {raw_text[:100]}")
            return None, None
        parsed = json.loads(json_match.group())
        score = int(parsed.get("score", 0))
        enriched = dict(item)
        enriched["ai_score"] = score
        enriched["ai_summary"] = parsed.get("summary", "")
        enriched["ai_reason"] = parsed.get("reason", "")
        if score < min_score:
            logger.debug(f"  跳过低分内容 score={score}: {item.get('title', '')[:40]}")
            return None, enriched
        return enriched, None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
    except Exception as e:
        logger.error(f"AI 摘要失败: {e}")
    return None, None


def _batch_select_by_titles(
    items: list[dict],
    focus: str,
    taste_examples: list[dict],
    call_kwargs: dict,
    language: str,
    max_keep: int,
    backend: str = "litellm",
    history_titles: list[str] | None = None,
) -> list[int]:
    """
    第一阶段：仅凭标题+简介，一次 API 调用批量筛选值得深读的条目。

    Returns:
        值得保留的条目下标列表（0-based）。失败时回退为全量下标。
    """
    lang_label = "中文" if language == "zh" else "English"

    items_text = ""
    for i, item in enumerate(items):
        source = item.get("source", "unknown").upper()
        title = item.get("title", "")
        desc = (item.get("description") or item.get("content_snippet") or "")[:80]
        items_text += f"[{i}] [{source}] {title}"
        if desc:
            items_text += f"  —  {desc}"
        items_text += "\n"

    focus_line = f"用户今日关注方向：**{focus}**\n\n" if focus else ""

    taste_hint = ""
    if taste_examples:
        taste_hint = "用户历史偏好（高分内容样例）：\n"
        for ex in taste_examples[:3]:
            taste_hint += f"- {ex['title']}\n"
        taste_hint += "\n"

    history_hint = ""
    if history_titles:
        capped = history_titles[:200]
        history_hint = "以下是过去7天已推送过的内容标题，请避免推荐语义上相似的内容：\n"
        for t in capped:
            history_hint += f"- {t}\n"
        history_hint += "\n"
        logger.info(f"  历史去重：加载 {len(capped)} 条历史标题到 Stage 1 提示词")

    user_message = f"""{focus_line}{taste_hint}{history_hint}以下是 {len(items)} 条待筛选内容（格式：[序号] [来源] 标题  —  简介）：

{items_text}
请从中选出最多 {max_keep} 条最值得深度阅读的内容。

严格按照以下 JSON 格式返回（不要包含其他文字）：
{{"selected": [0, 3, 7]}}

selected 数组填入值得保留的条目序号（0-based 整数）。"""

    try:
        # 标题筛选只需要短回复，压缩 token 用量
        filter_kwargs = {**call_kwargs, "max_tokens": 256}
        messages = [
            {"role": "system", "content": f"你是内容筛选助手，请用{lang_label}思考，只输出 JSON。"},
            {"role": "user", "content": user_message},
        ]
        raw_text = _call_ai(messages, backend, filter_kwargs)
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            parsed = json.loads(json_match.group())
            selected = parsed.get("selected", [])
            valid = [i for i in selected if isinstance(i, int) and 0 <= i < len(items)]
            logger.info(f"  第一阶段筛选：{len(items)} → {len(valid)} 条入围")
            return valid[:max_keep]
    except Exception as e:
        logger.warning(f"标题批量筛选失败，回退到全量处理: {e}")

    return list(range(len(items)))


def _ai_dedup_against_history(
    items: list[dict],
    history_records: list[dict],
    call_kwargs: dict,
    language: str,
    backend: str = "litellm",
) -> list[int]:
    """
    阶段 A：当前候选与历史记录比较，先做严格去重（title/url 短信息）。
    """
    if not items:
        return []
    if not history_records:
        return list(range(len(items)))

    capped_history = history_records[:200]
    lang_label = "中文" if language == "zh" else "English"
    items_text = "\n".join(_short_item_line(i, item) for i, item in enumerate(items))
    history_text = "\n".join(
        f"- [{idx}] {str(rec.get('title', '')).strip()} | "
        f"url={str(rec.get('url', '')).strip()} | "
        f"source={str(rec.get('source', '')).strip()}"
        for idx, rec in enumerate(capped_history)
    )

    user_message = f"""请执行严格的历史去重判断（Strict Dedup）。

规则：
1) 优先以 URL 一致性判重：URL 规范化后相同则视为重复。
2) URL 不同但标题几乎一致、且语义明显是同一条新闻时，才判重。
3) 不要做主题级“泛化去重”（同主题不同新闻必须保留）。

当前候选（{len(items)} 条）：
{items_text}

历史已推送（最近 7 天，{len(capped_history)} 条）：
{history_text}

请仅输出 JSON：
{{
  "kept": [0, 2, 5],
  "dropped": [{{"index": 1, "reason": "same normalized url as history"}}]
}}
其中 kept 为保留的当前候选序号（0-based）。"""

    try:
        dedup_kwargs = {**call_kwargs, "max_tokens": 700}
        messages = [
            {"role": "system", "content": f"你是严格去重助手，请用{lang_label}思考，只输出 JSON。"},
            {"role": "user", "content": user_message},
        ]
        raw_text = _call_ai(messages, backend, dedup_kwargs)
        parsed = _parse_json_dict(raw_text)
        if parsed and isinstance(parsed.get("kept"), list):
            kept: list[int] = []
            seen: set[int] = set()
            for idx in parsed.get("kept", []):
                if not isinstance(idx, int) or not (0 <= idx < len(items)):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                kept.append(idx)

            dropped = parsed.get("dropped", [])
            if (
                not kept
                and items
                and not (isinstance(dropped, list) and len(dropped) >= len(items))
            ):
                logger.warning("  历史去重 AI 结果异常（kept 为空且无充分 dropped 信息），改用 fallback")
                return _fallback_dedup_against_history(items, capped_history)

            logger.info(
                f"  历史去重（AI）：{len(items)} → {len(kept)} "
                f"(history={len(capped_history)})"
            )
            return kept
    except Exception as e:
        logger.warning(f"历史去重 AI 失败，改用 fallback: {e}")

    return _fallback_dedup_against_history(items, capped_history)


def _ai_dedup_across_candidates(
    candidates: list[dict],
    focus: str,
    call_kwargs: dict,
    language: str,
    backend: str = "litellm",
) -> list[dict]:
    """
    阶段 B：标题筛选后、精读前做跨源去重。重复组由 AI 选择保留项。
    """
    if len(candidates) <= 1:
        return candidates

    lang_label = "中文" if language == "zh" else "English"
    focus_line = f"用户关注方向：{focus}\n\n" if focus else ""
    items_text = "\n".join(_short_item_line(i, item) for i, item in enumerate(candidates))
    user_message = f"""{focus_line}请对以下候选做跨源去重，目标是去掉“同一新闻/同一事件”的重复条目。

规则：
1) URL 规范化后一致，视为重复。
2) URL 不同但标题几乎一致且明显同一事件，可判为重复。
3) 对每个重复组，必须选择 1 条“最值得保留”的代表项。
4) 不要把“同主题但不同事件”的新闻误判成重复。

候选列表（{len(candidates)} 条）：
{items_text}

请仅输出 JSON：
{{
  "keep": [0, 3, 4],
  "groups": [
    {{"keep": 0, "drop": [1, 2], "reason": "same event from different sources"}}
  ]
}}
其中 keep 为最终保留的候选序号（0-based）。"""

    try:
        dedup_kwargs = {**call_kwargs, "max_tokens": 800}
        messages = [
            {"role": "system", "content": f"你是跨源去重助手，请用{lang_label}思考，只输出 JSON。"},
            {"role": "user", "content": user_message},
        ]
        raw_text = _call_ai(messages, backend, dedup_kwargs)
        parsed = _parse_json_dict(raw_text)
        if parsed and isinstance(parsed.get("keep"), list):
            keep: list[int] = []
            seen: set[int] = set()
            for idx in parsed.get("keep", []):
                if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                keep.append(idx)

            groups = parsed.get("groups", [])
            if not keep and candidates:
                logger.warning("  跨源去重 AI 结果异常（keep 为空），改用 fallback")
                return _fallback_dedup_across_candidates(candidates)

            logger.info(
                f"  跨源去重（AI）：{len(candidates)} → {len(keep)} "
                f"(groups={len(groups) if isinstance(groups, list) else 0})"
            )
            return [candidates[i] for i in keep]
    except Exception as e:
        logger.warning(f"跨源去重 AI 失败，改用 fallback: {e}")

    return _fallback_dedup_across_candidates(candidates)


def _ai_pick_fill_candidates(
    current_candidates: list[dict],
    remaining_pool: list[dict],
    need_count: int,
    focus: str,
    call_kwargs: dict,
    language: str,
    backend: str = "litellm",
) -> list[int]:
    """
    去重后候选不足时，从剩余池中补选“值得补全”的条目。
    允许返回空列表（代表不建议补全）。
    """
    if need_count <= 0 or not remaining_pool:
        return []

    lang_label = "中文" if language == "zh" else "English"
    current_preview = "\n".join(
        f"- {str(item.get('title', '')).strip()}"
        for item in current_candidates[:30]
        if str(item.get("title", "")).strip()
    )
    pool = remaining_pool[:120]
    pool_text = "\n".join(_short_item_line(i, item) for i, item in enumerate(pool))
    focus_line = f"用户关注方向：{focus}\n\n" if focus else ""

    user_message = f"""{focus_line}当前已入围 {len(current_candidates)} 条内容，仍缺 {need_count} 条以接近日报上限。
请在“剩余候选池”中挑选最多 {need_count} 条值得补全的内容；如果没有明显值得补全的内容，可以返回空数组。

当前已入围标题（用于避免重复）：
{current_preview or "(none)"}

剩余候选池（{len(pool)} 条）：
{pool_text}

请仅输出 JSON：
{{
  "supplement": [1, 5, 8],
  "reason": "..."
}}
其中 supplement 为剩余候选池的序号（0-based）。"""

    try:
        pick_kwargs = {**call_kwargs, "max_tokens": 500}
        messages = [
            {"role": "system", "content": f"你是内容补全助手，请用{lang_label}思考，只输出 JSON。"},
            {"role": "user", "content": user_message},
        ]
        raw_text = _call_ai(messages, backend, pick_kwargs)
        parsed = _parse_json_dict(raw_text)
        if parsed and isinstance(parsed.get("supplement"), list):
            selected: list[int] = []
            seen: set[int] = set()
            for idx in parsed.get("supplement", []):
                if not isinstance(idx, int) or not (0 <= idx < len(pool)):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                selected.append(idx)
                if len(selected) >= need_count:
                    break
            logger.info(
                f"  补全候选（AI）：pool={len(pool)} need={need_count} selected={len(selected)}"
            )
            return selected
    except Exception as e:
        logger.warning(f"补全候选 AI 失败，跳过补全: {e}")

    logger.info(f"  补全候选（fallback）：pool={len(pool)} need={need_count} selected=0")
    return []


def _normalize_source_minimums(raw_cfg) -> dict[str, int]:
    """
    解析来源保底配置。

    默认保底：
      - github >= 5
      - youtube >= 2

    配置方式（config.ai.min_items_per_source）支持覆盖与关闭：
      min_items_per_source:
        github: 5
        youtube: 2
      # 设置为 0 或负数可关闭某来源保底
    """
    minimums: dict[str, int] = {"github": 5, "youtube": 2}
    if not isinstance(raw_cfg, dict):
        return minimums

    for source, value in raw_cfg.items():
        src = str(source).strip().lower()
        if not src:
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            minimums.pop(src, None)
        else:
            minimums[src] = n
    return minimums


def _ensure_source_candidates(
    raw_items: list[dict],
    selected_indices: list[int],
    source_minimums: dict[str, int],
    max_keep: int,
) -> list[int]:
    """
    在阶段一候选池中补齐来源保底，避免某来源在标题筛选阶段被完全筛掉。
    """
    if not source_minimums:
        return selected_indices[:max_keep]

    selected: list[int] = []
    seen: set[int] = set()
    for idx in selected_indices:
        if isinstance(idx, int) and 0 <= idx < len(raw_items) and idx not in seen:
            selected.append(idx)
            seen.add(idx)

    added_counts: dict[str, int] = {}
    for source, minimum in source_minimums.items():
        current = sum(1 for idx in selected if raw_items[idx].get("source", "") == source)
        need = max(0, minimum - current)
        if need == 0:
            continue
        for idx, item in enumerate(raw_items):
            if need == 0:
                break
            if idx in seen:
                continue
            if item.get("source", "") != source:
                continue
            selected.append(idx)
            seen.add(idx)
            need -= 1
            added_counts[source] = added_counts.get(source, 0) + 1

    if added_counts:
        logger.info(f"  阶段一补齐来源候选: {added_counts}")

    if len(selected) <= max_keep:
        return selected

    # 超出 max_keep 时，优先保留来源保底所需条目，再按原顺序补齐其余条目
    protected: list[int] = []
    protected_counts: dict[str, int] = {}
    for idx in selected:
        src = raw_items[idx].get("source", "")
        limit = source_minimums.get(src, 0)
        if limit > 0 and protected_counts.get(src, 0) < limit:
            protected.append(idx)
            protected_counts[src] = protected_counts.get(src, 0) + 1

    if len(protected) >= max_keep:
        trimmed = protected[:max_keep]
    else:
        protected_set = set(protected)
        trimmed = list(protected)
        for idx in selected:
            if len(trimmed) >= max_keep:
                break
            if idx in protected_set:
                continue
            trimmed.append(idx)

    logger.info(f"  阶段一候选裁剪：{len(selected)} → {len(trimmed)} 条")
    return trimmed


def _item_key(item: dict) -> str:
    normalized_url = _normalize_url(str(item.get("url", "")))
    if normalized_url:
        return normalized_url
    source = str(item.get("source", "unknown")).strip().lower()
    title = _normalize_title(str(item.get("title", "")))
    return f"{source}::{title}"


def _enforce_source_minimums(
    selected: list[dict],
    high_score_items: list[dict],
    low_score_items: list[dict],
    source_minimums: dict[str, int],
    max_output: int,
) -> list[dict]:
    """
    在最终输出阶段执行来源保底：
      1) 优先使用高分条目补齐
      2) 不足时用低分条目兜底
    """
    if not source_minimums:
        return selected[:max_output]

    result = list(selected)
    used_keys = {_item_key(item) for item in result}

    pools: dict[str, list[dict]] = {}
    for item in high_score_items + low_score_items:
        src = item.get("source", "")
        if src not in source_minimums:
            continue
        pools.setdefault(src, []).append(item)

    for items in pools.values():
        items.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

    supplemented: dict[str, int] = {}

    for source, minimum in source_minimums.items():
        if minimum <= 0:
            continue
        pool = pools.get(source, [])
        pool_idx = 0

        while sum(1 for item in result if item.get("source", "") == source) < minimum:
            if pool_idx >= len(pool):
                break

            candidate = pool[pool_idx]
            pool_idx += 1
            key = _item_key(candidate)
            if key in used_keys:
                continue

            if len(result) < max_output:
                result.append(candidate)
                used_keys.add(key)
                supplemented[source] = supplemented.get(source, 0) + 1
                continue

            # 已满时，尝试替换掉“可被挤出的低分项”
            counts = Counter(item.get("source", "") for item in result)
            evict_idx = None
            evict_score = None
            for idx, existing in enumerate(result):
                existing_src = existing.get("source", "")
                if counts[existing_src] <= source_minimums.get(existing_src, 0):
                    continue
                score = existing.get("ai_score", 0)
                if evict_idx is None or score < evict_score:
                    evict_idx = idx
                    evict_score = score

            if evict_idx is None:
                break

            removed = result.pop(evict_idx)
            used_keys.discard(_item_key(removed))
            result.append(candidate)
            used_keys.add(key)
            supplemented[source] = supplemented.get(source, 0) + 1

    if supplemented:
        logger.info(f"  最终来源保底补齐: {supplemented}")

    result.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    return result[:max_output]


def summarize_items(
    raw_items: list[dict],
    config: dict,
    min_score: Optional[int] = None,
    max_output: Optional[int] = None,
    focus: str = "",
) -> list[dict]:
    """
    两阶段处理：
      阶段一：AI 仅看标题+简介，一次调用批量筛选入围条目
              （RSS 在此阶段会多条目进池，之后按 per_feed_limit 封顶）
      阶段二：仅对入围条目逐条调用 AI，生成完整评分+摘要

    Args:
        raw_items: 来自各 collector 的原始数据列表
        config:    AppConfig dict
        min_score: 低于此分数的条目被过滤（默认读 config）
        max_output: 最多返回条目数（默认读 config）
        focus:     本次筛选方向（来自 schedule.focus），传给 AI 做相关度评分

    Returns:
        过滤并排序后的列表，每项新增：
            - ai_score: int (1-10)
            - ai_summary: str
            - ai_reason: str
    """
    if not raw_items:
        return []

    ai_cfg = config.get("ai", {})
    rss_cfg = config.get("collectors", {}).get("rss", {})

    min_score_default = _safe_positive_int(ai_cfg.get("min_relevance_score", 5), 5)
    min_score = (
        _safe_positive_int(min_score, min_score_default)
        if min_score is not None
        else min_score_default
    )

    default_cap = 15
    raw_config_cap = ai_cfg.get("max_items_per_digest", default_cap)
    config_cap = _safe_positive_int(raw_config_cap, default_cap)
    if config_cap != raw_config_cap:
        logger.warning(
            f"ai.max_items_per_digest 非法({raw_config_cap!r})，回退默认值 {default_cap}"
        )

    if max_output is None:
        requested_max = config_cap
    else:
        requested_max = _safe_positive_int(max_output, config_cap)
        if requested_max != max_output:
            logger.warning(f"max_output 参数非法({max_output!r})，回退配置上限 {config_cap}")
    effective_max_output = min(requested_max, config_cap)

    logger.info(
        "  新闻筛选开始: "
        f"raw_count={len(raw_items)} "
        f"requested_max={requested_max} "
        f"config_cap={config_cap} "
        f"effective_cap={effective_max_output} "
        f"min_score={min_score}"
    )

    # RSS 每 feed 最多进入摘要阶段的条数
    rss_per_feed_limit: int = _safe_positive_int(rss_cfg.get("max_items_per_feed", 3), 3)
    model     = os.environ.get("AI_MODEL")     or ai_cfg.get("model", "openai/gpt-4o-mini")
    api_base  = os.environ.get("AI_API_BASE")  or ai_cfg.get("api_base") or None
    max_tokens = ai_cfg.get("max_tokens", 512)
    taste_limit = ai_cfg.get("taste_examples_limit", 8)
    source_minimums = _normalize_source_minimums(ai_cfg.get("min_items_per_source"))
    language = config.get("app", {}).get("language", "zh")

    backend = os.environ.get("AI_BACKEND") or ai_cfg.get("backend", "litellm")
    api_key = os.environ.get("AI_API_KEY", "")
    if backend == "litellm" and not api_key:
        logger.error("AI_API_KEY 未配置，跳过 AI 摘要（backend=litellm 需要 API key）")
        return raw_items[:effective_max_output]

    call_kwargs: dict = dict(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    if api_base:
        call_kwargs["api_base"] = api_base

    taste_examples = load_taste_examples(config, limit=taste_limit)
    history_records = load_recent_history_records(config, days=7, limit=600)
    history_titles = []
    seen_history_titles: set[str] = set()
    for rec in history_records:
        title = str(rec.get("title", "")).strip()
        if not title or title in seen_history_titles:
            continue
        seen_history_titles.add(title)
        history_titles.append(title)
        if len(history_titles) >= 120:
            break

    # ── 阶段 A：历史去重（title/url 短信息）──────────────────────
    kept_indices_after_history = _ai_dedup_against_history(
        raw_items,
        history_records,
        call_kwargs=call_kwargs,
        language=language,
        backend=backend,
    )
    items_after_history = [raw_items[i] for i in kept_indices_after_history]
    logger.info(f"  after_history_dedup_count={len(items_after_history)}")
    if not items_after_history:
        logger.info(
            "  新闻筛选完成: final_count=0 "
            f"effective_cap={effective_max_output} requested_max={requested_max} config_cap={config_cap}"
        )
        return []

    # ── 阶段一：标题批量筛选 ──────────────────────────────────
    # 预留 effective_max_output 的 2 倍进入第二阶段，给评分留余量
    max_keep = min(effective_max_output * 2, len(items_after_history))
    selected_indices = _batch_select_by_titles(
        items_after_history,
        focus,
        taste_examples,
        call_kwargs,
        language,
        max_keep,
        backend=backend, history_titles=history_titles,
    )
    selected_indices = _ensure_source_candidates(
        items_after_history, selected_indices, source_minimums, max_keep
    )
    candidates = [items_after_history[i] for i in selected_indices]
    logger.info(f"  after_stage1_select_count={len(candidates)}")
    if not candidates:
        logger.info(
            "  新闻筛选完成: final_count=0 "
            f"effective_cap={effective_max_output} requested_max={requested_max} config_cap={config_cap}"
        )
        return []

    # ── 阶段 B：跨源去重（精读前）──────────────────────────────
    candidates = _ai_dedup_across_candidates(
        candidates,
        focus=focus,
        call_kwargs=call_kwargs,
        language=language,
        backend=backend,
    )
    logger.info(f"  after_cross_source_dedup_count={len(candidates)}")
    if not candidates:
        logger.info(
            "  新闻筛选完成: final_count=0 "
            f"effective_cap={effective_max_output} requested_max={requested_max} config_cap={config_cap}"
        )
        return []

    # ── 阶段 B 后：RSS 每 feed 封顶 ────────────────────────────
    # 去重后先做一次 RSS feed 封顶，得到可进入精读的基础池
    feed_counts: dict[str, int] = {}
    capped: list[dict] = []
    for item in candidates:
        if item.get("source") == "rss":
            feed_key = item.get("feed_title", item.get("url", "unknown"))
            feed_counts[feed_key] = feed_counts.get(feed_key, 0) + 1
            if feed_counts[feed_key] > rss_per_feed_limit:
                continue
        capped.append(item)
    if len(capped) < len(candidates):
        logger.info(f"  RSS per-feed 封顶：{len(candidates)} → {len(capped)} 条进入摘要")
    candidates = capped

    # ── 去重后不足上限时：从剩余池补全候选（允许补不满）──────────
    if len(candidates) < effective_max_output:
        need_fill = effective_max_output - len(candidates)
        existing_keys = {_item_key(item) for item in candidates}
        remaining_pool = [
            item for item in items_after_history
            if _item_key(item) not in existing_keys
        ]
        fill_indices = _ai_pick_fill_candidates(
            current_candidates=candidates,
            remaining_pool=remaining_pool,
            need_count=need_fill,
            focus=focus,
            call_kwargs=call_kwargs,
            language=language,
            backend=backend,
        )
        filled = 0
        for idx in fill_indices:
            if idx < 0 or idx >= len(remaining_pool):
                continue
            item = remaining_pool[idx]
            key = _item_key(item)
            if key in existing_keys:
                continue
            if item.get("source") == "rss":
                feed_key = item.get("feed_title", item.get("url", "unknown"))
                if feed_counts.get(feed_key, 0) >= rss_per_feed_limit:
                    continue
                feed_counts[feed_key] = feed_counts.get(feed_key, 0) + 1
            candidates.append(item)
            existing_keys.add(key)
            filled += 1
            if len(candidates) >= effective_max_output:
                break
        logger.info(
            f"  after_fill_candidates_count={len(candidates)} "
            f"(filled={filled}, need={need_fill})"
        )

    if not candidates:
        logger.info(
            "  新闻筛选完成: final_count=0 "
            f"effective_cap={effective_max_output} requested_max={requested_max} config_cap={config_cap}"
        )
        return []

    # ── 精读前：为 YouTube 入选视频补充字幕（仅对最终候选）────────
    yt_candidates = [item for item in candidates if item.get("source") == "youtube"]
    if yt_candidates:
        logger.info(f"  拉取 YouTube 字幕（{len(yt_candidates)} 个视频）...")
        try:
            from src.collectors.youtube_collector import _get_transcript
            for item in yt_candidates:
                video_id = item.get("video_id", "")
                if not video_id:
                    url = item.get("url", "")
                    if "v=" in url:
                        video_id = url.split("v=")[-1].split("&")[0]
                if video_id:
                    item["transcript_snippet"] = _get_transcript(video_id)
        except Exception as e:
            logger.warning(f"YouTube 字幕补充失败: {e}")

    # ── 阶段二：并行完整评分+摘要 ─────────────────────────────
    system_prompt = _build_system_prompt(taste_examples, language, focus=focus)
    results: list[dict] = []
    low_score_pool: list[dict] = []
    max_workers = ai_cfg.get("max_workers", 5)
    total = len(candidates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_single_item,
                item, system_prompt, backend, call_kwargs, min_score, i, total,
            ): item
            for i, item in enumerate(candidates)
        }
        for future in as_completed(futures):
            high, low = future.result()
            if high is not None:
                results.append(high)
            elif low is not None:
                low_score_pool.append(low)

    logger.info(f"  after_stage2_score_count={len(results)}")
    results.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

    # 按来源分桶，每源保底

    source_buckets: dict[str, list] = {}
    for item in results:
        src = item.get("source", "unknown")
        if src not in source_buckets:
            source_buckets[src] = []
        source_buckets[src].append(item)

    per_source_min = (
        effective_max_output // len(source_buckets)
        if source_buckets
        else effective_max_output
    )

    selected: list[dict] = []
    leftover: list[dict] = []
    for items in source_buckets.values():
        selected.extend(items[:per_source_min])
        leftover.extend(items[per_source_min:])

    remaining = effective_max_output - len(selected)
    if remaining > 0 and leftover:
        leftover.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        selected.extend(leftover[:remaining])

    selected = _enforce_source_minimums(
        selected=selected,
        high_score_items=results,
        low_score_items=low_score_pool,
        source_minimums=source_minimums,
        max_output=effective_max_output,
    )
    final_items = selected[:effective_max_output]
    final_items.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    logger.info(
        "  新闻筛选完成: "
        f"final_count={len(final_items)} "
        f"effective_cap={effective_max_output} "
        f"requested_max={requested_max} "
        f"config_cap={config_cap}"
    )
    return final_items


def generate_digest_summary(
    news_items: list[dict],
    config: dict,
    focus: str = "",
) -> str:
    """
    对已筛选的 news_items 生成「今日要点」整体总结。
    单次 AI 调用，token 消耗极小。

    Returns:
        总结文本（纯文本，含换行），失败时返回空字符串。
    """
    if not news_items:
        return ""

    ai_cfg = config.get("ai", {})
    model    = os.environ.get("AI_MODEL")    or ai_cfg.get("model", "openai/gpt-4o-mini")
    api_base = os.environ.get("AI_API_BASE") or ai_cfg.get("api_base") or None
    backend  = os.environ.get("AI_BACKEND") or ai_cfg.get("backend", "litellm")
    api_key  = os.environ.get("AI_API_KEY", "")
    language = config.get("app", {}).get("language", "zh")

    if backend == "litellm" and not api_key:
        return ""

    call_kwargs: dict = dict(model=model, api_key=api_key, max_tokens=600)
    if api_base:
        call_kwargs["api_base"] = api_base

    lang_label = "中文" if language == "zh" else "English"
    focus_line = f"本次关注方向：{focus}\n\n" if focus else ""

    items_text = ""
    for i, item in enumerate(news_items, 1):
        source  = item.get("source", "").upper()
        title   = item.get("title", "")
        summary = item.get("ai_summary", "")
        score   = item.get("ai_score", "?")
        items_text += f"{i}. [{source}][{score}/10] {title}\n"
        if summary:
            items_text += f"   {summary}\n"

    user_message = f"""{focus_line}以下是今日精选的 {len(news_items)} 条内容：

{items_text}
请用{lang_label}撰写「今日要点」总结：
- 提炼 3-5 条最值得关注的主题或趋势
- 每条要点 1-2 句，言简意赅
- 覆盖不同领域（AI/科技/金融/政治等）
- 直接输出要点列表，每条以「• 」开头，不需要标题或其他说明文字"""

    try:
        messages = [
            {"role": "system", "content": f"你是专业的信息分析师，擅长跨领域提炼要点，请用{lang_label}输出。"},
            {"role": "user",   "content": user_message},
        ]
        return _call_ai(messages, backend, call_kwargs)
    except Exception as e:
        logger.warning(f"生成今日要点失败: {e}")
        return ""
