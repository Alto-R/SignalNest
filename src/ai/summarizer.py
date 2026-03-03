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
from typing import Optional

try:
    import litellm
    litellm.suppress_debug_info = True
except ImportError:
    litellm = None  # type: ignore

from src.ai.cli_backend import _call_ai
from src.ai.feedback import load_taste_examples, load_recent_titles

logger = logging.getLogger(__name__)


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
    return item.get("url") or f"{item.get('source', 'unknown')}::{item.get('title', '')}"


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
    min_score = min_score or ai_cfg.get("min_relevance_score", 5)
    max_output = max_output or ai_cfg.get("max_items_per_digest", 15)
    # RSS 每 feed 最多进入摘要阶段的条数
    rss_per_feed_limit: int = rss_cfg.get("max_items_per_feed", 3)
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
        return raw_items[:max_output]

    call_kwargs: dict = dict(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    if api_base:
        call_kwargs["api_base"] = api_base

    taste_examples = load_taste_examples(config, limit=taste_limit)
    history_titles = load_recent_titles(config, days=7)

    # ── 阶段一：标题批量筛选 ──────────────────────────────────
    # 预留 max_output 的 2 倍进入第二阶段，给评分留余量
    max_keep = min(max_output * 2, len(raw_items))
    selected_indices = _batch_select_by_titles(
        raw_items, focus, taste_examples, call_kwargs, language, max_keep,
        backend=backend, history_titles=history_titles,
    )
    selected_indices = _ensure_source_candidates(
        raw_items, selected_indices, source_minimums, max_keep
    )
    candidates = [raw_items[i] for i in selected_indices]

    # ── 阶段一后：为 YouTube 入选视频补充字幕（精读素材）────────
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

    # ── 阶段一后：RSS 每 feed 封顶 ────────────────────────────
    # RSS 初始多抓了很多条（max_items_per_feed_initial），这里把每个 feed 的候选
    # 限制到 rss_per_feed_limit 条，防止单个 feed 占满摘要名额
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

    results.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

    # 按来源分桶，每源保底

    source_buckets: dict[str, list] = {}
    for item in results:
        src = item.get("source", "unknown")
        if src not in source_buckets:
            source_buckets[src] = []
        source_buckets[src].append(item)

    per_source_min = max_output // len(source_buckets) if source_buckets else max_output

    selected: list[dict] = []
    leftover: list[dict] = []
    for items in source_buckets.values():
        selected.extend(items[:per_source_min])
        leftover.extend(items[per_source_min:])

    remaining = max_output - len(selected)
    if remaining > 0 and leftover:
        leftover.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        selected.extend(leftover[:remaining])

    selected = _enforce_source_minimums(
        selected=selected,
        high_score_items=results,
        low_score_items=low_score_pool,
        source_minimums=source_minimums,
        max_output=max_output,
    )
    selected.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    return selected


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
