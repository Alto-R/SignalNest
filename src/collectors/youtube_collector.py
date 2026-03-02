"""
youtube_collector.py - YouTube 内容抓取器
==========================================
v4: 两阶段架构——采集阶段只拿标题+播放量，字幕由 summarizer 阶段一筛选后按需拉取。
    新增：AI 根据 focus 自动推导关键词，通过 YouTube Search API 搜索其他频道视频。
"""

import json
import logging
import os
import re
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

logger = logging.getLogger(__name__)

YT_API_BASE = "https://www.googleapis.com/youtube/v3"
VIDEO_URL = "https://www.youtube.com/watch?v={video_id}"


class YouTubeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get(self, endpoint: str, **params) -> dict:
        params["key"] = self.api_key
        resp = self.session.get(f"{YT_API_BASE}/{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def channels(self, **params) -> dict:
        return self.get("channels", **params)

    def playlist_items(self, **params) -> dict:
        return self.get("playlistItems", **params)

    def videos(self, **params) -> dict:
        return self.get("videos", **params)

    def search(self, **params) -> dict:
        return self.get("search", **params)


def _normalize_sort_by(value: str, field_name: str, default: str = "views") -> str:
    """
    规范化排序方式，当前仅支持 views / date。
    """
    raw = (value or "").strip().lower()
    if raw in {"views", "date"}:
        return raw
    logger.warning(f"{field_name}={value!r} 非法，回退为 {default!r}")
    return default


def _get_transcript(video_id: str, max_chars: int = 2000) -> str:
    """拉取视频字幕（供 summarizer 阶段一筛选后调用）。"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        for lang in ["zh", "zh-Hans", "zh-CN", "en"]:
            try:
                transcript = transcript_list.find_transcript([lang])
                entries = transcript.fetch()
                text = " ".join(e["text"] for e in entries)
                return text[:max_chars]
            except Exception:
                continue

        try:
            transcript = transcript_list.find_generated_transcript(["en", "zh"])
            entries = transcript.fetch()
            text = " ".join(e["text"] for e in entries)
            return text[:max_chars]
        except Exception:
            pass

    except (NoTranscriptFound, TranscriptsDisabled):
        pass
    except Exception as e:
        logger.debug(f"获取字幕失败 video_id={video_id}: {e}")

    return ""


def _get_video_stats(yt: YouTubeClient, video_ids: list[str]) -> dict[str, int]:
    """批量获取视频播放量，返回 {video_id: view_count}。"""
    if not video_ids:
        return {}
    stats: dict[str, int] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        try:
            resp = yt.videos(id=",".join(chunk), part="statistics")
            for item in resp.get("items", []):
                vid_id = item["id"]
                vc = item.get("statistics", {}).get("viewCount", "0")
                stats[vid_id] = int(vc)
        except Exception as e:
            logger.warning(f"获取视频统计失败: {e}")
    return stats


def _fetch_channel_videos(
    yt: YouTubeClient,
    channel_id: str,
    days_back: int,
    max_results: int,
    sort_by: str = "views",
) -> list[dict]:
    """
    获取订阅频道近期视频的标题和播放量（不拉字幕）。

    策略：
      1. 获取 max_results × 3 条最新上传
      2. 批量拉取播放量排序
      3. 返回 top-N 基础信息，字幕由 summarizer 阶段一后按需拉取
    """
    raw_videos: list[dict] = []
    try:
        ch_response = yt.channels(id=channel_id, part="contentDetails")
        channels = ch_response.get("items", [])
        if not channels:
            logger.warning(f"频道 {channel_id} 不存在或无权限")
            return []

        uploads_playlist_id = channels[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        fetch_count = min(max_results * 3, 50)
        pl_response = yt.playlist_items(
            playlistId=uploads_playlist_id,
            part="snippet",
            maxResults=fetch_count,
        )

        for item in pl_response.get("items", []):
            snippet = item.get("snippet", {})
            published = snippet.get("publishedAt", "")
            if published:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_dt < cutoff:
                    continue

            video_id = snippet.get("resourceId", {}).get("videoId", "")
            if not video_id:
                continue

            title = snippet.get("title", "")
            if not title:
                continue

            raw_videos.append({
                "video_id": video_id,
                "title": title,
                "description": snippet.get("description", "")[:300],
                "channel": snippet.get("channelTitle", ""),
                "published_at": published,
                "view_count": 0,
            })

    except requests.HTTPError as e:
        logger.error(f"频道视频获取失败 channel_id={channel_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"频道视频获取异常 channel_id={channel_id}: {e}")
        return []

    if not raw_videos:
        return []

    if sort_by == "views":
        video_ids = [v["video_id"] for v in raw_videos]
        stats = _get_video_stats(yt, video_ids)
        for v in raw_videos:
            v["view_count"] = stats.get(v["video_id"], 0)
        raw_videos.sort(key=lambda x: x["view_count"], reverse=True)

    # 取 top-N，不拉字幕（字幕由 summarizer 阶段一筛选后按需拉取）
    results: list[dict] = []
    for v in raw_videos[:max_results]:
        results.append({
            "video_id": v["video_id"],
            "title": v["title"],
            "url": VIDEO_URL.format(video_id=v["video_id"]),
            "description": v["description"],
            "channel": v["channel"],
            "published_at": v["published_at"],
            "view_count": v["view_count"],
            "transcript_snippet": "",   # 占位，由 summarizer 按需填充
            "source": "youtube",
        })

    return results


def _search_by_keyword(
    yt: YouTubeClient,
    keyword: str,
    days_back: int,
    max_results: int,
    sort_by: str = "views",
) -> list[dict]:
    """
    通过关键词搜索 YouTube 视频，支持按热度/时间排序，返回基础信息（不含字幕）。
    两种排序模式都先多拉候选（max_results * 3）再本地筛选。
    """
    sort_by = _normalize_sort_by(sort_by, "search_sort_by")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    published_after = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    search_order = "viewCount" if sort_by == "views" else "date"
    fetch_count = min(max_results * 3, 50)

    raw_videos: list[dict] = []
    try:
        resp = yt.search(
            q=keyword,
            part="snippet",
            type="video",
            order=search_order,
            publishedAfter=published_after,
            maxResults=fetch_count,
        )
        for item in resp.get("items", []):
            video_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            if not video_id or not title:
                continue
            raw_videos.append({
                "video_id": video_id,
                "title": title,
                "description": snippet.get("description", "")[:300],
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": 0,
            })
    except Exception as e:
        logger.warning(f"YouTube 关键词搜索失败 keyword={keyword!r}: {e}")
        return []

    if not raw_videos:
        return []

    if sort_by == "views":
        # 批量拉播放量，二次排序
        video_ids = [v["video_id"] for v in raw_videos]
        stats = _get_video_stats(yt, video_ids)
        for v in raw_videos:
            v["view_count"] = stats.get(v["video_id"], 0)
        raw_videos.sort(key=lambda x: x["view_count"], reverse=True)
    else:
        # date 模式也执行本地二次筛选，保证“多拉再筛”行为一致
        raw_videos.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    results: list[dict] = []
    for v in raw_videos[:max_results]:
        results.append({
            "video_id": v["video_id"],
            "title": v["title"],
            "url": VIDEO_URL.format(video_id=v["video_id"]),
            "description": v["description"],
            "channel": v["channel"],
            "published_at": v["published_at"],
            "view_count": v["view_count"],
            "transcript_snippet": "",
            "source": "youtube",
        })

    return results


def _ai_generate_keywords(focus: str, config: dict) -> list[str]:
    """
    调用 AI 根据 focus 方向生成 3-5 个 YouTube 搜索关键词短语。
    失败时返回空列表（graceful degradation）。
    """
    import litellm

    ai_cfg = config.get("ai", {})
    model = os.environ.get("AI_MODEL") or ai_cfg.get("model", "openai/gpt-4o-mini")
    api_base = os.environ.get("AI_API_BASE") or ai_cfg.get("api_base") or None
    api_key = os.environ.get("AI_API_KEY", "")

    if not api_key:
        logger.warning("AI_API_KEY 未配置，跳过关键词生成")
        return []

    call_kwargs: dict = dict(model=model, api_key=api_key, max_tokens=200)
    if api_base:
        call_kwargs["api_base"] = api_base

    prompt = f"""根据以下关注方向，生成 3-5 个适合在 YouTube 搜索的英文关键词短语（每个 2-4 个词）。
关键词要能找到近期的高质量相关视频。

关注方向：{focus}

严格按以下 JSON 格式返回（不包含其他文字）：
{{"keywords": ["keyword phrase 1", "keyword phrase 2", "keyword phrase 3"]}}"""

    try:
        import litellm
        litellm.suppress_debug_info = True
        response = litellm.completion(
            messages=[
                {"role": "system", "content": "你是 YouTube 内容搜索助手，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            **call_kwargs,
        )
        raw = response.choices[0].message.content.strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            parsed = json.loads(m.group())
            keywords = parsed.get("keywords", [])
            if isinstance(keywords, list):
                return [str(k) for k in keywords[:5] if k]
    except Exception as e:
        logger.warning(f"AI 生成 YouTube 关键词失败: {e}")

    return []


def collect_youtube(
    config: dict,
    focus: str = "",
    max_total: Optional[int] = None,
) -> list[dict]:
    """
    采集 YouTube 视频（标题+播放量，不含字幕）。

    两路来源：
      1. 订阅频道（channel_ids）：按热度取 top-N
      2. 关键词搜索（enable_keyword_search=true）：AI 根据 focus 推导关键词搜索

    字幕由 summarizer 阶段一筛选后按需拉取，以节省 API 配额。
    """
    yt_cfg = config.get("collectors", {}).get("youtube", {})
    if not yt_cfg.get("enabled", True):
        return []

    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        logger.warning("YOUTUBE_API_KEY 未配置，跳过 YouTube 抓取")
        return []

    yt = YouTubeClient(api_key)
    max_per_channel = yt_cfg.get("max_results_per_channel", 5)
    days_back = yt_cfg.get("days_lookback", 3)
    sort_by = _normalize_sort_by(yt_cfg.get("sort_by", "views"), "sort_by")
    channel_ids = yt_cfg.get("channel_ids", [])
    enable_keyword_search = yt_cfg.get("enable_keyword_search", False)
    max_search_results = yt_cfg.get("max_search_results", 10)
    search_sort_by = _normalize_sort_by(
        yt_cfg.get("search_sort_by", sort_by),
        "search_sort_by",
    )
    # 关键词搜索单独的时间窗口（热点时效性更短，默认比订阅频道窗口小）
    search_days_back = yt_cfg.get("search_days_lookback", days_back)

    collected: list[dict] = []
    seen_urls: set[str] = set()

    # ── 路线一：订阅频道 ──────────────────────────────────────
    for channel_id in channel_ids:
        videos = _fetch_channel_videos(
            yt, channel_id, days_back,
            max_results=max_per_channel,
            sort_by=sort_by,
        )
        for v in videos:
            if v["url"] not in seen_urls:
                seen_urls.add(v["url"])
                collected.append(v)

    logger.info(f"YouTube 订阅频道：{len(collected)} 个视频")

    # ── 路线二：AI 关键词搜索（其他频道） ──────────────────────
    if enable_keyword_search and focus:
        logger.info("🔍 AI 推导 YouTube 搜索关键词...")
        keywords = _ai_generate_keywords(focus, config)
        if keywords:
            logger.info(f"   关键词: {keywords}")
            search_added = 0
            for kw in keywords:
                videos = _search_by_keyword(
                    yt,
                    kw,
                    search_days_back,
                    max_results=max_search_results,
                    sort_by=search_sort_by,
                )
                for v in videos:
                    if v["url"] not in seen_urls:
                        seen_urls.add(v["url"])
                        collected.append(v)
                        search_added += 1
            logger.info(f"YouTube 关键词搜索：新增 {search_added} 个视频")
    elif enable_keyword_search and not focus:
        logger.info("enable_keyword_search=true 但 focus 为空，跳过关键词搜索")

    if max_total:
        collected = collected[:max_total]

    logger.info(f"YouTube: 共收集 {len(collected)} 个视频（字幕将在 AI 筛选后按需拉取）")
    return collected
