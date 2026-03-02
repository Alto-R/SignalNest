<div align="center">

# SignalNest 📡

每天定时推送的个人 AI 日报 —— 聚合 GitHub / YouTube / RSS，两阶段 AI 筛选摘要，邮件直达收件箱

[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-docker-部署)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![supercronic](https://img.shields.io/badge/scheduler-supercronic-orange?style=flat-square)](https://github.com/aptible/supercronic)

[![邮件推送](https://img.shields.io/badge/Email-HTML富文本-00D4AA?style=flat-square)](#)
[![飞书推送](https://img.shields.io/badge/飞书-Webhook-00D4AA?style=flat-square)](https://www.feishu.cn/)
[![企业微信推送](https://img.shields.io/badge/企业微信-Webhook-00D4AA?style=flat-square)](https://work.weixin.qq.com/)

**[中文](README.md)** | **[English](README-EN.md)**

</div>

<br>

## 📑 快速导航

<div align="center">

| | | |
|:---:|:---:|:---:|
| [🚀 快速开始](#-快速开始) | [⚙️ 配置详解](#️-配置详解) | [🐳 Docker 部署](#-docker-部署) |
| [🎯 核心功能](#-核心功能) | [🧠 偏好学习](#-内容偏好学习) | [❓ 常见问题](#-常见问题) |

</div>

<br>

## 🎯 核心功能

- **三大信息源**：GitHub 热门仓库 / YouTube 精选视频 / RSS 订阅，按需组合
- **focus 定向筛选**：每个调度可设置今日关注方向，AI 优先推送与方向高度相关的内容
- **两阶段 AI 处理**：先批量标题筛选（省 token），再对入选内容精读评分+摘要，高效又精准
- **今日要点总结**：AI 在摘要后额外生成 3-5 条跨领域「今日要点」，一眼掌握全局
- **来源保底机制**：可按来源设置最小条数（默认 GitHub≥5、YouTube≥2），防止单一来源挤占
- **YouTube 双路采集**：订阅频道（按热度/最新排序）+ AI 根据 focus 自动推导关键词搜索其他频道
- **偏好学习**：通过反馈打分，AI 逐渐学习你的内容偏好，推送越来越精准
- **个人助手**：晨间日程提醒 + 项目任务到期检查（逾期 / 今日 / 即将到期）
- **多渠道推送**：邮件（HTML 富文本）+ 飞书 + 企业微信，个人内容仅发送给发件人
- **多时间点调度**：在 `config.yaml` 中任意定义 cron 时间点，不同时段推送不同内容
- **灵活 AI 后端**：支持 LiteLLM（任意云端 API）/ Claude CLI / Codex CLI 三种后端

<br>

## 🚀 快速开始

### 第一步：配置环境变量

```bash
cd SignalNest/docker/
cp .env.example .env
```

编辑 `docker/.env`，填写必填项：

```dotenv
# AI（必填）
AI_API_KEY=your_api_key_here
AI_MODEL=openai/gpt-4o          # LiteLLM 格式：provider/model_name
AI_API_BASE=                    # 中转服务填写端点，官方接口留空

# 邮件（必填）
EMAIL_FROM=your_email@qq.com
EMAIL_PASSWORD=your_smtp_password   # QQ/163 邮箱使用「授权码」，非登录密码
EMAIL_TO=recipient@example.com      # 多收件人用逗号分隔
```

<details>
<summary>可选配置（GitHub / YouTube / 飞书 / 企业微信）</summary>

```dotenv
# GitHub Token（不填则每小时限 60 次请求）
GITHUB_TOKEN=ghp_xxxxx

# YouTube Data API v3（不填则跳过 YouTube 采集）
YOUTUBE_API_KEY=AIzaSy_xxxxx

# 飞书群机器人
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxx

# 企业微信群机器人
WEWORK_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxx
```

</details>

> **QQ 邮箱授权码**：登录 QQ 邮箱 → 设置 → 账户 → 开启 SMTP 服务 → 生成授权码

---

### 第二步：调整调度与信息源

编辑 `config/config.yaml`：

```yaml
schedules:
  - name: "早间日报"
    cron: "0 8 * * *"
    content: [schedule, todos, news]   # 日程 + 项目提醒 + 新闻
    sources: [github, youtube, rss]
    focus: "AI Agent、大模型工程化与开源生态最新进展"  # AI 筛选方向
    subject_prefix: "早安 | SignalNest"

  - name: "晚间日报"
    cron: "0 21 * * *"
    content: [news]
    sources: [github, youtube, rss]
    focus: "今日科技与 AI 行业动态、产品发布与研究突破"
    subject_prefix: "晚间精选 | SignalNest"
```

`content` 可选值：

| 值 | 说明 | 对应文件 |
| --- | --- | --- |
| `news` | 抓取信息源 + 两阶段 AI 摘要 + 今日要点 | — |
| `schedule` | 今日日程（AI 解析） | `config/personal/schedule.md` |
| `todos` | 活跃项目与待办任务（AI 解析） | `config/personal/projects.md` |

`focus` 字段：每次调度的关注方向，AI 评分时以此为首要参考，留空则仅按历史偏好过滤。

---

### 第三步：配置个人助手（可选）

个人文件支持**任意 Markdown 格式**，由 AI 负责理解和提取结构化数据，无需遵循固定 YAML 格式。

<details>
<summary><code>config/personal/schedule.md</code> — 课程表与每周日程</summary>

```markdown
---
semester_start: 2025-09-09
---

## 课程表

| 课程 | 星期 | 时间 | 地点 | 周次 |
|------|------|------|------|------|
| 机器学习理论 | 周四 | 09:00-11:35 | 理科一号楼108 | 1-16 |
| 自然语言处理 | 周二 | 13:00-15:35 | 二教206 | 3-16奇 |

## 每日固定

- 07:30 晨间锻炼
- 22:30 睡前阅读 & 复盘

## 周一

- 09:00 组会 @ 理科五号楼201 // 带论文进展PPT
```

AI 会根据今天的日期和星期自动计算当前周次，只提取适用的课程和日程。
详见 `config/personal/schedule_example.md`。

</details>

<details>
<summary><code>config/personal/projects.md</code> — 项目与待办任务</summary>

```markdown
## 毕业论文

> 软截止：2026-06-30

- [x] 完成文献综述
- [ ] 完成第三章初稿 <!-- 2026-03-20 -->
- [ ] 整理实验数据 <!-- 2026-04-01 -->
- [ ] 提交导师审阅

## 课程作业

- [ ] 机器学习理论 - 第三次作业 <!-- 2026-03-07 -->
- [ ] 自然语言处理 - 阅读报告 <!-- 2026-03-14 -->
```

`[x]` 已完成的任务不会出现在日报中。`<!-- YYYY-MM-DD -->` 为软 deadline，逾期只提示"建议完成"，不强制报警。
详见 `config/personal/projects_example.md`。

</details>

---

### 第四步：启动

```bash
cd SignalNest/docker/
docker compose up -d
```

查看日志：

```bash
docker logs -f signalnest
```

<br>

## 💻 本地开发（不使用 Docker）

```bash
# 安装依赖
pip install -r requirements.txt

# 复制并填写凭证
cp docker/.env.example docker/.env

# 预览模式（打印结果，不发送通知）
python -m src.main --schedule-name "早间日报" --dry-run

# 正式运行
python -m src.main --schedule-name "早间日报"
```

<br>

## 🐳 Docker 部署

### 常用命令

```bash
# 启动（后台，按 cron 自动触发）
docker compose up -d

# 代码变更后重新构建
docker compose up -d --build

# 仅重启（config.yaml / personal/ 修改后）
docker compose restart

# 环境变量变更后重建容器
docker compose up -d --force-recreate

# 停止
docker compose down
```

### 立即触发测试

在 `docker/.env` 中设置：

```dotenv
IMMEDIATE_RUN=true          # 启动时立即执行一次
SCHEDULE_NAME=早间日报       # 留空则使用第一个 schedule
```

然后 `docker compose up -d --force-recreate`。

### 数据持久化

`data/` 通过 Docker volume 挂载到宿主机，`feedback.db`（偏好反馈历史）重建容器不丢失。
每次运行结果自动归档到 `data/history/`，按运行时间戳保存，不会覆盖历史记录。

<br>

## 🧠 内容偏好学习

每次日报运行后自动生成 `data/last_digest.json`：

```json
{
  "date": "2026-03-02",
  "source": "github",
  "title": "vllm-project/vllm",
  "ai_score": 9,
  "ai_summary": "高性能 LLM 推理引擎...",
  "user_score": null,
  "user_notes": ""
}
```

将感兴趣的条目 `user_score` 改为 1-5 整数，**下次运行时自动应用** —— AI 将参考你的历史高分内容进行过滤，推送越来越符合你的口味。

> `data/` 通过 Docker volume 挂载到宿主机，直接编辑文件即可，无需进入容器。

<br>

## ⚙️ 配置详解

### AI 设置

```yaml
ai:
  backend: "litellm"           # litellm（默认）/ claude-cli / codex-cli
  model: "openai/gpt-4o"       # LiteLLM 格式，env AI_MODEL 优先
  api_base: ""                 # 自定义端点，env AI_API_BASE 优先
  min_relevance_score: 5       # 低于此分数（1-10）的内容被过滤
  max_items_per_digest: 20     # 每次最多展示条目数
  min_items_per_source:        # 来源保底（可选）
    github: 5
    youtube: 2
  max_tokens: 2048             # 每条摘要最大 token 数
  max_workers: 10              # 第二阶段并行 AI 调用数
```

**AI 后端选项**：

| 后端 | 说明 | 是否需要 API Key |
| --- | --- | :---: |
| `litellm`（默认） | 调用任意 OpenAI 兼容云端 API | 是（`AI_API_KEY`） |
| `claude-cli` | 调用本机 `claude --print` 命令（Claude Code CLI） | 否 |
| `codex-cli` | 调用本机 `codex -q` 命令（OpenAI Codex CLI） | 否 |

通过环境变量 `AI_BACKEND` 或 `config.yaml` 中的 `ai.backend` 切换。

`min_items_per_source` 在「标题筛选阶段 + 最终出稿阶段」双重补齐。高分条目不足时优先补充该来源的低分候选；若采集阶段本身不足（如 API 失败、近期无新视频），则以实际可用条目为准。

### GitHub 采集

爬取 `github.com/trending`，由 AI 按 `focus` 方向过滤，无需手动维护关键词。

```yaml
collectors:
  github:
    enabled: true
    trending_since: "daily"       # daily / weekly / monthly
    trending_languages: []        # 留空抓所有语言，或指定如 ["python", "typescript"]
    max_repos: 25                 # 最多抓取的仓库数
```

### YouTube 采集

两路来源并行，字幕在 AI 标题筛选通过后才按需拉取，节省 API 配额。

```yaml
collectors:
  youtube:
    enabled: true                  # 需要配置 YOUTUBE_API_KEY
    # ── 路线①：订阅频道 ──────────────────────────────────────
    channel_ids:
      - "UCnUYZLuoy1rq1aVMwx4aTzw"   # Lex Fridman Podcast
      - "UCcefcZRL2oaA_uBNeo5UOWg"   # Y Combinator
    max_results_per_channel: 3    # 每个频道最终保留的视频数
    days_lookback: 7              # 只抓最近 N 天的视频
    sort_by: "views"              # "views"（热度）/ "date"（最新）
    # ── 路线②：AI 关键词搜索（其他频道）─────────────────────
    enable_keyword_search: true   # 开启后额外消耗一次 AI 调用 + YouTube Search 配额
    search_sort_by: "views"       # 排序："views"（热度）/ "date"（最新）
    max_search_results: 5         # 每个关键词最多取多少条视频
    search_days_lookback: 3       # 关键词搜索时间窗口（独立于订阅频道）
```

开启 `enable_keyword_search` 后，AI 根据当次 `focus` 自动推导 3-5 个英文搜索词，通过 YouTube Search API 覆盖订阅频道之外的内容。

### RSS 订阅源

两阶段抓取：每 feed 先多拿标题供 AI 批量筛选，入选后再精读。

```yaml
collectors:
  rss:
    enabled: true
    days_lookback: 2
    max_items_per_feed_initial: 10  # 每 feed 初始拿多少条标题（供批量筛选）
    max_items_per_feed: 3           # 每 feed 最终进入精读的文章数上限
    feeds:
      - id: "hacker-news"
        name: "Hacker News"
        url: "https://hnrss.org/frontpage"
      # 添加更多...
```

修改 `config.yaml` 后直接 `docker compose restart` 即可，无需重新构建。

### 通知渠道

```yaml
notifications:
  email:  { enabled: true }
  feishu: { enabled: true }   # 同时在 .env 配置 FEISHU_WEBHOOK_URL
  wework: { enabled: true }   # 同时在 .env 配置 WEWORK_WEBHOOK_URL
```

> **隐私保护**：`schedule` / `todos`（个人日程和项目）属于个人内容，仅发送给 `EMAIL_FROM`（发件人自己），其他收件人只收到新闻部分。

<br>

## ❓ 常见问题

### Q：邮件发送失败，提示 535 认证错误

QQ/163 邮箱需使用「授权码」而非登录密码。QQ 邮箱：设置 → 账户 → 开启 SMTP → 生成授权码。

### Q：GitHub 采集很慢或报速率限制错误

未配置 `GITHUB_TOKEN` 时每小时只有 60 次 API 请求。在 GitHub Settings → Developer Settings → Personal Access Tokens 生成 Token（无需勾选任何权限）填入 `.env`。

### Q：YouTube 未采集到内容，报 403

需在 Google Cloud Console 启用 YouTube Data API v3，并确认 API Key 没有 HTTP 引用来源限制。

### Q：开启 `enable_keyword_search` 后 YouTube 配额消耗增加

关键词搜索每次额外消耗一次 AI 调用（生成关键词）+ 若干 YouTube Search API 请求。YouTube Data API v3 每日免费配额 10,000 单位，Search 每次约 100 单位，订阅频道拉取约 1 单位。可通过降低 `max_search_results` 控制用量。

### Q：配置了来源保底，为什么仍未达到目标数量？

保底机制只在「已采集到候选」的前提下补齐。若某来源当次抓取本身不足（如 `days_lookback` 窗口内无新视频，或 API 临时失败），最终条数会小于目标值。可通过增大 `days_lookback`、增加 `channel_ids` 或查看日志排查。

### Q：如何添加新的 RSS 源

编辑 `config/config.yaml` 中的 `collectors.rss.feeds`，添加 `{id, name, url}`，然后 `docker compose restart`。

### Q：如何立即触发一次运行

在 `docker/.env` 设置 `IMMEDIATE_RUN=true` 和 `SCHEDULE_NAME=早间日报`，然后重建容器。本地开发时直接运行 `python -m src.main --schedule-name "早间日报"`。

### Q：不想用云端 API，能否本地运行 AI？

将 `AI_BACKEND=claude-cli` 或 `AI_BACKEND=codex-cli` 写入 `.env`，分别调用本机已安装的 Claude Code CLI 或 OpenAI Codex CLI，无需 API Key。

## 📚 致谢

感谢 [TrendRadar](https://github.com/sansan0/TrendRadar) 和 [obsidian-daily-digest](https://github.com/iamseeley/obsidian-daily-digest) 的启发
