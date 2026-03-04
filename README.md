<div align="center">

# SignalNest 📡

一个自托管的个人 AI 日报系统：定时聚合 GitHub Trending / YouTube / RSS，经两阶段 AI 筛选与摘要后，通过 Email / 飞书 / 企业微信推送。

[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-快速开始)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Scheduler](https://img.shields.io/badge/Scheduler-supercronic-orange?style=flat-square)](https://github.com/aptible/supercronic)
[![Mode](https://img.shields.io/badge/Executor-agent%20(default)-00A86B?style=flat-square)](#-agent-模式)

**[中文](README.md)** | **[English](README-EN.md)**

</div>

<br>

## 📑 快速导航

<div align="center">

| | | |
|:---:|:---:|:---:|
| [🚀 快速开始](#-快速开始) | [⚙️ 配置说明](#️-配置说明) | [🤖 Agent 模式](#-agent-模式) |
| [💻 本地开发](#-本地开发不使用-docker) | [🗂️ 数据与偏好学习](#️-数据与偏好学习) | [❓ 常见问题](#-常见问题) |

</div>

<br>

## ✨ 核心特性

- 多源采集：`github` / `youtube` / `rss` 按 schedule 自由组合。
- 两阶段 AI：先批量标题筛选，再对入选条目逐条评分与摘要。
- 历史去重：自动注入 `data/history/` 最近 7 天标题，降低重复推送。
- 来源保底：`ai.min_items_per_source` 支持来源最小条数（默认 GitHub>=5、YouTube>=2）。
- YouTube 双通路：订阅频道 + 基于 `focus` 的关键词搜索（可选）。
- 个人助手：解析 `config/personal/schedule.md` 和 `projects.md`，并支持 `schedule-<name>.md` / `projects-<name>.md` 按收件人定制。
- 偏好学习：通过 `data/last_digest.json` 的 `user_score` 写入 `feedback.db`，形成个性化 few-shot。
- 多渠道通知：邮件 HTML 模板 + 飞书 webhook + 企业微信 webhook。
- 隐私分发：`EMAIL_FROM` 保持默认个人内容；其他收件人若存在同名专属文件则发送定制版，否则仅发送新闻模块。
- Agent-only 执行链路：仅保留本地 agent 调度路径。

## 🏗️ 架构总览

```text
Docker entrypoint (supercronic)
  -> python -m src.main --schedule-name <name>

main.py (agent-only)
  -> agent.kernel            (LLM planning + tool calls + session persistence)
  -> agent.tools             (collect/summarize/payload/dispatch tools)
  -> ai.feedback + data/history + data/last_digest.json
```

## 🚀 快速开始

### 1) 配置 Docker 环境变量

```bash
cd SignalNest/docker
cp .env.example .env
```

编辑 `docker/.env`，至少填写：

- `AI_API_KEY`、`AI_MODEL`（当 `AI_BACKEND=litellm`）
- `EMAIL_FROM`、`EMAIL_PASSWORD`、`EMAIL_TO`（如需按人定制，`EMAIL_TO` 使用 `姓名:邮箱` 格式）

常用可选项：

- `GITHUB_TOKEN`
- `YOUTUBE_API_KEY`
- `FEISHU_WEBHOOK_URL`
- `WEWORK_WEBHOOK_URL`
- `EMAIL_OPENING_AI_NAMES`（开场句启用名单，默认 `yy`）
- `EMAIL_OPENING_YY`（`yy` 的手写开场句；手写优先，未配置时才走 AI 生成）

### 2) 准备个人文件（可选）

```bash
cd ..
cp config/personal/schedule_example.md config/personal/schedule.md
cp config/personal/projects_example.md config/personal/projects.md

# 可选：按收件人姓名准备专属文件（需与 EMAIL_TO 中姓名精确匹配）
cp config/personal/schedule_example.md config/personal/schedule-yy.md
cp config/personal/projects_example.md config/personal/projects-yy.md
```

`schedule.md` / `projects.md` 支持自由 Markdown，结构由 AI 解析，不要求固定 schema。  
若 `EMAIL_TO` 包含 `yy:foo@example.com`，系统会尝试读取 `schedule-yy.md` 与 `projects-yy.md`，并在发给 `yy` 的邮件中注入可用的个人模块（有啥发啥）。

### 3) 配置调度

编辑 `config/config.yaml`，重点字段：

- `schedules[].name`：调度名
- `schedules[].cron`：cron 表达式
- `schedules[].content`：`news` / `schedule` / `todos`
- `schedules[].sources`：`github` / `youtube` / `rss`
- `schedules[].focus`：本次关注方向（AI 打分优先级）
- `schedules[].subject_prefix`：通知标题前缀

### 4) 启动

```bash
cd docker
docker compose up -d --build
```

查看日志：

```bash
docker logs -f signalnest
```

### 5) 立即触发一次（可选）

设置 `docker/.env`：

```dotenv
IMMEDIATE_RUN=true
SCHEDULE_NAME=早间日报
```

然后执行：

```bash
docker compose up -d --force-recreate
```

## 💻 本地开发（不使用 Docker）

安装依赖：

```bash
pip install -r requirements.txt
```

准备本地环境变量：

```bash
cp docker/.env.example .env
```

本地运行时，`src/config_loader.py` 读取的是仓库根目录 `.env`（不是 `docker/.env`）。

### Agent-only 流程

```bash
python -m src.main --schedule-name "早间日报" --dry-run
python -m src.main --schedule-name "早间日报"
```

## ⚙️ 配置说明

### 关键配置块

| 配置路径 | 作用 |
| --- | --- |
| `app.timezone` / `app.language` | 全局时区与输出语言 |
| `schedules[]` | 定时计划与内容组合 |
| `collectors.github` | GitHub Trending 抓取参数 |
| `collectors.youtube` | 订阅频道 + 关键词搜索参数 |
| `collectors.rss` | RSS 抓取窗口与每 feed 限制 |
| `ai.*` | 后端、模型、阈值、并发和来源保底 |
| `agent.*` | Agent 步数与策略（含 `max_steps_hard_limit`） |
| `notifications.*` | 各通知渠道启用开关 |
| `storage.*` | 数据目录与任务前瞻天数 |

### AI 后端

| 后端 | 说明 | 需要 API Key |
| --- | --- | :---: |
| `litellm`（默认） | 调用 OpenAI 兼容云端 API | 是（`AI_API_KEY`） |
| `claude-cli` | 调用本机 `claude --print` | 否 |
| `codex-cli` | 调用本机 `codex -q` | 否 |

环境变量优先级高于 `config.yaml`：

- `AI_BACKEND`
- `AI_MODEL`
- `AI_API_BASE`
- `AI_API_KEY`

### Docker 执行模式

`docker/entrypoint.sh` 支持：

- `RUN_MODE=cron`（默认）：解析 `config.yaml` 生成 crontab，由 supercronic 持续触发。
- `RUN_MODE=once`：只执行一次后退出。

调度命令固定为：`python -m src.main --schedule-name <name>`（agent-only）。

## 🤖 Agent 模式

Agent 核心位于 `src/agent/`，流程是“LLM 规划 -> 工具调用 -> 状态持久化”。

### 内置工具

- `collect_github`
- `collect_rss`
- `collect_youtube`
- `collect_all_news`
- `summarize_news`
- `read_today_schedule`
- `read_active_projects`
- `build_digest_payload`
- `dispatch_notifications`（副作用工具）

### 策略与持久化

- 工具策略：allowlist / denylist / `allow_side_effects`（统一由 `config.agent.policy` 驱动）
- 步数配置：`agent.max_steps` / `agent.schedule_max_steps`，并受 `agent.max_steps_hard_limit` 统一裁剪
- 调度副作用开关：`agent.schedule_allow_side_effects`（控制 `--schedule-name` 是否允许真实发送）
- 上下文回看窗口：`agent.recent_turns_context_limit`（注入最近 N 轮摘要到规划提示词）
- 会话库：`data/agent_sessions.db`
- 定时调度运行会自动持久化（固定启用，不提供开关）
- 配置收敛：agent 运行参数统一在 `config.agent`，启动时执行严格必填校验

## 🗂️ 数据与偏好学习

运行后核心文件：

- `data/last_digest.json`：本次条目与可编辑 `user_score`
- `data/history/digest_*.json`：历史归档
- `data/feedback.db`：用户反馈数据库
- `data/agent_sessions.db`：Agent 会话与工具轨迹

反馈流程：

1. 编辑 `data/last_digest.json`，将感兴趣条目的 `user_score` 填为 `1-5`。
2. 下一次运行自动写入 `feedback.db`。
3. 后续摘要会把高分历史作为 taste examples 参与打分。

## ❓ 常见问题

### 邮件报 535 认证失败

请使用 SMTP 授权码，不要使用邮箱登录密码（QQ/163 尤其如此）。

### 本地运行读不到环境变量

检查是否在仓库根目录放置 `.env`。本地模式不会读取 `docker/.env`。

### YouTube 没抓到内容

检查：

- `collectors.youtube.enabled=true`
- `YOUTUBE_API_KEY` 已配置
- `days_lookback` / `search_days_lookback` 是否太短

### 使用 `claude-cli` / `codex-cli` 后关键词搜索未生效

当前 `youtube_collector._ai_generate_keywords()` 固定使用 LiteLLM + `AI_API_KEY`。如果没有 `AI_API_KEY`，会跳过关键词搜索，但订阅频道采集仍然可用。

### 开启了通知但任务仍失败

`dispatch()` 默认要求至少一个启用渠道发送成功，否则会抛错。请检查 SMTP/Webhook 可用性。

## 🔐 安全建议

- 不要提交 `docker/.env` 或根目录 `.env`。
- `config/personal/schedule.md`、`projects.md` 及 `schedule-*.md`、`projects-*.md` 可能包含个人信息，建议仅本地保存。
- 凭据泄露后请立即轮换 key/password。

## 📚 致谢

灵感来源：

- [TrendRadar](https://github.com/sansan0/TrendRadar)
- [obsidian-daily-digest](https://github.com/Lantern567/obsidian-daily-digest)

