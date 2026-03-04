#!/bin/bash
set -e

# 检查配置文件
if [ ! -f "/app/config/config.yaml" ]; then
    echo "❌ 配置文件缺失: /app/config/config.yaml"
    echo "   请确认 volume 挂载正确：- ./config:/app/config:ro"
    exit 1
fi

# 保存环境变量，确保 cron 任务可以继承
env >> /etc/environment

SCHEDULE_ARG="--schedule-name"
echo "🤖 调度执行器: agent-only"

case "${RUN_MODE:-cron}" in

"once")
    echo "🔄 单次执行模式: ${SCHEDULE_NAME:-（使用第一个 schedule）}"
    cd /app && exec python -m src.main ${SCHEDULE_ARG} "${SCHEDULE_NAME:-}"
    ;;

"cron")
    echo "⏰ 定时模式：从 config.yaml 解析调度计划..."

    # 从 config.yaml 读取 schedules 列表，生成 crontab
    python3 -c "
import sys, yaml

try:
    with open('/app/config/config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    print(f'ERROR: 读取 config.yaml 失败: {e}', file=sys.stderr)
    sys.exit(1)

schedules = cfg.get('schedules', [])
if not schedules:
    print('ERROR: config.yaml 中没有定义任何 schedules', file=sys.stderr)
    sys.exit(1)

arg = '${SCHEDULE_ARG}'
for s in schedules:
    cron = s.get('cron', '').strip()
    name = s.get('name', '').strip()
    if not cron or not name:
        continue
    # 单引号包裹 name，避免空格问题
    print(f\"{cron} cd /app && python -m src.main {arg} '{name}'\")
" > /tmp/crontab

    if [ $? -ne 0 ]; then
        echo "❌ crontab 生成失败"
        exit 1
    fi

    echo "📅 生成的 crontab："
    cat /tmp/crontab

    if ! supercronic -test /tmp/crontab; then
        echo "❌ crontab 格式验证失败"
        exit 1
    fi

    # 立即执行一次（如果配置了）
    if [ "${IMMEDIATE_RUN:-false}" = "true" ]; then
        echo "▶ 立即执行一次: ${SCHEDULE_NAME:-（使用第一个 schedule）}"
        cd /app && python -m src.main ${SCHEDULE_ARG} "${SCHEDULE_NAME:-}" || true
    fi

    echo "🚀 启动 supercronic..."
    exec supercronic -passthrough-logs /tmp/crontab
    ;;

*)
    exec "$@"
    ;;
esac
