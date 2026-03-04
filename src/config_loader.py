"""
config_loader.py - 加载 config.yaml + .env，返回统一配置字典
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 容器内路径 / 本地开发路径
_BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path(__file__).parent))
_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", _BASE_DIR.parent / "config" / "config.yaml"))
_ENV_PATH = _BASE_DIR.parent / ".env"


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_agent_config(cfg: dict) -> None:
    agent = cfg.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("config.agent 缺失或类型错误，必须为对象")

    policy = agent.get("policy")
    missing: list[str] = []

    def _require(path: str, value: Any) -> None:
        if value is None:
            missing.append(path)

    _require("agent.max_steps", agent.get("max_steps"))
    _require("agent.schedule_max_steps", agent.get("schedule_max_steps"))
    _require("agent.max_steps_hard_limit", agent.get("max_steps_hard_limit"))
    _require("agent.schedule_allow_side_effects", agent.get("schedule_allow_side_effects"))
    _require("agent.recent_turns_context_limit", agent.get("recent_turns_context_limit"))
    _require("agent.require_dispatch_tool_call", agent.get("require_dispatch_tool_call"))
    _require("agent.fallback_response_max_tokens", agent.get("fallback_response_max_tokens"))
    _require("agent.session_title_template", agent.get("session_title_template"))

    if not isinstance(policy, dict):
        missing.extend(
            [
                "agent.policy.allow_tools",
                "agent.policy.deny_tools",
                "agent.policy.allow_side_effects",
            ]
        )
    else:
        _require("agent.policy.allow_tools", policy.get("allow_tools"))
        _require("agent.policy.deny_tools", policy.get("deny_tools"))
        _require("agent.policy.allow_side_effects", policy.get("allow_side_effects"))

    if missing:
        raise ValueError(
            "config.agent 缺少必填项: " + ", ".join(sorted(set(missing)))
        )

    if not _is_positive_int(agent["max_steps"]):
        raise ValueError("agent.max_steps 必须为正整数")
    if not _is_positive_int(agent["schedule_max_steps"]):
        raise ValueError("agent.schedule_max_steps 必须为正整数")
    if not _is_positive_int(agent["max_steps_hard_limit"]):
        raise ValueError("agent.max_steps_hard_limit 必须为正整数")
    if not _is_positive_int(agent["recent_turns_context_limit"]):
        raise ValueError("agent.recent_turns_context_limit 必须为正整数")
    if not _is_positive_int(agent["fallback_response_max_tokens"]):
        raise ValueError("agent.fallback_response_max_tokens 必须为正整数")
    if not _is_bool(agent["schedule_allow_side_effects"]):
        raise ValueError("agent.schedule_allow_side_effects 必须为布尔值")
    if not _is_bool(agent["require_dispatch_tool_call"]):
        raise ValueError("agent.require_dispatch_tool_call 必须为布尔值")
    if not isinstance(agent["session_title_template"], str) or not agent[
        "session_title_template"
    ].strip():
        raise ValueError("agent.session_title_template 必须为非空字符串")

    policy_cfg = agent["policy"]
    if not _is_str_list(policy_cfg["allow_tools"]):
        raise ValueError("agent.policy.allow_tools 必须为字符串数组")
    if not _is_str_list(policy_cfg["deny_tools"]):
        raise ValueError("agent.policy.deny_tools 必须为字符串数组")
    if not _is_bool(policy_cfg["allow_side_effects"]):
        raise ValueError("agent.policy.allow_side_effects 必须为布尔值")


def load_config() -> dict:
    """
    加载并合并 config.yaml + .env，返回 AppConfig dict。
    .env 文件仅在本地开发时使用，Docker 中由 docker-compose 注入 env vars。
    """
    load_dotenv(_ENV_PATH, override=False)

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"配置文件未找到: {_CONFIG_PATH}")

    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError("config.yaml 为空或格式错误")

    # 确保必要的顶层 key 存在
    cfg.setdefault("app", {})
    cfg.setdefault("schedules", [])
    cfg.setdefault("collectors", {})
    cfg.setdefault("ai", {})
    cfg.setdefault("agent", {})
    cfg.setdefault("notifications", {})
    cfg.setdefault("storage", {})

    _validate_agent_config(cfg)

    # 注入 storage data_dir（容器内固定路径或本地 data/）
    if not cfg["storage"].get("data_dir"):
        cfg["storage"]["data_dir"] = str(_BASE_DIR.parent / "data")

    # personal YAML 文件路径
    cfg["_personal_dir"] = str(_CONFIG_PATH.parent / "personal")

    return cfg
