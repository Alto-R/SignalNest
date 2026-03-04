"""
Minimal tool policy layer for local agent runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ToolLike(Protocol):
    name: str
    side_effect: bool


@dataclass(frozen=True)
class ToolPolicy:
    """
    Minimal policy:
      - allowlist / denylist
      - side-effect gate
    """

    allow_tools: set[str] | None
    deny_tools: set[str]
    allow_side_effects: bool

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "ToolPolicy":
        policy_cfg = config["agent"]["policy"]

        allow_tools_raw = policy_cfg["allow_tools"]
        deny_tools_raw = policy_cfg["deny_tools"]
        allow_side_effects = bool(policy_cfg["allow_side_effects"])

        allow_tools = {str(x).strip() for x in allow_tools_raw if str(x).strip()}
        deny_tools = {str(x).strip() for x in deny_tools_raw if str(x).strip()}

        return cls(
            allow_tools=allow_tools if allow_tools else None,
            deny_tools=deny_tools,
            allow_side_effects=allow_side_effects,
        )

    def check(self, tool: ToolLike) -> tuple[bool, str | None]:
        if tool.name in self.deny_tools:
            return False, f"tool '{tool.name}' is denied by policy"

        if self.allow_tools is not None and tool.name not in self.allow_tools:
            return False, f"tool '{tool.name}' is not in allowlist"

        if tool.side_effect and not self.allow_side_effects:
            return (
                False,
                f"tool '{tool.name}' has side effects and allow_side_effects=false",
            )

        return True, None
