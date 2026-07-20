"""OracleAgent: executes a task's `oracle_solution` through the agent tools.

Design rules enforced here:
  * Every step goes through ToolRegistry — same user, same permissions, same
    action surface as the LLM agent. No raw-API privileges.
  * References to world entities are declared as `resolve` specs and re-found
    through the tools (read/search/pinned), never taken from setup knowledge.
  * The marker registry is used ONLY as a cross-check: if the tool-based
    resolution finds a different entity than the marker says, the oracle fails
    loudly — that's a task-design bug the validation harness must surface.

Resolver kinds:
  {"resolve": "post_in_channel", "channel": c, "must_contain": s, "expect_marker": m?}
  {"resolve": "pinned_post",     "channel": c, "must_contain": s, "expect_marker": m?}
"""
from __future__ import annotations

import copy

from gym.agents.base import Agent, ToolCall, Transcript
from gym.env import MattermostEnv
from gym.tools import ToolRegistry


class OracleError(RuntimeError):
    pass


class OracleAgent(Agent):
    name = "oracle"

    def __init__(self, env: MattermostEnv, registry: ToolRegistry, solution: list[dict]):
        self.env = env
        self.registry = registry
        self.solution = copy.deepcopy(solution)
        self._i = 0

    def act(self, transcript: Transcript) -> ToolCall | None:
        if self._i >= len(self.solution):
            return None
        step = self.solution[self._i]
        self._i += 1
        args = {k: self._resolve(v) for k, v in step.get("args", {}).items()}
        return ToolCall(step["tool"], args)

    # ---------------------------------------------------------------- resolve
    def _resolve(self, value):
        if not (isinstance(value, dict) and "resolve" in value):
            return value
        kind = value["resolve"]
        if kind == "post_in_channel":
            found = self._find_in_channel(value["channel"], value["must_contain"])
        elif kind == "pinned_post":
            found = self._find_pinned(value["channel"], value["must_contain"])
        else:
            raise OracleError(f"unknown resolver: {kind}")
        if found is None:
            raise OracleError(f"resolver {kind} found nothing for {value!r}")
        # Cross-check against the marker registry (validation-only guard).
        expect = value.get("expect_marker")
        if expect:
            marked = self.env.resolve_marker(expect)
            if marked["id"] != found:
                raise OracleError(
                    f"resolver {kind} found post {found} but marker '{expect}' "
                    f"is {marked['id']} — task design bug (weak anchor).")
        return found

    def _find_in_channel(self, channel: str, needle: str) -> str | None:
        for page in range(10):
            res = self.registry.call("read_channel", {"channel": channel, "page": page})
            if "error" in res:
                raise OracleError(res["error"])
            posts = res["posts"]
            for p in posts:
                if needle.lower() in p["message"].lower():
                    return p["post_id"]
            if not posts:
                return None
        return None

    def _find_pinned(self, channel: str, needle: str) -> str | None:
        res = self.registry.call("get_pinned_messages", {"channel": channel})
        if "error" in res:
            raise OracleError(res["error"])
        for p in res["pinned"]:
            if needle.lower() in p["message"].lower():
                return p["post_id"]
        return None
