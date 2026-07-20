"""NullAgent: does nothing. Any task it passes has a grader satisfied by the
initial world state — a broken grader. Cheap smoke test, required 0."""
from __future__ import annotations

from gym.agents.base import Agent, ToolCall, Transcript


class NullAgent(Agent):
    name = "null"

    def act(self, transcript: Transcript) -> ToolCall | None:
        return None
