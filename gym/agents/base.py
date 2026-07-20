"""Agent interface: everything that acts in an episode implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    tool: str
    args: dict


@dataclass
class Transcript:
    steps: list[dict] = field(default_factory=list)

    def record(self, call: ToolCall, result: dict) -> None:
        self.steps.append({"tool": call.tool, "args": call.args, "result": result})


class Agent(ABC):
    """An agent observes the task prompt + its own transcript and emits the
    next tool call. Returning a `finish` call (or None) ends the episode."""

    name = "agent"

    def start(self, prompt: str, tool_specs: list[dict]) -> None:
        """Called once at episode start."""

    @abstractmethod
    def act(self, transcript: Transcript) -> ToolCall | None:
        ...
