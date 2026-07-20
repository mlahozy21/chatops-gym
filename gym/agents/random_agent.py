"""RandomAgent: k syntactically valid but random tool calls.

Deliberately a weak test (with ~12 tools and 20 calls it will essentially
never solve an L3 task) — it exists to catch graders satisfied by *any*
mutation of the world, which NullAgent cannot detect. The heavy lifting of
false-positive detection is mutation testing (validation/mutations.py).
"""
from __future__ import annotations

import random

from gym.agents.base import Agent, ToolCall, Transcript
from gym.world import CHANNELS, USERNAMES

_WORDS = ["sync", "update", "ping", "done", "check", "review", "notes"]


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int, k: int = 8):
        self.rng = random.Random(seed)
        self.k = k
        self._i = 0

    def act(self, transcript: Transcript) -> ToolCall | None:
        if self._i >= self.k:
            return None
        self._i += 1
        rng = self.rng
        channel = rng.choice([c[0] for c in CHANNELS])
        choice = rng.random()
        if choice < 0.35:
            return ToolCall("send_message",
                            {"channel": channel, "text": " ".join(rng.sample(_WORDS, 3))})
        if choice < 0.55:
            return ToolCall("read_channel", {"channel": channel})
        if choice < 0.70:
            return ToolCall("set_channel_header",
                            {"channel": channel, "header": rng.choice(_WORDS)})
        if choice < 0.85:
            return ToolCall("create_channel",
                            {"name": f"rand-{rng.randrange(10_000)}"})
        return ToolCall("invite_user",
                        {"channel": channel, "username": rng.choice(USERNAMES)})
