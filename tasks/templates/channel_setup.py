"""Template: channel_setup (L1 single action / L2 chained actions).

L1: create a public channel with an explicitly named handle.
L2: create it, set its header, and invite a named user — 3 chained actions,
still no retrieval.

Failure mode targeted: instruction-following slips on exact entity names
(channel handle format, right username) and dropped steps in short chains.
"""
from __future__ import annotations

import random

from tasks.templates.base import TaskTemplate

_WORDS_A = ["q3", "launch", "billing", "growth", "infra", "brand", "sales", "beta"]
_WORDS_B = ["retro", "budget", "planning", "triage", "review", "sync", "war-room", "ideas"]
_USERS = ["diana", "arjun", "sofia", "mateo", "yuki", "amara", "liam", "priya"]
_HEADERS = [
    "Weekly sync notes and action items",
    "All updates for the {a} {b} workstream",
    "Coordination space — check pinned items first",
]


class ChannelSetupTemplate(TaskTemplate):
    template_id = "channel_setup"

    def __init__(self, level: str = "L1"):
        assert level in ("L1", "L2")
        self.difficulty = level

    def generate(self, seed: int, n_history: int = 40) -> dict:
        rng = random.Random(seed * 6271 + 29)
        a, b = rng.choice(_WORDS_A), rng.choice(_WORDS_B)
        name = f"{a}-{b}"
        member = rng.choice(_USERS)
        header = rng.choice(_HEADERS).format(a=a, b=b)

        if self.difficulty == "L1":
            prompt = f"Create a new public channel named #{name}."
            params = {"channel_name": name}
            oracle = [
                {"tool": "create_channel", "args": {"name": name, "display_name": name}},
                {"tool": "finish", "args": {"summary": f"created #{name}"}},
            ]
        else:
            prompt = (f"Set up a new public channel #{name} for the {a} {b} workstream: "
                      f"its header should say \"{header}\" and @{member} should be added to it.")
            params = {"channel_name": name,
                      "header": {"type": "text", "value": header},
                      "member": member}
            oracle = [
                {"tool": "create_channel", "args": {"name": name, "display_name": name}},
                {"tool": "set_channel_header", "args": {"channel": name, "header": header}},
                {"tool": "invite_user", "args": {"channel": name, "username": member}},
                {"tool": "finish", "args": {"summary": f"set up #{name}"}},
            ]

        return {
            "task_id": self.instance_id(seed),
            "template_id": self.template_id,
            "seed": seed,
            "difficulty": self.difficulty,
            "category": "entity_actions",
            "failure_mode": "entity-name slips; dropped steps in action chains",
            "prompt": prompt,
            "setup": {"world_seed": seed, "n_history": n_history, "overlays": []},
            "grader": {"module": "graders.families.channel_setup", "params": params},
            "oracle_solution": oracle,
            "meta": {"name": name, "member": member, "header": header},
        }


TEMPLATE_L1 = ChannelSetupTemplate("L1")
TEMPLATE_L2 = ChannelSetupTemplate("L2")
