"""Deterministic world generation.

A `world_seed` fully determines a plausible organization: users, channels and
a body of background chatter that acts as natural distractor material for
retrieval tasks. Tasks add their own deltas on top via overlays (see env.py).

Determinism contract: same seed -> same usernames, same channels, same history
texts in the same order. Entity *IDs* are NOT stable across resets (Mattermost
assigns them); everything downstream anchors on stable keys (names, marker
texts), never on IDs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from faker import Faker

# Stable roster: prompts may reference these by @username, so they are fixed,
# not sampled. Display names/emails are faked deterministically.
USERNAMES = [
    "diana", "arjun", "sofia", "mateo", "yuki", "amara",
    "liam", "priya", "carlos", "ingrid", "tomas", "zara",
]

CHANNELS = [
    ("q3-planning", "Q3 Planning"),
    ("finance", "Finance"),
    ("support", "Support"),
    ("eng-oncall", "Eng Oncall"),
    ("marketing", "Marketing"),
    ("product", "Product"),
    ("random", "Random"),
    ("hr", "HR"),
]

_CHATTER = [
    "Has anyone seen the latest {thing} numbers?",
    "I'll push the {thing} update after lunch.",
    "Reminder: {thing} review is on {weekday}.",
    "Can someone take a look at the {thing} ticket?",
    "The {thing} dashboard looks off to me.",
    "Shipped the fix for the {thing} issue.",
    "Let's sync on {thing} tomorrow morning.",
    "Draft for the {thing} doc is ready for comments.",
    "Anyone else seeing slowness in {thing} today?",
    "Great work on the {thing} launch, team!",
    "I'm OOO on {weekday}, ping {user} for {thing}.",
    "Moved the {thing} meeting to {weekday}.",
]

_THINGS = [
    "onboarding", "billing", "signup", "retention", "churn", "deploy",
    "roadmap", "OKR", "pricing", "newsletter", "hiring", "analytics",
    "mobile", "API", "search", "backup",
]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class HistoryPost:
    channel: str          # channel name (stable key)
    username: str
    text: str
    thread_parent: int | None = None  # index into the history list, or None


@dataclass
class WorldSpec:
    seed: int
    usernames: list[str] = field(default_factory=lambda: list(USERNAMES))
    channels: list[tuple[str, str]] = field(default_factory=lambda: list(CHANNELS))
    history: list[HistoryPost] = field(default_factory=list)


def build_world(seed: int, n_history: int = 60, reply_rate: float = 0.2) -> WorldSpec:
    """Build the deterministic world spec (no API calls here)."""
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    world = WorldSpec(seed=seed)
    channel_names = [c[0] for c in world.channels]

    for i in range(n_history):
        template = rng.choice(_CHATTER)
        text = template.format(
            thing=rng.choice(_THINGS),
            weekday=rng.choice(_WEEKDAYS),
            user="@" + rng.choice(world.usernames),
        )
        channel = rng.choice(channel_names)
        username = rng.choice(world.usernames)
        parent: int | None = None
        if i > 3 and rng.random() < reply_rate:
            # Reply to an earlier root post in the same channel, if any.
            candidates = [
                j for j, p in enumerate(world.history)
                if p.channel == channel and p.thread_parent is None
            ]
            if candidates:
                parent = rng.choice(candidates)
                text = rng.choice([
                    "+1", "Agreed.", "On it.", "Good catch, thanks!",
                    f"cc @{rng.choice(world.usernames)}", "Will do.",
                ])
        world.history.append(HistoryPost(channel, username, text, parent))

    # Give Faker a use so linters don't flag it and future templates can use
    # fake profiles; deterministic because Faker.seed was called.
    _ = fake.name()
    return world
