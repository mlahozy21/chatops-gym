"""Grader contract.

Graders verify FINAL STATE, not trajectory, and always return a binary verdict
with a human-readable reason (debugging graders at scale without reasons is
impossible). They are trusted infrastructure: unlike the oracle, they may use
the env's marker registry and admin read access directly.

Known limits of this grading model (documented, not hidden): ephemeral actions,
pure-read tasks, "don't do X" prohibitions that leave no trace in the diff, and
destroy-then-restore trajectories that land on a correct final state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gym.env import MattermostEnv


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


class Grader(ABC):
    @abstractmethod
    def grade(self, env: MattermostEnv, params: dict) -> Verdict:
        ...
