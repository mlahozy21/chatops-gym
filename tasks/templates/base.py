"""Template contract.

A template is a pure function seed -> task instance. Prompt, world overlays,
grader params, oracle solution and (in validation/mutations.py) the mutants
are all derived from the SAME sampled parameters — consistent by construction.
Each template also exposes continuous difficulty knobs; they are the answer to
curriculum saturation: regenerate harder without designing new families.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TaskTemplate(ABC):
    template_id: str
    difficulty: str  # structural prior: L1..L4 (empirical calibration is per-model)

    @abstractmethod
    def generate(self, seed: int, **knobs) -> dict:
        ...

    def instance_id(self, seed: int) -> str:
        return f"{self.template_id}.{self.difficulty}.{seed:04d}"
