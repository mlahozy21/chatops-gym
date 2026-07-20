"""Episode runner: task -> agent -> grader -> 0/1."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from gym.agents.base import Agent, Transcript
from gym.env import MattermostEnv
from gym.tools import ToolRegistry

MAX_TOOL_CALLS = 20


@dataclass
class EpisodeResult:
    task_id: str
    agent: str
    passed: bool
    reason: str
    n_calls: int
    transcript: Transcript


def load_grader(task: dict):
    module = importlib.import_module(task["grader"]["module"])
    return module.GRADER  # each grader family module exposes a GRADER instance


def run_episode(env: MattermostEnv, agent: Agent, task: dict,
                max_calls: int = MAX_TOOL_CALLS, reset: bool = True) -> EpisodeResult:
    if reset:
        env.reset(task)
    registry = ToolRegistry(env)
    transcript = Transcript()
    agent.start(task["prompt"], registry.specs())

    for _ in range(max_calls):
        call = agent.act(transcript)
        if call is None:
            break
        result = registry.call(call.tool, call.args)
        transcript.record(call, result)
        if call.tool == "finish":
            break

    grader = load_grader(task)
    verdict = grader.grade(env, task["grader"]["params"])
    return EpisodeResult(task["task_id"], agent.name, verdict.passed, verdict.reason,
                         len(transcript.steps), transcript)
