"""Curriculum validation harness. No task ships without passing this.

Per-instance contract:
  oracle (through agent tools)  -> 1   task is solvable with the agent's action surface
  every mutant                  -> 0   grader rejects plausible near-misses
  null agent                    -> 0   grader not satisfied by initial state
  random agent (n seeds)        -> 0   grader not satisfied by arbitrary world mutations

Runs in CI on every PR (see .github/workflows/validate.yml).

Usage:
    python -m validation.validate_curriculum [--tasks tasks/instances] [--limit N]
                                             [--random-seeds 2] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from gym.agents.null import NullAgent
from gym.agents.oracle import OracleAgent, OracleError
from gym.agents.random_agent import RandomAgent
from gym.env import MattermostEnv
from gym.runner import run_episode
from gym.tools import ToolRegistry
from validation.mutations import mutants_for


def load_tasks(path: pathlib.Path, limit: int | None, only: str = "*") -> list[dict]:
    files = sorted(path.glob(f"{only}.json"))
    if limit:
        files = files[:limit]
    return [json.loads(f.read_text()) for f in files]


def check(condition_name: str, task: dict, expected: bool, env: MattermostEnv,
          agent_factory, failures: list[dict]) -> bool:
    env.reset(task)
    registry = ToolRegistry(env)
    try:
        agent = agent_factory(env, registry)
        result = run_episode(env, agent, task, reset=False)
        passed, reason = result.passed, result.reason
    except OracleError as e:
        passed, reason = False, f"oracle error: {e}"
    ok = passed == expected
    if not ok:
        failures.append({"task": task["task_id"], "check": condition_name,
                         "expected": int(expected), "got": int(passed), "reason": reason})
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {condition_name:32s} expected={int(expected)} got={int(passed)} ({reason})")
    return ok


def validate_task(env: MattermostEnv, task: dict, random_seeds: int,
                  failures: list[dict]) -> None:
    print(f"\n=== {task['task_id']} ({task['difficulty']}) ===")
    check("oracle_through_tools", task, True, env,
          lambda e, r: OracleAgent(e, r, task["oracle_solution"]), failures)
    for m in mutants_for(task):
        check(f"mutant:{m.name}", task, False, env,
              lambda e, r, m=m: OracleAgent(e, r, m.solution), failures)
    check("null_agent", task, False, env, lambda e, r: NullAgent(), failures)
    for s in range(random_seeds):
        check(f"random_agent[{s}]", task, False, env,
              lambda e, r, s=s: RandomAgent(seed=task["seed"] * 100 + s), failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/instances")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default="*",
                    help="glob over task ids, e.g. 'thread_reply.L3.01*' to "
                         "validate just a new batch")
    ap.add_argument("--random-seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = load_tasks(pathlib.Path(args.tasks), args.limit, args.only)
    if not tasks:
        print("no task instances found — run scripts/generate_tasks.py first", file=sys.stderr)
        return 2

    env = MattermostEnv()
    failures: list[dict] = []
    t0 = time.time()
    for task in tasks:
        validate_task(env, task, args.random_seeds, failures)

    n_checks = sum(1 for _ in tasks) and None  # readability only
    print(f"\n{'=' * 60}")
    print(f"validated {len(tasks)} tasks in {time.time() - t0:.0f}s — "
          f"{len(failures)} failing checks")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"tasks": len(tasks), "failures": failures}, indent=2))
    if failures:
        print(json.dumps(failures, indent=2))
        return 1
    print("curriculum is valid: every task solvable via agent tools, "
          "every grader rejects all near-miss mutants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
