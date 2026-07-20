"""Evaluate an LLM agent over the curriculum; write results + transcripts.

Transcripts of failures are first-class outputs: they are the empirical
evidence behind each family's `failure_mode` (see analysis/transcripts/).

Usage:
    python -m scripts.run_eval --backend ollama:qwen2.5:7b [--limit N]
                               [--out results/qwen25.jsonl]
"""
from __future__ import annotations

import argparse
import json
import pathlib

from gym.agents.llm import LLMAgent, make_backend
from gym.env import MattermostEnv
from gym.runner import run_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="ollama:MODEL | anthropic:MODEL | openai:MODEL")
    ap.add_argument("--tasks", default="tasks/instances")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default="*", help="glob over task ids, e.g. '*.01*'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--transcripts", default="analysis/transcripts")
    args = ap.parse_args()

    files = sorted(pathlib.Path(args.tasks).glob(f"{args.only}.json"))
    if args.limit:
        files = files[: args.limit]

    def slug(s: str) -> str:
        return s.replace(":", "_").replace("/", "_")

    out_path = pathlib.Path(args.out or f"results/{slug(args.backend)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tdir = pathlib.Path(args.transcripts)
    tdir.mkdir(parents=True, exist_ok=True)

    env = MattermostEnv()
    backend = make_backend(args.backend)  # build once; local backends load a model
    # Append, don't clobber: batches accumulate; the report dedupes by task_id
    # keeping the most recent episode.
    with out_path.open("a") as fh:
        for f in files:
            task = json.loads(f.read_text())
            agent = LLMAgent(backend, name=args.backend)
            result = run_episode(env, agent, task)
            row = {"task_id": task["task_id"], "template": task["template_id"],
                   "difficulty": task["difficulty"], "backend": args.backend,
                   "passed": result.passed, "reason": result.reason,
                   "n_calls": result.n_calls}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            mark = "PASS" if result.passed else "FAIL"
            print(f"[{mark}] {task['task_id']}: {result.reason} ({result.n_calls} calls)")
            if not result.passed:
                (tdir / f"{task['task_id']}.{slug(agent.name)}.json").write_text(
                    json.dumps({"task": task["task_id"], "prompt": task["prompt"],
                                "verdict": result.reason,
                                "steps": result.transcript.steps}, indent=2))
    print(f"\nresults -> {out_path}; failure transcripts -> {tdir}/")


if __name__ == "__main__":
    main()
