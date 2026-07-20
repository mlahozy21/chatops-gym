"""Generate task instances from templates.

Difficulty knobs are the answer to curriculum saturation: when a model passes
everything (the calibration report flags SATURATED), regenerate harder with
--hard instead of designing new families. Use --seed-start to give the harder
batch its own id range so it coexists with the base batch.

Usage:
    python -m scripts.generate_tasks [--per-template 5] [--out tasks/instances]
                                     [--hard] [--seed-start 1]
"""
from __future__ import annotations

import argparse
import json
import pathlib

from tasks.templates.channel_setup import TEMPLATE_L1, TEMPLATE_L2
from tasks.templates.rollup_reply import TEMPLATE as ROLLUP_REPLY
from tasks.templates.thread_reply import TEMPLATE as THREAD_REPLY

TEMPLATES = [TEMPLATE_L1, TEMPLATE_L2, THREAD_REPLY, ROLLUP_REPLY]

# Knob presets per template_id. The base profile is each template's defaults.
HARD_KNOBS = {
    # More near-miss figures + a deeper haystack to bury the needle in.
    "thread_reply": {"n_distractors": 6, "n_history": 150},
    # channel_setup has no retrieval; extra history only adds mild noise.
    "channel_setup": {"n_history": 80},
    # More components to aggregate, more superseded drafts per component.
    "rollup_reply": {"n_components": 4, "n_decoys": 3, "n_history": 150},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--out", default="tasks/instances")
    ap.add_argument("--hard", action="store_true", help="apply HARD_KNOBS presets")
    ap.add_argument("--seed-start", type=int, default=1,
                    help="first seed (use e.g. 100 for a --hard batch so task "
                         "ids don't collide with the base batch)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for tpl in TEMPLATES:
        knobs = HARD_KNOBS.get(tpl.template_id, {}) if args.hard else {}
        for seed in range(args.seed_start, args.seed_start + args.per_template):
            task = tpl.generate(seed, **knobs)
            if args.hard:
                task["knob_profile"] = "hard"
            path = out / f"{task['task_id']}.json"
            path.write_text(json.dumps(task, indent=2) + "\n")
            n += 1
    profile = "hard" if args.hard else "base"
    print(f"generated {n} {profile} instances from {len(TEMPLATES)} templates into {out}/")


if __name__ == "__main__":
    main()
