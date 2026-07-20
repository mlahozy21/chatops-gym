"""Difficulty calibration report: pass-rates per level and family.

The structural L1-L4 taxonomy is a prior; the truth is per-model. This script
re-derives the empirical curve for whatever backend produced the results file,
and flags tasks that fall outside the useful RLVR gradient band for that model.

Usage:
    python -m calibration.report results/ollama_qwen2.5_7b.jsonl [--band 0.10 0.60]
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

LEVELS = ["L1", "L2", "L3", "L4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--band", nargs=2, type=float, default=[0.10, 0.60],
                    metavar=("LO", "HI"),
                    help="useful gradient band for THIS model (pass-rate)")
    args = ap.parse_args()

    raw = [json.loads(line) for line in pathlib.Path(args.results).read_text().splitlines() if line]
    if not raw:
        raise SystemExit("empty results file")
    # run_eval appends across batches/retries; keep the latest row per task.
    latest: dict[str, dict] = {}
    for r in raw:
        latest[r["task_id"]] = r
    rows = list(latest.values())
    backend = rows[0]["backend"]

    by_level: dict[str, list[bool]] = defaultdict(list)
    by_family: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_level[r["difficulty"]].append(r["passed"])
        by_family[(r["template"], r["difficulty"])].append(r["passed"])

    print(f"\nDifficulty calibration — backend: {backend} ({len(rows)} episodes)")
    print(f"{'level':8s}{'n':>5s}{'pass-rate':>12s}   curve")
    for lvl in LEVELS:
        if lvl not in by_level:
            continue
        xs = by_level[lvl]
        rate = sum(xs) / len(xs)
        bar = "#" * round(rate * 40)
        print(f"{lvl:8s}{len(xs):>5d}{rate:>11.0%}   {bar}")

    lo, hi = args.band
    print(f"\nPer-family (useful gradient band for this model: {lo:.0%}-{hi:.0%}):")
    for (fam, lvl), xs in sorted(by_family.items(), key=lambda kv: kv[0][1]):
        rate = sum(xs) / len(xs)
        if rate > hi:
            note = "SATURATED for this model -> raise difficulty knobs / move down a level"
        elif rate < lo:
            note = "TOO HARD for this model -> lower knobs / move up a level"
        else:
            note = "in band"
        print(f"  {fam:20s} {lvl}  n={len(xs):<4d} pass={rate:>4.0%}  [{note}]")


if __name__ == "__main__":
    main()
