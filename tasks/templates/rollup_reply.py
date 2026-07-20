"""Template: rollup_reply (L4 — cross-channel aggregation, no retrieval shortcut).

Why this family exists (observed, not assumed): Qwen2.5-32B saturated the
whole base curriculum AND the --hard batch (15/15, near-optimal calls). Root
cause of the L3 saturation: the answer is *pinned*, so `get_pinned_messages`
gives strong models an O(1) retrieval path — no amount of history distractors
competes with a pin. The structural fixes here:

  * no pins — figures live in plain message history;
  * status language: each true figure coexists with superseded drafts
    ("draft had it at $X", "final approved: $Y") that must be discriminated;
  * aggregation: the asked-for number exists NOWHERE in the workspace — the
    agent must find N component figures across N channels and sum them;
  * still exactly one threaded reply, collateral-checked.

Difficulty knobs: n_components (channels to aggregate), n_decoys (superseded
drafts per component), n_history (haystack depth).
"""
from __future__ import annotations

import random

from tasks.templates.base import TaskTemplate

_COMPONENTS = [
    ("marketing budget", "marketing"),
    ("travel budget", "hr"),
    ("cloud spend", "eng-oncall"),
    ("tooling budget", "product"),
    ("events budget", "random"),
]
_ASK_CHANNELS = ["q3-planning", "finance"]
_ASKERS = ["diana", "arjun", "sofia", "priya", "carlos", "ingrid"]

_FINAL_PHRASES = [
    "Final {name} approved for Q3: ${v:,}.",
    "Sign-off done — Q3 {name} is ${v:,} (final).",
    "Confirmed with leadership: {name} for Q3 approved at ${v:,}.",
]
_DECOY_PHRASES = [
    "First draft had the {name} at ${v:,}, superseded since.",
    "If we hadn't cut scope, the {name} would be around ${v:,}.",
    "Last year's {name} was ${v:,}, for reference.",
    "Early proposal: {name} at ${v:,} — NOT final.",
]


class RollupReplyTemplate(TaskTemplate):
    template_id = "rollup_reply"
    difficulty = "L4"

    def generate(self, seed: int, n_components: int = 3, n_decoys: int = 2,
                 n_history: int = 100) -> dict:
        rng = random.Random(seed * 104729 + 71)
        comps = rng.sample(_COMPONENTS, n_components)
        ask_channel = rng.choice(_ASK_CHANNELS)
        asker = rng.choice(_ASKERS)

        values: list[int] = []
        overlays: list[dict] = []
        decoy_values: list[int] = []
        comp_list = ", ".join(c[0] for c in comps[:-1]) + f" and {comps[-1][0]}"

        question = (f"Can someone give me ONE combined number for our Q3 "
                    f"{comp_list}? I keep seeing partial figures everywhere.")
        overlays.append({"op": "post_message", "channel": ask_channel, "user": asker,
                        "text": question, "marker": "root_question"})

        for name, channel in comps:
            v = rng.randrange(20, 180) * 500
            values.append(v)
            phrase = rng.choice(_FINAL_PHRASES).format(name=name, v=v)
            overlays.append({"op": "post_message", "channel": channel,
                             "user": rng.choice(_ASKERS), "text": phrase,
                             "marker": f"final_{channel}"})
            for _ in range(n_decoys):
                dv = v
                while dv == v or dv in decoy_values:
                    dv = v + rng.choice([-1, 1]) * rng.randrange(1, 25) * 500
                decoy_values.append(dv)
                overlays.append({"op": "post_message", "channel": channel,
                                 "user": rng.choice(_ASKERS),
                                 "text": rng.choice(_DECOY_PHRASES).format(name=name, v=dv)})
        total = sum(values)

        prompt = (f"In ~{ask_channel}, {asker} asked for one combined Q3 figure "
                  f"covering {comp_list}. Work out the total from the final approved "
                  f"figures (they're posted in the relevant team channels — beware of "
                  f"superseded drafts) and reply in the thread of {asker}'s question "
                  f"with the combined amount.")

        oracle = [{"tool": "read_channel", "args": {"channel": ask_channel}}]
        for _, channel in comps:
            oracle.append({"tool": "read_channel", "args": {"channel": channel}})
        oracle += [
            {"tool": "reply_in_thread", "args": {
                "root_post_id": {"resolve": "post_in_channel", "channel": ask_channel,
                                 "must_contain": "ONE combined number",
                                 "expect_marker": "root_question"},
                "text": f"Combined Q3 total for {comp_list}: ${total:,} "
                        f"({' + '.join(f'${v:,}' for v in values)}).",
            }},
            {"tool": "finish", "args": {"summary": "replied with the combined total"}},
        ]

        return {
            "task_id": self.instance_id(seed),
            "template_id": self.template_id,
            "seed": seed,
            "difficulty": self.difficulty,
            "category": "aggregation",
            "failure_mode": ("observed at L3: pinned answers give strong models an O(1) "
                             "shortcut — here the target number exists nowhere and must "
                             "be computed from de-pinned, status-ambiguous sources"),
            "prompt": prompt,
            "setup": {"world_seed": seed, "n_history": n_history, "overlays": overlays},
            "grader": {
                "module": "graders.families.rollup_reply",
                "params": {
                    "root_marker": "root_question",
                    "answer": {"type": "money", "value": total,
                               "allow_extra": values},  # breakdown is fine; decoys are not
                    "max_replies": 1,
                },
            },
            "oracle_solution": oracle,
            "meta": {"components": [c[0] for c in comps],
                     "channels": [c[1] for c in comps],
                     "values": values, "total": total,
                     "decoys": decoy_values, "ask_channel": ask_channel,
                     "question": question},
        }


TEMPLATE = RollupReplyTemplate()
