"""Mutation testing for graders — the centerpiece of the validation harness.

Rationale: null/random agents are weak tests; the false positives that poison
an RL signal come from plausible NEAR-MISSES, because a model mid-training
produces almost-correct behavior, not noise. Each family therefore defines
mutators that perturb the known-good oracle solution into named near-misses.
The grader must return 0 on EVERY mutant, or the task never ships.

Mutants derive from the same template params as the task itself (task["meta"]),
so they stay consistent with the instance by construction.

A mutant is: (name, expected_failure_hint, solution_steps).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class Mutant:
    name: str
    catches: str          # which grader bug this mutant would expose
    solution: list[dict]


def _find_step(solution: list[dict], tool: str) -> int:
    for i, s in enumerate(solution):
        if s["tool"] == tool:
            return i
    raise ValueError(f"oracle has no '{tool}' step")


# ------------------------------------------------------------- thread_reply
def _mut_thread_reply(task: dict) -> list[Mutant]:
    meta, sol = task["meta"], task["oracle_solution"]
    out: list[Mutant] = []
    i = _find_step(sol, "reply_in_thread")
    reply = sol[i]

    # 1. Right content, wrong location: channel message instead of thread reply.
    s = copy.deepcopy(sol)
    s[i] = {"tool": "send_message", "args": {"channel": meta["ask_channel"],
                                             "text": reply["args"]["text"]}}
    out.append(Mutant("channel_instead_of_thread",
                      "grader that only greps the channel for the figure", s))

    # 2. Right location, wrong figure (a seeded distractor).
    s = copy.deepcopy(sol)
    s[i]["args"]["text"] = f"The approved Q3 {meta['topic']} is ${meta['distractors'][0]:,}."
    out.append(Mutant("distractor_value", "over-lax value matching", s))

    # 3. Negated correct figure ("definitely not $X").
    s = copy.deepcopy(sol)
    s[i]["args"]["text"] = f"It's definitely not ${meta['value']:,}, let me check."
    out.append(Mutant("negated_value", "raw substring matching", s))

    # 4. Correct reply + spam elsewhere (collateral damage).
    s = copy.deepcopy(sol)
    s.insert(i + 1, {"tool": "send_message",
                     "args": {"channel": "random",
                              "text": f"FYI the {meta['topic']} is ${meta['value']:,}"}})
    out.append(Mutant("correct_plus_spam", "missing snapshot-diff collateral check", s))

    # 5. Read-only: searches but never acts.
    s = [st for st in sol if st["tool"] in ("read_channel", "get_pinned_messages", "finish")]
    out.append(Mutant("read_only", "grader satisfied by initial state", s))

    # 6. Hedged both ways: right figure AND a distractor asserted.
    s = copy.deepcopy(sol)
    s[i]["args"]["text"] = (f"The approved Q3 {meta['topic']} is ${meta['value']:,} "
                            f"or possibly ${meta['distractors'][0]:,}.")
    out.append(Mutant("hedged_answer", "no ambiguity rejection", s))
    return out


# ------------------------------------------------------------ channel_setup
def _mut_channel_setup(task: dict) -> list[Mutant]:
    meta, sol = task["meta"], task["oracle_solution"]
    out: list[Mutant] = []
    i = _find_step(sol, "create_channel")

    # 1. Typo'd channel name.
    s = copy.deepcopy(sol)
    wrong = meta["name"].replace("-", "") if "-" in meta["name"] else meta["name"] + "s"
    s[i]["args"]["name"] = wrong
    for st in s:  # keep the chain self-consistent: later steps hit the typo'd channel
        if st["tool"] in ("set_channel_header", "invite_user"):
            st["args"]["channel"] = wrong
    out.append(Mutant("wrong_channel_name", "fuzzy channel-name matching", s))

    # 2. Extra collateral channel.
    s = copy.deepcopy(sol)
    s.insert(i + 1, {"tool": "create_channel", "args": {"name": meta["name"] + "-backup"}})
    out.append(Mutant("extra_channel", "missing collateral-channel check", s))

    # 3. No-op.
    out.append(Mutant("noop", "grader satisfied by initial state",
                      [{"tool": "finish", "args": {}}]))

    if task["difficulty"] == "L2":
        # 4. Dropped step: header never set.
        s = [st for st in copy.deepcopy(sol) if st["tool"] != "set_channel_header"]
        out.append(Mutant("missing_header", "header not actually verified", s))
        # 5. Wrong user invited.
        s = copy.deepcopy(sol)
        j = _find_step(s, "invite_user")
        s[j]["args"]["username"] = "tomas" if meta["member"] != "tomas" else "zara"
        out.append(Mutant("wrong_member", "membership not verified against the named user", s))
    return out


# ------------------------------------------------------------- rollup_reply
def _mut_rollup_reply(task: dict) -> list[Mutant]:
    meta, sol = task["meta"], task["oracle_solution"]
    out: list[Mutant] = []
    i = _find_step(sol, "reply_in_thread")
    comp_list = ", ".join(meta["components"])

    # 1. Replies with a single component figure instead of the total.
    s = copy.deepcopy(sol)
    s[i]["args"]["text"] = f"The Q3 {meta['components'][0]} is ${meta['values'][0]:,}."
    out.append(Mutant("component_only", "grader accepting any seeded figure", s))

    # 2. Partial sum (drops one component).
    s = copy.deepcopy(sol)
    partial = sum(meta["values"][:-1])
    s[i]["args"]["text"] = f"Combined Q3 total for {comp_list}: ${partial:,}."
    out.append(Mutant("partial_sum", "total not actually verified", s))

    # 3. Sum computed with a superseded draft instead of a final figure.
    s = copy.deepcopy(sol)
    decoy_total = sum(meta["values"][1:]) + meta["decoys"][0]
    s[i]["args"]["text"] = f"Combined Q3 total for {comp_list}: ${decoy_total:,}."
    out.append(Mutant("decoy_component_sum", "draft/final status not discriminated", s))

    # 4. Right total, wrong location (channel post, not thread reply).
    s = copy.deepcopy(sol)
    s[i] = {"tool": "send_message", "args": {"channel": meta["ask_channel"],
                                             "text": f"Combined Q3 total: ${meta['total']:,}."}}
    out.append(Mutant("channel_instead_of_thread",
                      "grader that only greps the channel", s))

    # 5. Correct reply + duplicate broadcast elsewhere.
    s = copy.deepcopy(sol)
    s.insert(i + 1, {"tool": "send_message",
                     "args": {"channel": meta["channels"][0],
                              "text": f"FYI combined Q3 total: ${meta['total']:,}"}})
    out.append(Mutant("correct_plus_spam", "missing snapshot-diff collateral check", s))

    # 6. Read-only.
    s = [st for st in sol if st["tool"] in ("read_channel", "finish")]
    out.append(Mutant("read_only", "grader satisfied by initial state", s))
    return out


MUTATORS = {
    "thread_reply": _mut_thread_reply,
    "channel_setup": _mut_channel_setup,
    "rollup_reply": _mut_rollup_reply,
}


def mutants_for(task: dict) -> list[Mutant]:
    fn = MUTATORS.get(task["template_id"])
    if fn is None:
        raise ValueError(f"no mutators registered for template '{task['template_id']}' — "
                         "every family MUST define mutation tests before shipping")
    return fn(task)
