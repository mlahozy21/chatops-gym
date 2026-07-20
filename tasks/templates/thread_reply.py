"""Template: thread_reply (L3 — retrieval across channels + threaded action).

Failure mode targeted (observed, see analysis/transcripts/): agents locate the
right information but reply as a NEW channel message instead of in the asker's
thread, or pick a distractor value when several similar figures exist.

Difficulty knobs:
  n_distractors  — similar-but-wrong figures seeded near the true one
  n_history      — depth of background chatter the needle is buried in
"""
from __future__ import annotations

import random

from tasks.templates.base import TaskTemplate

_TOPICS = [
    ("marketing budget", "finance"),
    ("travel budget", "finance"),
    ("hiring budget", "finance"),
    ("cloud spend cap", "finance"),
    ("offsite budget", "finance"),
]
_ASK_CHANNELS = ["q3-planning", "product", "marketing"]
_ASKERS = ["diana", "arjun", "sofia", "priya", "carlos"]

_QUESTIONS = [
    "Does anyone know the final {topic} for Q3?",
    "Quick one — what did the {topic} end up being for Q3?",
    "Can someone confirm the approved {topic} for this quarter?",
]
_PROMPTS = [
    "Someone in ~{ask_channel} asked about the {topic} and never got an answer. "
    "Find that message and reply in its thread with the correct figure. "
    "The approved figure is recorded somewhere in the ~{hide_channel} channel.",
    "There's an unanswered question about the {topic} in ~{ask_channel}. "
    "Reply in the thread of that question with the right amount — you'll find "
    "the approved number in ~{hide_channel}.",
]


class ThreadReplyTemplate(TaskTemplate):
    template_id = "thread_reply"
    difficulty = "L3"

    def generate(self, seed: int, n_distractors: int = 2, n_history: int = 60) -> dict:
        rng = random.Random(seed * 7919 + 13)
        topic, hide_channel = _TOPICS[rng.randrange(len(_TOPICS))]
        ask_channel = _ASK_CHANNELS[rng.randrange(len(_ASK_CHANNELS))]
        asker = _ASKERS[rng.randrange(len(_ASKERS))]
        value = rng.randrange(20, 200) * 500  # e.g. 48500
        pretty = f"${value:,}"

        question = _QUESTIONS[rng.randrange(len(_QUESTIONS))].format(topic=topic)
        prompt = _PROMPTS[rng.randrange(len(_PROMPTS))].format(
            topic=topic, ask_channel=ask_channel, hide_channel=hide_channel)

        overlays = [
            {"op": "post_message", "channel": ask_channel, "user": asker,
             "text": question, "marker": "root_question"},
            {"op": "pin_message", "channel": hide_channel, "user": "ingrid",
             "text": f"Q3 {topic} approved: {pretty}", "marker": "answer_pin"},
        ]
        # Distractors: plausible near-miss figures in the same channel.
        distractor_values = []
        for _ in range(n_distractors):
            dv = value
            while dv == value or dv in distractor_values:
                dv = value + rng.choice([-1, 1]) * rng.randrange(1, 20) * 500
            distractor_values.append(dv)
            overlays.append({
                "op": "post_message", "channel": hide_channel,
                "user": _ASKERS[rng.randrange(len(_ASKERS))],
                "text": rng.choice([
                    f"Early draft had the {topic} at ${dv:,}, superseded since.",
                    f"FYI last year's {topic} was ${dv:,}.",
                    f"If we hadn't cut scope, {topic} would be around ${dv:,}.",
                ]),
            })

        return {
            "task_id": self.instance_id(seed),
            "template_id": self.template_id,
            "seed": seed,
            "difficulty": self.difficulty,
            "category": "retrieval_and_action",
            "failure_mode": "replies in channel instead of thread; picks distractor figure",
            "prompt": prompt,
            "setup": {"world_seed": seed, "n_history": n_history, "overlays": overlays},
            "grader": {
                "module": "graders.families.thread_reply",
                "params": {
                    "root_marker": "root_question",
                    "answer": {"type": "money", "value": value},
                    "max_replies": 1,
                },
            },
            "oracle_solution": [
                {"tool": "read_channel", "args": {"channel": ask_channel}},
                {"tool": "get_pinned_messages", "args": {"channel": hide_channel}},
                {"tool": "reply_in_thread", "args": {
                    "root_post_id": {"resolve": "post_in_channel", "channel": ask_channel,
                                     "must_contain": question[:40],
                                     "expect_marker": "root_question"},
                    "text": f"The approved Q3 {topic} is {pretty}.",
                }},
                {"tool": "finish", "args": {"summary": "replied in thread with the figure"}},
            ],
            # Extra params mutations need (not used by the grader).
            "meta": {"topic": topic, "ask_channel": ask_channel, "hide_channel": hide_channel,
                     "value": value, "distractors": distractor_values, "question": question},
        }


TEMPLATE = ThreadReplyTemplate()
