"""Grader for the `rollup_reply` family (L4 — cross-channel aggregation).

Passes iff the agent posted exactly one reply in the asker's thread asserting
the correct TOTAL (component figures may appear — they're in allow_extra —
but any decoy/draft figure or other stray number fails the ambiguity check),
with no collateral world changes.
"""
from __future__ import annotations

from graders import matchers
from graders.base import Grader, Verdict
from gym.env import MattermostEnv


class RollupReplyGrader(Grader):
    def grade(self, env: MattermostEnv, params: dict) -> Verdict:
        root = env.resolve_marker(params["root_marker"])
        replies = env.get_thread_replies(root["id"], author_id=env.agent_user_id)

        if not replies:
            return Verdict(False, "no agent reply in the target thread")
        if len(replies) > params.get("max_replies", 1):
            return Verdict(False, f"{len(replies)} replies in thread; expected 1")

        ok, why = matchers.match(params["answer"], replies[0]["message"])
        if not ok:
            return Verdict(False, f"reply content check failed: {why}")

        diff = env.diff_since_reset()
        stray = diff.agent_posts_outside(env.agent_user_id, thread_root=root["id"])
        if stray:
            where = sorted({p.channel_name for p in stray})
            return Verdict(False, f"collateral agent posts outside the thread in: {where}")
        if diff.new_channels or diff.changed_headers:
            return Verdict(False, "collateral world changes (channels/headers)")
        return Verdict(True, "ok")


GRADER = RollupReplyGrader()
