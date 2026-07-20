"""Grader for the `thread_reply` family (L3: retrieval + threaded action).

Passes iff the agent posted exactly one reply in the thread of the marked root
post, the reply asserts the required typed value (negation-guarded), and the
world diff shows no agent activity outside that thread.
"""
from __future__ import annotations

from graders import matchers
from graders.base import Grader, Verdict
from gym.env import MattermostEnv


class ThreadReplyGrader(Grader):
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


GRADER = ThreadReplyGrader()
