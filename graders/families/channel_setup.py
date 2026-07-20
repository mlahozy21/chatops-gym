"""Grader for the `channel_setup` family (L1/L2: named entity actions).

L1: create a public channel with the given name.
L2 additions (present in params when the instance is L2): header text must
match (typed), and a given user must be a member.

Anti-collateral: exactly the expected new channel, no stray posts or header
changes elsewhere.
"""
from __future__ import annotations

from graders import matchers
from graders.base import Grader, Verdict
from gym.env import MattermostEnv


class ChannelSetupGrader(Grader):
    def grade(self, env: MattermostEnv, params: dict) -> Verdict:
        name = params["channel_name"]
        cid = env.channel_id(name)
        if cid is None:
            return Verdict(False, f"channel '{name}' does not exist")

        ch = env.admin.get(f"/channels/{cid}")
        if ch["type"] != "O":
            return Verdict(False, "channel exists but is not public")

        if "header" in params:
            ok, why = matchers.match(params["header"], ch.get("header", ""))
            if not ok:
                return Verdict(False, f"header check failed: {why}")

        if "member" in params:
            members = {m["user_id"] for m in env.admin.get_channel_members(cid)}
            if env.user_id(params["member"]) not in members:
                return Verdict(False, f"user '{params['member']}' is not a member")

        diff = env.diff_since_reset()
        extra = [c for c in diff.new_channels if c["name"] != name]
        if extra:
            return Verdict(False, f"collateral channels created: {[c['name'] for c in extra]}")
        stray_posts = diff.posts_by(env.agent_user_id)
        if len(stray_posts) > params.get("max_posts", 0):
            return Verdict(False, f"unexpected agent posts: {len(stray_posts)}")
        bad_headers = [h for h in diff.changed_headers if h["name"] != name]
        if bad_headers:
            return Verdict(False, f"collateral header changes: {[h['name'] for h in bad_headers]}")
        return Verdict(True, "ok")


GRADER = ChannelSetupGrader()
