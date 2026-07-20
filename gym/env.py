"""MattermostEnv: episode lifecycle, markers, snapshot-diff.

Reset strategy (namespacing by team): every episode gets a *fresh Mattermost
team*, seeded deterministically from the task's world_seed, with the task's
overlays applied on top. This is fast (no global wipe), naturally concurrent
(N episodes = N teams), and makes the snapshot scope trivial (the team).

Markers: overlays that create entities the grader or oracle must reference
declare a `marker`. The env records marker -> concrete entity for THIS episode.
Graders resolve markers directly (they are trusted infrastructure). The oracle
does NOT: it re-finds entities through the agent's own tools, and the harness
only uses the marker registry to *verify* the oracle found the right one.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from gym.client import ApiError, MattermostClient
from gym.config import CONFIG
from gym.snapshot import PostRecord, WorldDiff, WorldSnapshot, diff_snapshots
from gym.world import WorldSpec, build_world


class EnvError(RuntimeError):
    pass


class MattermostEnv:
    def __init__(self, base_url: str | None = None):
        self.admin = MattermostClient(base_url)
        self.agent = MattermostClient(base_url)
        self._bootstrapped = False
        # Per-episode state
        self.team_id: str | None = None
        self.team_name: str | None = None
        self.markers: dict[str, dict] = {}
        self._snapshot: WorldSnapshot | None = None
        self._users: dict[str, str] = {}       # username -> user_id
        self._channels: dict[str, str] = {}    # channel name -> channel_id
        self.world: WorldSpec | None = None

    # ------------------------------------------------------------------ boot
    def bootstrap(self) -> None:
        """Idempotent server-level setup: admin + global users. Runs once."""
        if self._bootstrapped:
            return
        # The preview container takes ~30-90s to become healthy after `up -d`;
        # poll instead of failing on the first attempt.
        deadline = time.time() + 120
        while not self.admin.ping():
            if time.time() > deadline:
                raise EnvError(f"Mattermost not reachable at {self.admin.base_url} "
                               "after 120s. Is the container running? "
                               "`docker compose -f docker/docker-compose.yml up -d`")
            print("waiting for Mattermost to become healthy...", flush=True)
            time.sleep(5)
        # First user created on a fresh server becomes system admin.
        try:
            self.admin.create_user(CONFIG.admin_username, CONFIG.admin_email, CONFIG.admin_password)
        except ApiError:
            pass  # already exists
        self.admin.login(CONFIG.admin_username, CONFIG.admin_password)

        for username, email, pw in [(CONFIG.agent_username, CONFIG.agent_email, CONFIG.agent_password)]:
            self._ensure_user(username, email, pw)
        self._bootstrapped = True

    def _ensure_user(self, username: str, email: str, password: str) -> str:
        try:
            user = self.admin.create_user(username, email, password)
        except ApiError:
            user = self.admin.get_user_by_username(username)
        self._users[username] = user["id"]
        return user["id"]

    # ----------------------------------------------------------------- reset
    def reset(self, task: dict) -> None:
        """Fresh episode team seeded from the task's world + overlays."""
        self.bootstrap()
        setup = task["setup"]
        world = build_world(setup["world_seed"], n_history=setup.get("n_history", 60))
        self.world = world
        self.markers = {}
        self._channels = {}

        suffix = uuid.uuid4().hex[:8]
        self.team_name = f"ep-{suffix}"
        team = self.admin.create_team(self.team_name, f"Episode {suffix}")
        self.team_id = team["id"]

        # Users: global (idempotent), then joined to this team.
        for username in world.usernames:
            uid = self._users.get(username) or self._ensure_user(
                username, f"{username}@example.com", CONFIG.user_password)
            self.admin.add_team_member(self.team_id, uid)
        self.admin.add_team_member(self.team_id, self._users[CONFIG.agent_username])

        # Channels.
        for name, display in world.channels:
            ch = self.admin.create_channel(self.team_id, name, display)
            self._channels[name] = ch["id"]
            for username in world.usernames:
                try:
                    self.admin.add_channel_member(ch["id"], self._users[username])
                except ApiError:
                    pass
            self.admin.add_channel_member(ch["id"], self._users[CONFIG.agent_username])

        # History (as each author, via impersonation-free per-user clients would
        # be N logins; admin posting on behalf is not supported — so we log in
        # once per distinct author lazily and cache the client).
        author_clients: dict[str, MattermostClient] = {}
        posted_ids: list[str] = []
        for hp in world.history:
            client = author_clients.get(hp.username)
            if client is None:
                client = MattermostClient(self.admin.base_url)
                client.login(hp.username, CONFIG.user_password)
                author_clients[hp.username] = client
            root_id = posted_ids[hp.thread_parent] if hp.thread_parent is not None else ""
            post = client.create_post(self._channels[hp.channel], hp.text, root_id=root_id)
            posted_ids.append(post["id"])

        # Overlays: the task's own deltas. Markers get registered here.
        for ov in setup.get("overlays", []):
            self._apply_overlay(ov, author_clients)

        # Log the agent in and snapshot the pristine world.
        self.agent = MattermostClient(self.admin.base_url)
        self.agent.login(CONFIG.agent_username, CONFIG.agent_password)
        self._snapshot = self.take_snapshot()

    def _author_client(self, username: str,
                       cache: dict[str, MattermostClient]) -> MattermostClient:
        client = cache.get(username)
        if client is None:
            client = MattermostClient(self.admin.base_url)
            client.login(username, CONFIG.user_password)
            cache[username] = client
        return client

    def _apply_overlay(self, ov: dict, author_clients: dict) -> None:
        op = ov["op"]
        if op == "post_message":
            client = self._author_client(ov["user"], author_clients)
            root_id = ""
            if ov.get("root_marker"):
                root_id = self.markers[ov["root_marker"]]["id"]
            post = client.create_post(self._channels[ov["channel"]], ov["text"], root_id=root_id)
            if ov.get("marker"):
                self.markers[ov["marker"]] = {"type": "post", "id": post["id"],
                                              "channel_id": post["channel_id"], "text": ov["text"]}
        elif op == "pin_message":
            client = self._author_client(ov.get("user", self.world.usernames[0]), author_clients)
            post = client.create_post(self._channels[ov["channel"]], ov["text"])
            client.pin_post(post["id"])
            if ov.get("marker"):
                self.markers[ov["marker"]] = {"type": "post", "id": post["id"],
                                              "channel_id": post["channel_id"], "text": ov["text"]}
        elif op == "set_channel_header":
            self.admin.patch_channel(self._channels[ov["channel"]], {"header": ov["header"]})
        else:
            raise EnvError(f"unknown overlay op: {op}")

    # -------------------------------------------------------------- snapshot
    def take_snapshot(self) -> WorldSnapshot:
        snap = WorldSnapshot()
        for ch in self.admin.get_team_channels(self.team_id):
            snap.channels[ch["id"]] = {"name": ch["name"], "header": ch.get("header", ""),
                                       "type": ch["type"]}
            page = 0
            while True:
                data = self.admin.get_channel_posts(ch["id"], page=page)
                posts = data.get("posts", {})
                for pid, p in posts.items():
                    snap.posts[pid] = PostRecord(
                        id=pid, channel_id=p["channel_id"], channel_name=ch["name"],
                        user_id=p["user_id"], message=p["message"], root_id=p.get("root_id", ""),
                        is_pinned=bool(p.get("is_pinned")), type=p.get("type", ""),
                    )
                if len(posts) < 200:
                    break
                page += 1
            members = self.admin.get_channel_members(ch["id"])
            snap.members[ch["id"]] = frozenset(m["user_id"] for m in members)
        return snap

    def diff_since_reset(self) -> WorldDiff:
        if self._snapshot is None:
            raise EnvError("no snapshot; call reset() first")
        return diff_snapshots(self._snapshot, self.take_snapshot())

    # --------------------------------------------------------------- helpers
    @property
    def agent_user_id(self) -> str:
        return self._users[CONFIG.agent_username]

    def user_id(self, username: str) -> str:
        return self._users[username]

    def channel_id(self, name: str) -> str | None:
        if name in self._channels:
            return self._channels[name]
        try:  # channels the agent created during the episode
            ch = self.admin.get_channel_by_name(self.team_id, name)
            return ch["id"]
        except ApiError:
            return None

    def resolve_marker(self, marker: str) -> dict:
        if marker not in self.markers:
            raise EnvError(f"unknown marker: {marker}")
        return self.markers[marker]

    def get_thread_replies(self, root_id: str, author_id: str | None = None) -> list[dict]:
        thread = self.admin.get_thread(root_id)
        replies = [p for pid, p in thread.get("posts", {}).items()
                   if pid != root_id and p.get("root_id") == root_id]
        replies.sort(key=lambda p: p["create_at"])
        if author_id:
            replies = [p for p in replies if p["user_id"] == author_id]
        return replies
