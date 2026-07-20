"""The agent's action surface: ~12 tools wrapping the Mattermost API.

This is the ONLY way any agent — LLM, oracle, or control — acts on the world.
The oracle running through this exact layer (same user, same permissions) is
what makes "oracle passes" mean "solvable by the agent", not merely "solvable
by the API". If a task needs a capability these tools don't expose, oracle
validation fails and the task never ships.

Every tool returns a JSON-serializable dict. Errors return {"error": ...}
instead of raising: agents must be able to observe and recover from failures.
"""
from __future__ import annotations

from typing import Any, Callable

from gym.client import ApiError
from gym.env import MattermostEnv

MAX_RESULT_POSTS = 25


def _post_view(env: MattermostEnv, p: dict, channel_name: str | None = None) -> dict:
    return {
        "post_id": p["id"],
        "channel": channel_name or p.get("channel_name") or "",
        "user": env.username_of(p["user_id"]),
        "message": p["message"],
        "root_id": p.get("root_id", ""),
        "is_pinned": bool(p.get("is_pinned")),
    }


# Attach a reverse user lookup to the env (kept here to keep env.py lean).
def _username_of(self: MattermostEnv, user_id: str) -> str:
    for name, uid in self._users.items():
        if uid == user_id:
            return name
    try:
        return self.admin.get(f"/users/{user_id}")["username"]
    except Exception:
        return "unknown"


MattermostEnv.username_of = _username_of  # type: ignore[attr-defined]


class Tool:
    def __init__(self, name: str, description: str, params: dict,
                 fn: Callable[[MattermostEnv, dict], dict]):
        self.name, self.description, self.params, self.fn = name, description, params, fn

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.params}


class ToolRegistry:
    def __init__(self, env: MattermostEnv):
        self.env = env
        self.tools: dict[str, Tool] = {}
        for t in _build_tools():
            self.tools[t.name] = t

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> dict:
        if name not in self.tools:
            return {"error": f"unknown tool '{name}'. Available: {sorted(self.tools)}"}
        try:
            return self.tools[name].fn(self.env, args or {})
        except ApiError as e:
            return {"error": f"api error: {e.message}"}
        except KeyError as e:
            return {"error": f"missing or invalid argument: {e}"}
        except Exception as e:  # defensive: a tool bug must not kill the episode
            return {"error": f"tool failure: {type(e).__name__}: {e}"}


# --------------------------------------------------------------------- tools
def _list_channels(env: MattermostEnv, args: dict) -> dict:
    chans = env.admin.get_team_channels(env.team_id)
    return {"channels": [{"name": c["name"], "display_name": c["display_name"],
                          "header": c.get("header", ""), "type": c["type"]} for c in chans]}


def _read_channel(env: MattermostEnv, args: dict) -> dict:
    cid = env.channel_id(args["channel"])
    if cid is None:
        return {"error": f"channel not found: {args['channel']}"}
    page = int(args.get("page", 0))
    data = env.agent.get_channel_posts(cid, page=page, per_page=MAX_RESULT_POSTS)
    posts = sorted(data.get("posts", {}).values(), key=lambda p: p["create_at"], reverse=True)
    return {"page": page, "note": "newest first; increase page for older posts",
            "posts": [_post_view(env, p, args["channel"]) for p in posts]}


def _search_messages(env: MattermostEnv, args: dict) -> dict:
    data = env.agent.search_posts(env.team_id, args["query"])
    order = data.get("order", [])[:MAX_RESULT_POSTS]
    posts = data.get("posts", {})
    out = []
    for pid in order:
        p = posts[pid]
        cname = next((c["name"] for c in env.admin.get_team_channels(env.team_id)
                      if c["id"] == p["channel_id"]), "")
        out.append(_post_view(env, p, cname))
    return {"results": out}


def _get_thread(env: MattermostEnv, args: dict) -> dict:
    data = env.agent.get_thread(args["post_id"])
    posts = sorted(data.get("posts", {}).values(), key=lambda p: p["create_at"])
    return {"thread": [_post_view(env, p) for p in posts]}


def _send_message(env: MattermostEnv, args: dict) -> dict:
    cid = env.channel_id(args["channel"])
    if cid is None:
        return {"error": f"channel not found: {args['channel']}"}
    post = env.agent.create_post(cid, args["text"])
    return {"ok": True, "post_id": post["id"]}


def _reply_in_thread(env: MattermostEnv, args: dict) -> dict:
    root = env.agent.get_thread(args["root_post_id"])  # validates existence
    root_post = root["posts"][args["root_post_id"]]
    # Replying to a reply must target the true root.
    true_root = root_post.get("root_id") or args["root_post_id"]
    post = env.agent.create_post(root_post["channel_id"], args["text"], root_id=true_root)
    return {"ok": True, "post_id": post["id"], "root_id": true_root}


def _create_channel(env: MattermostEnv, args: dict) -> dict:
    ch = env.agent.create_channel(env.team_id, args["name"],
                                  args.get("display_name", args["name"]),
                                  purpose=args.get("purpose", ""))
    return {"ok": True, "channel": ch["name"]}


def _invite_user(env: MattermostEnv, args: dict) -> dict:
    cid = env.channel_id(args["channel"])
    if cid is None:
        return {"error": f"channel not found: {args['channel']}"}
    uid = env._users.get(args["username"])
    if uid is None:
        return {"error": f"user not found: {args['username']}"}
    env.agent.add_channel_member(cid, uid)
    return {"ok": True}


def _add_reaction(env: MattermostEnv, args: dict) -> dict:
    env.agent.add_reaction(env.agent.user_id, args["post_id"], args["emoji"])
    return {"ok": True}


def _pin_message(env: MattermostEnv, args: dict) -> dict:
    env.agent.pin_post(args["post_id"])
    return {"ok": True}


def _get_pinned(env: MattermostEnv, args: dict) -> dict:
    cid = env.channel_id(args["channel"])
    if cid is None:
        return {"error": f"channel not found: {args['channel']}"}
    data = env.agent.get_pinned_posts(cid)
    posts = sorted(data.get("posts", {}).values(), key=lambda p: p["create_at"])
    return {"pinned": [_post_view(env, p, args["channel"]) for p in posts]}


def _set_channel_header(env: MattermostEnv, args: dict) -> dict:
    cid = env.channel_id(args["channel"])
    if cid is None:
        return {"error": f"channel not found: {args['channel']}"}
    env.agent.patch_channel(cid, {"header": args["header"]})
    return {"ok": True}


def _finish(env: MattermostEnv, args: dict) -> dict:
    return {"done": True, "summary": args.get("summary", "")}


def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _build_tools() -> list[Tool]:
    return [
        Tool("list_channels", "List all channels in the workspace.", _p({}, []), _list_channels),
        Tool("read_channel", "Read recent posts in a channel (paginated, newest first).",
             _p({"channel": {"type": "string"}, "page": {"type": "integer"}}, ["channel"]),
             _read_channel),
        Tool("search_messages", "Full-text search across all channels.",
             _p({"query": {"type": "string"}}, ["query"]), _search_messages),
        Tool("get_thread", "Get all messages in the thread containing a post.",
             _p({"post_id": {"type": "string"}}, ["post_id"]), _get_thread),
        Tool("send_message", "Post a new message to a channel (NOT a thread reply).",
             _p({"channel": {"type": "string"}, "text": {"type": "string"}},
                ["channel", "text"]), _send_message),
        Tool("reply_in_thread", "Reply in the thread of an existing post.",
             _p({"root_post_id": {"type": "string"}, "text": {"type": "string"}},
                ["root_post_id", "text"]), _reply_in_thread),
        Tool("create_channel", "Create a public channel. Name must be lowercase-with-hyphens.",
             _p({"name": {"type": "string"}, "display_name": {"type": "string"},
                 "purpose": {"type": "string"}}, ["name"]), _create_channel),
        Tool("invite_user", "Add a user to a channel.",
             _p({"channel": {"type": "string"}, "username": {"type": "string"}},
                ["channel", "username"]), _invite_user),
        Tool("add_reaction", "React to a post with an emoji (e.g. 'white_check_mark').",
             _p({"post_id": {"type": "string"}, "emoji": {"type": "string"}},
                ["post_id", "emoji"]), _add_reaction),
        Tool("pin_message", "Pin a post to its channel.",
             _p({"post_id": {"type": "string"}}, ["post_id"]), _pin_message),
        Tool("get_pinned_messages", "List the pinned posts of a channel.",
             _p({"channel": {"type": "string"}}, ["channel"]), _get_pinned),
        Tool("set_channel_header", "Set a channel's header text.",
             _p({"channel": {"type": "string"}, "header": {"type": "string"}},
                ["channel", "header"]), _set_channel_header),
        Tool("finish", "Declare the task complete. Call when done.",
             _p({"summary": {"type": "string"}}, []), _finish),
    ]
