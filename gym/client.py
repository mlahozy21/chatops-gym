"""Thin deterministic client for the Mattermost REST API (v4).

Only the endpoints the gym needs. Raises ApiError with the server's message on
failure so graders and the validation harness always see *why* a call failed.
"""
from __future__ import annotations

from typing import Any, Optional

import requests

from gym.config import CONFIG


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


class MattermostClient:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or CONFIG.base_url).rstrip("/")
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.token = token
        self.user_id: Optional[str] = None

    # -- plumbing ------------------------------------------------------------
    def _req(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/api/v4{path}"
        resp = self.session.request(method, url, timeout=CONFIG.request_timeout, **kwargs)
        if resp.status_code >= 400:
            try:
                msg = resp.json().get("message", resp.text)
            except Exception:
                msg = resp.text
            raise ApiError(resp.status_code, msg)
        if resp.text:
            return resp.json()
        return None

    def get(self, path: str, **kw) -> Any:
        return self._req("GET", path, **kw)

    def post(self, path: str, json: Any = None, **kw) -> Any:
        return self._req("POST", path, json=json, **kw)

    def put(self, path: str, json: Any = None, **kw) -> Any:
        return self._req("PUT", path, json=json, **kw)

    def delete(self, path: str, **kw) -> Any:
        return self._req("DELETE", path, **kw)

    # -- auth ----------------------------------------------------------------
    def login(self, login_id: str, password: str) -> None:
        url = f"{self.base_url}/api/v4/users/login"
        resp = self.session.post(
            url, json={"login_id": login_id, "password": password},
            timeout=CONFIG.request_timeout,
        )
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text)
        self.token = resp.headers["Token"]
        self.session.headers["Authorization"] = f"Bearer {self.token}"
        self.user_id = resp.json()["id"]

    def ping(self) -> bool:
        try:
            self.get("/system/ping")
            return True
        except Exception:
            return False

    # -- users ---------------------------------------------------------------
    def create_user(self, username: str, email: str, password: str) -> dict:
        return self.post("/users", {"username": username, "email": email, "password": password})

    def get_user_by_username(self, username: str) -> dict:
        return self.get(f"/users/username/{username}")

    # -- teams ---------------------------------------------------------------
    def create_team(self, name: str, display_name: str) -> dict:
        return self.post("/teams", {"name": name, "display_name": display_name, "type": "O"})

    def add_team_member(self, team_id: str, user_id: str) -> dict:
        return self.post(f"/teams/{team_id}/members", {"team_id": team_id, "user_id": user_id})

    def get_team_channels(self, team_id: str, per_page: int = 200) -> list[dict]:
        return self.get(f"/teams/{team_id}/channels?per_page={per_page}")

    # -- channels ------------------------------------------------------------
    def create_channel(self, team_id: str, name: str, display_name: str,
                       purpose: str = "", type_: str = "O") -> dict:
        return self.post("/channels", {
            "team_id": team_id, "name": name, "display_name": display_name,
            "purpose": purpose, "type": type_,
        })

    def get_channel_by_name(self, team_id: str, name: str) -> dict:
        return self.get(f"/teams/{team_id}/channels/name/{name}")

    def patch_channel(self, channel_id: str, patch: dict) -> dict:
        return self.put(f"/channels/{channel_id}/patch", patch)

    def add_channel_member(self, channel_id: str, user_id: str) -> dict:
        return self.post(f"/channels/{channel_id}/members", {"user_id": user_id})

    def get_channel_members(self, channel_id: str, per_page: int = 200) -> list[dict]:
        return self.get(f"/channels/{channel_id}/members?per_page={per_page}")

    # -- posts ---------------------------------------------------------------
    def create_post(self, channel_id: str, message: str, root_id: str = "") -> dict:
        return self.post("/posts", {"channel_id": channel_id, "message": message, "root_id": root_id})

    def get_channel_posts(self, channel_id: str, page: int = 0, per_page: int = 200) -> dict:
        return self.get(f"/channels/{channel_id}/posts?page={page}&per_page={per_page}")

    def get_thread(self, post_id: str) -> dict:
        return self.get(f"/posts/{post_id}/thread")

    def pin_post(self, post_id: str) -> None:
        self.post(f"/posts/{post_id}/pin")

    def get_pinned_posts(self, channel_id: str) -> dict:
        return self.get(f"/channels/{channel_id}/pinned")

    def search_posts(self, team_id: str, terms: str) -> dict:
        return self.post(f"/teams/{team_id}/posts/search", {"terms": terms, "is_or_search": False})

    def add_reaction(self, user_id: str, post_id: str, emoji_name: str) -> dict:
        return self.post("/reactions", {"user_id": user_id, "post_id": post_id, "emoji_name": emoji_name})
