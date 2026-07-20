"""Central configuration for the gym.

Everything is overridable via environment variables so the same code runs
locally, in CI, and against a pooled deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    base_url: str = os.environ.get("CHATOPS_GYM_URL", "http://localhost:8065")
    admin_username: str = os.environ.get("CHATOPS_GYM_ADMIN", "gym-admin")
    admin_password: str = os.environ.get("CHATOPS_GYM_ADMIN_PW", "GymAdmin123!")
    admin_email: str = os.environ.get("CHATOPS_GYM_ADMIN_EMAIL", "gym-admin@example.com")
    # The identity the agent under evaluation acts as. Oracle, LLM and control
    # agents all act through this same user — same permissions, same tools.
    agent_username: str = "agent"
    agent_password: str = "AgentUser123!"
    agent_email: str = "agent@example.com"
    user_password: str = "WorldUser123!"  # seeded world users
    request_timeout: float = 15.0


CONFIG = Config()
