"""LLM agent: a minimal ReAct loop with pluggable backends.

Backends: Ollama (local, free — the default), Anthropic, OpenAI. The protocol
is deliberately simple JSON-in-text rather than native function calling so any
chat model works, including small local ones:

    model must answer with exactly one JSON object:
    {"thought": "...", "tool": "<tool_name>", "args": {...}}

Difficulty calibration is per-model by design: point this at any backend and
`calibration/report.py` re-measures the whole curve.
"""
from __future__ import annotations

import json
import os
import re

import requests

from gym.agents.base import Agent, ToolCall, Transcript

SYSTEM_TEMPLATE = """You are an assistant operating a Slack-like workspace through tools.

TASK:
{prompt}

TOOLS (call one per turn):
{tools}

Rules:
- Respond with EXACTLY one JSON object, no other text:
  {{"thought": "<brief reasoning>", "tool": "<name>", "args": {{...}}}}
- Observe each tool result before deciding the next call.
- When the task is complete, call the tool "finish" with a short summary.
- You have a limited number of turns; be efficient."""


class Backend:
    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OllamaBackend(Backend):
    def __init__(self, model: str = "qwen2.5:7b", url: str | None = None):
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")

    def chat(self, messages: list[dict]) -> str:
        resp = requests.post(f"{self.url}/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": 0.0},
        }, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class AnthropicBackend(Backend):
    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.model = model
        self.api_key = os.environ["ANTHROPIC_API_KEY"]

    def chat(self, messages: list[dict]) -> str:
        system = ""
        if messages and messages[0]["role"] == "system":
            system, messages = messages[0]["content"], messages[1:]
        resp = requests.post("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
        }, json={"model": self.model, "max_tokens": 1024, "system": system,
                 "messages": messages}, timeout=120)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


class OpenAIBackend(Backend):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.environ["OPENAI_API_KEY"]
        # Any OpenAI-compatible endpoint works (Groq, Together, vLLM, ...).
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def chat(self, messages: list[dict]) -> str:
        resp = requests.post(f"{self.base_url}/chat/completions", headers={
            "Authorization": f"Bearer {self.api_key}",
        }, json={"model": self.model, "messages": messages, "temperature": 0.0},
            timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class HFBackend(Backend):
    """Local HuggingFace transformers backend (CPU-friendly fallback when
    Ollama isn't available). Fine for small models, e.g.
    `hf:Qwen/Qwen2.5-0.5B-Instruct`; use Ollama or an API for 7B+."""

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch  # lazy: heavy deps only if this backend is used
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = getattr(torch, os.environ.get("CHATOPS_GYM_HF_DTYPE", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=dtype, device_map="cpu")

    def chat(self, messages: list[dict]) -> str:
        import torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def make_backend(spec: str) -> Backend:
    """'ollama:qwen2.5:7b' | 'anthropic:claude-sonnet-4-5' | 'openai:gpt-4o-mini'
    | 'hf:Qwen/Qwen2.5-0.5B-Instruct' (local CPU fallback)"""
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaBackend(model or "qwen2.5:7b")
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-sonnet-4-5")
    if kind == "openai":
        return OpenAIBackend(model or "gpt-4o-mini")
    if kind == "hf":
        return HFBackend(model or "Qwen/Qwen2.5-0.5B-Instruct")
    raise ValueError(f"unknown backend: {spec}")


class LLMAgent(Agent):
    def __init__(self, backend: Backend, name: str = "llm"):
        self.backend = backend
        self.name = name
        self.messages: list[dict] = []

    def start(self, prompt: str, tool_specs: list[dict]) -> None:
        tools = "\n".join(f"- {t['name']}: {t['description']} args={json.dumps(t['parameters']['properties'])}"
                          for t in tool_specs)
        self.messages = [{"role": "system",
                          "content": SYSTEM_TEMPLATE.format(prompt=prompt, tools=tools)},
                         {"role": "user", "content": "Begin."}]

    def act(self, transcript: Transcript) -> ToolCall | None:
        if transcript.steps:
            last = transcript.steps[-1]
            self.messages.append({"role": "user",
                                  "content": "Tool result:\n" + json.dumps(last["result"])[:4000]})
        raw = self.backend.chat(self.messages)
        self.messages.append({"role": "assistant", "content": raw})
        parsed = self._parse(raw)
        if parsed is None:
            # One repair attempt per turn: tell the model its output was invalid.
            self.messages.append({"role": "user",
                                  "content": "Invalid response. Reply with ONE JSON object: "
                                             '{"thought": "...", "tool": "...", "args": {...}}'})
            raw = self.backend.chat(self.messages)
            self.messages.append({"role": "assistant", "content": raw})
            parsed = self._parse(raw)
            if parsed is None:
                return None  # give up: episode ends, task fails
        return parsed

    @staticmethod
    def _parse(raw: str) -> ToolCall | None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return ToolCall(obj["tool"], obj.get("args", {}) or {})
        except Exception:
            return None
