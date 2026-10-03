"""Model access: a thin wrapper around the Anthropic Messages API, plus a scripted stand-in.

`create(...)` is used by the agent loop (tool use); `complete_json(...)` asks for a reply that
matches a JSON schema (structured outputs). `ScriptedLLM` replays canned turns so everything
can be tested offline, without an API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

DEFAULT_MODEL = "claude-opus-5-5"

# Models that accept the server-side refusal fallback in its "default" form.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

# Approximate USD per million tokens: (input, output, cache read, cache write).
# Used only for the cost estimate in reports; check current pricing before relying on it.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class ClaudeLLM:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 16000):
        import anthropic  # imported here so the offline tests never need the SDK configured

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def create(self, *, system: str, messages: list[dict], tools: list[dict] | None = None, output_format: dict | None = None):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},  # each turn re-reads the conversation prefix from cache
        }
        output_config = {}
        if not self.model.startswith("claude-haiku"):
            # Adaptive thinking with readable summaries for the transcript; effort sets how hard it thinks.
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
            output_config["effort"] = self.effort
        if output_format:
            output_config["format"] = output_format
        if output_config:
            request["output_config"] = output_config
        if tools:
            request["tools"] = tools
        if self.model in _FALLBACK_MODELS:
            # If a safety classifier declines a turn, the API re-runs it on a fallback model.
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        return self.client.messages.create(**request)

    def complete_json(self, *, system: str, messages: list[dict], schema: dict):
        """One request whose reply must match `schema`. Returns (parsed JSON, raw response)."""
        response = self.create(system=system, messages=messages, output_format={"type": "json_schema", "schema": schema})
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"model stopped with {response.stop_reason!r}")
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text), response


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    prices = PRICES.get(model)
    if prices is None:
        return None
    return sum(usage.get(f, 0) * p for f, p in zip(USAGE_FIELDS, prices)) / 1e6


class ScriptedLLM:
    """Replays a fixed script instead of calling a model.

    For `create`, each entry is a list of (tool_name, tool_input) pairs (one assistant turn of
    tool calls) or a string (a text-only turn). For `complete_json`, each entry is a dict.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[dict] = []

    def create(self, *, system: str, messages: list[dict], tools: list[dict] | None = None, output_format: dict | None = None):
        self.requests.append({"system": system, "tools": tools, "messages": list(messages)})
        step = self.script.pop(0) if self.script else "(script exhausted)"
        if isinstance(step, dict):
            content, stop_reason = [SimpleNamespace(type="text", text=json.dumps(step))], "end_turn"
        elif isinstance(step, str):
            content, stop_reason = [SimpleNamespace(type="text", text=step)], "end_turn"
        else:
            content = [
                SimpleNamespace(type="tool_use", id=f"toolu_{len(self.requests)}_{i}", name=name, input=args)
                for i, (name, args) in enumerate(step)
            ]
            stop_reason = "tool_use"
        usage = SimpleNamespace(**{f: 0 for f in USAGE_FIELDS})
        return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)

    def complete_json(self, *, system: str, messages: list[dict], schema: dict):
        response = self.create(system=system, messages=messages, output_format={"type": "json_schema", "schema": schema})
        return json.loads(response.content[0].text), response
