"""LLM client for deep analysis: trade reasoning, news & social sentiment.

Uses any OpenAI-compatible chat-completions endpoint. If no API key is
configured, Cortex falls back to deterministic text analysis so it remains
fully functional in "rules only" mode.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are Cortex, a disciplined swing-trading research assistant. "
    "You study multi-timeframe charts (15m, 1h, 4h, 1d, 1w, 1mo), focus on the 4-hour "
    "timeframe for entries, apply risk management, and NEVER place trades — you only advise. "
    "Be specific, cite evidence/sources where possible, and always give a risk opinion. "
    "Answer in valid JSON when JSON is requested."
)


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.3) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.enabled = bool(api_key)

    def chat(self, messages: list[dict[str, str]], response_format: str | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("LLM not configured")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        resp = requests.post(
            f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=120
        )
        if resp.status_code != 200:
            log.warning("LLM HTTP %s: %s", resp.status_code, resp.text[:300])
            raise RuntimeError(f"LLM request failed: {resp.status_code}")
        return resp.json()["choices"][0]["message"]["content"]

    def ask(self, prompt: str, system: str | None = None, response_format: str | None = None) -> str:
        sys_msg = system or _SYSTEM
        return self.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
        )

    def ask_json(self, prompt: str, system: str | None = None) -> dict:
        raw = self.ask(prompt, system=system, response_format="json")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip code fences if the model wrapped the JSON.
            stripped = raw.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("```", 2)[1]
                if stripped.startswith("json"):
                    stripped = stripped[4:]
            return json.loads(stripped)


def sentiment_from_text(text: str) -> dict:
    """Deterministic bag-of-words fallback sentiment (no LLM required)."""
    bull = {"beat", "surge", "gain", "upgrade", "growth", "strong", "record", "outperform", "positive", "bullish", "jump", "soar", "boost"}
    bear = {"miss", "plunge", "drop", "downgrade", "weak", "loss", "underperform", "sells", "concern", "bearish", "slump", "cut", "decline"}
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    b = len(words & bull)
    s = len(words & bear)
    score = (b - s) / max(1, b + s)
    label = "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")
    return {"label": label, "score": round(score, 3), "bullish_words": b, "bearish_words": s}