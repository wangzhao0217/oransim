"""Native Ollama chat provider.

Ollama exposes an OpenAI-compatible endpoint, but local reasoning models can
return their final answer in a separate ``content`` field only when the native
``/api/chat`` endpoint is called with ``think: false``. This adapter keeps
persona narratives parseable without changing the provider-neutral interface.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

from .base import GenerateResult

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))


@dataclass
class OllamaNativeProvider:
    """POST to Ollama's native ``/api/chat`` endpoint."""

    base_url: str
    timeout: float = DEFAULT_TIMEOUT
    name: str = "ollama"

    def build_body(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    def generate(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
    ) -> GenerateResult:
        if stream:
            raise ValueError("OllamaNativeProvider only supports buffered responses")
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/api/chat"
        body = self.build_body(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        content = ((parsed.get("message") or {}).get("content") or "").strip()
        return GenerateResult(
            content=content,
            usage={
                "prompt_tokens": int(parsed.get("prompt_eval_count", 0) or 0),
                "completion_tokens": int(parsed.get("eval_count", 0) or 0),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )


__all__ = ["OllamaNativeProvider"]
