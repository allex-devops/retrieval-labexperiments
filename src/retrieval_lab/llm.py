"""A small client for any OpenAI-compatible endpoint: chat and embeddings, with retries and backoff.

Configured from LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and EMBED_MODEL; nothing has a default,
so a missing setting fails loudly instead of quietly calling the wrong provider.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable

import httpx

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


class LLM:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise LLMError("no endpoint configured: set LLM_BASE_URL (see .env.example)")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL")
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = None
            try:
                resp = self._client.post(
                    f"{self.base_url}{path}", json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
                )
            except httpx.TransportError as e:  # refused, reset, timed out
                last = e
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"{resp.status_code} from {path}: {resp.text[:300]}")
                last = LLMError(f"{resp.status_code} from {path}")
                wait = _retry_after(resp)
            if attempt < self.retries:
                delay = self.backoff * 2**attempt * (0.5 + random.random() / 2)
                self.sleep(wait if wait is not None else delay)
        raise LLMError(f"gave up on {path} after {self.retries + 1} attempts: {last}") from last

    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int | None = None, **extra: Any) -> str:
        if not self.model:
            raise LLMError("no chat model configured: set LLM_MODEL")
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, **extra}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return self._post("/chat/completions", payload)["choices"][0]["message"].get("content") or ""

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.embed_model
        if not model:
            raise LLMError("no embedding model configured: set EMBED_MODEL")
        data = self._post("/embeddings", {"model": model, "input": texts})["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
