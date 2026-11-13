"""Optional client for a self-hosted FreeLLMAPI OpenAI-compatible endpoint.

This client does not supply a model, API key, or paid fallback. Configure an
exact model ID that is marked free in your own router and whose provider terms
you have checked. Keep all credentials on the server.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def chat(
    messages,
    *,
    task="general",
    model=None,
    api_key=None,
    base_url=None,
    max_tokens=2048,
    temperature=0.2,
    timeout=90,
):
    """Send messages to FreeLLMAPI and return the assistant's text.

    Required environment:
      LEO_FREELLMAPI_BASE_URL (for example http://127.0.0.1:3001/v1)
      LEO_FREELLMAPI_API_KEY
      LEO_FREELLMAPI_MODEL (an explicitly verified free model ID)

    `task` is accepted for API compatibility with Leo's other providers;
    model selection is controlled only by `model` or the environment.
    """
    key = api_key or os.getenv("LEO_FREELLMAPI_API_KEY")
    endpoint = (base_url or os.getenv("LEO_FREELLMAPI_BASE_URL", "")).rstrip("/")
    selected_model = model or os.getenv("LEO_FREELLMAPI_MODEL")

    if not key:
        raise RuntimeError("LEO_FREELLMAPI_API_KEY is not set on the server")
    if not endpoint:
        raise RuntimeError("LEO_FREELLMAPI_BASE_URL is not set")
    if not selected_model:
        raise RuntimeError(
            "Set LEO_FREELLMAPI_MODEL to an exact model ID you verified is free"
        )
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http:// or https:// URL")

    payload = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = Request(
        endpoint + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "Leo AI",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"FreeLLMAPI HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach FreeLLMAPI: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("FreeLLMAPI request timed out") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("FreeLLMAPI returned an unexpected response") from exc

    if not isinstance(content, str):
        raise RuntimeError("FreeLLMAPI returned non-text assistant content")
    return content
