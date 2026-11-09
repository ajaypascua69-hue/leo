"""OpenRouter client for explicitly allow-listed open-weight models.

This is an optional hosted-inference bridge. It does not download model weights
or replace Leo's independently trained model.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Only models with a clearly stated permissive license are enabled here.
# Free endpoint availability and quotas can change; verify before production use.
MODELS = {
    "general": {
        "id": "google/gemma-4-31b-it:free",
        "name": "Gemma 4 31B Instruct",
        "license": "Apache-2.0",
    },
    "coding": {
        "id": "cohere/north-mini-code:free",
        "name": "North Mini Code",
        "license": "Apache-2.0",
    },
}

def model_for_task(task: str = "general") -> str:
    """Return the configured open-weight model ID for a task."""
    key = "coding" if task.lower() in {"code", "coding", "programming", "development"} else "general"
    return MODELS[key]["id"]


def chat(messages, *, task="general", model=None, api_key=None,
         max_tokens=2048, temperature=0.2, timeout=90):
    """Send chat messages to OpenRouter and return the assistant text.

    Set OPENROUTER_API_KEY in the server environment. Never put API keys in
    source code, frontend JavaScript, committed .env files, or client apps.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set on the server")

    selected_model = model or model_for_task(task)
    allowed_ids = {entry["id"] for entry in MODELS.values()}
    if selected_model not in allowed_ids:
        raise ValueError("Model is not in Leo's open-weight allow-list")

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    payload = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("LEO_PUBLIC_URL", "https://github.com/ajaypascua69-hue/leo"),
            "X-Title": "Leo AI",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("OpenRouter request timed out") from exc

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected response") from exc
