"""Connector for local OpenAI-compatible inference servers.

Only loopback endpoints are accepted. This prevents accidentally sending prompts
to a remote host through a connector intended to be local.
"""
import json
import ipaddress
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .contracts import IntegrationManifest, IntegrationResult


class LocalOpenAICompatibleConnector:
    """Call a local OpenAI-compatible chat-completions endpoint.

    Compatible with local runtimes that expose /v1/chat/completions, for example
    LM Studio or an Ollama server configured with its OpenAI-compatible API.
    No model is bundled or downloaded by this connector.
    """

    manifest = IntegrationManifest(
        integration_id="leo.local_openai_compatible",
        display_name="Local OpenAI-compatible model",
        provider_type="local",
        capabilities=("chat_completion",),
        requires_network=True,
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "local-model",
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 512,
        api_key: str | None = None,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("local endpoint must use http and include a hostname")
        host = parsed.hostname.lower()
        is_loopback = host == "localhost"
        try:
            is_loopback = is_loopback or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if not is_loopback:
            raise ValueError("local connector only accepts localhost/loopback endpoints")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain credentials, query, or fragment")
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0 or max_output_tokens < 1:
            raise ValueError("timeout and output token limit must be positive")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key

    def invoke(self, input_text: str) -> IntegrationResult:
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text must be a non-empty string")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": input_text}],
            "temperature": 0.7,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            raise RuntimeError(f"local model server returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("could not reach the local model server") from exc

        if len(raw) > 2_000_000:
            raise RuntimeError("local model response exceeded the 2 MB safety limit")
        try:
            data = json.loads(raw.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("local model server returned an invalid chat-completions response") from exc
        if not isinstance(content, str):
            raise RuntimeError("local model response content was not text")

        return IntegrationResult(
            integration_id=self.manifest.integration_id,
            content=content,
            metadata={"provider": "local_openai_compatible", "model": self.model},
        )
