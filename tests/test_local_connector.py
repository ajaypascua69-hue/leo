import json
from urllib.request import Request

import pytest

from leo.integrations import (
    IntegrationPermissionError,
    IntegrationRegistry,
    LocalOpenAICompatibleConnector,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.payload


def test_local_connector_rejects_remote_hosts():
    with pytest.raises(ValueError, match="localhost/loopback"):
        LocalOpenAICompatibleConnector("https://example.com/v1")


def test_local_connector_posts_only_explicit_prompt(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            json.dumps({
                "choices": [{"message": {"content": "Hello from the local model"}}]
            }).encode("utf-8")
        )

    monkeypatch.setattr("leo.integrations.openai_compatible_local.urlopen", fake_urlopen)
    connector = LocalOpenAICompatibleConnector()
    result = connector.invoke("Hello Leo")

    assert result.content == "Hello from the local model"
    assert seen["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hello Leo"}]
    assert result.metadata["provider"] == "local_openai_compatible"


def test_registry_requires_allowlist_confirmation_and_network_permission(monkeypatch):
    connector = LocalOpenAICompatibleConnector()
    registry = IntegrationRegistry()
    registry.register(connector)

    with pytest.raises(IntegrationPermissionError):
        registry.invoke(
            connector.manifest.integration_id,
            "test",
            allowed_integrations={connector.manifest.integration_id},
            user_confirmed=True,
        )

    def fake_urlopen(request, timeout):
        return FakeResponse(
            b'{"choices":[{"message":{"content":"local reply"}}]}'
        )

    monkeypatch.setattr("leo.integrations.openai_compatible_local.urlopen", fake_urlopen)
    result = registry.invoke(
        connector.manifest.integration_id,
        "test",
        allowed_integrations={connector.manifest.integration_id},
        user_confirmed=True,
        network_enabled=True,
    )
    assert result.content == "local reply"


def test_connector_rejects_empty_prompt():
    connector = LocalOpenAICompatibleConnector()
    with pytest.raises(ValueError):
        connector.invoke("   ")
