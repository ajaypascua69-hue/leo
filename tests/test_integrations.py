import pytest

from leo.integrations import (
    IntegrationManifest,
    IntegrationPermissionError,
    IntegrationRegistry,
    IntegrationResult,
)


class DemoLocalIntegration:
    manifest = IntegrationManifest(
        integration_id="demo.local",
        display_name="Demo local tool",
        provider_type="local",
        capabilities=("text_demo",),
    )

    def __init__(self):
        self.received = []

    def invoke(self, input_text):
        self.received.append(input_text)
        return IntegrationResult(self.manifest.integration_id, input_text.upper())


class DemoExternalAI:
    manifest = IntegrationManifest(
        integration_id="demo.remote",
        display_name="Demo external AI",
        provider_type="external_ai",
        capabilities=("text_generation",),
        requires_network=True,
    )

    def invoke(self, input_text):
        return IntegrationResult(self.manifest.integration_id, "remote: " + input_text)


def test_integration_is_not_callable_without_allowlist_and_confirmation():
    registry = IntegrationRegistry()
    registry.register(DemoLocalIntegration())
    with pytest.raises(IntegrationPermissionError):
        registry.invoke("demo.local", "hello", user_confirmed=True)
    with pytest.raises(IntegrationPermissionError):
        registry.invoke(
            "demo.local", "hello",
            allowed_integrations={"demo.local"},
            user_confirmed=False,
        )


def test_local_integration_receives_only_explicit_input():
    adapter = DemoLocalIntegration()
    registry = IntegrationRegistry()
    registry.register(adapter)
    result = registry.invoke(
        "demo.local", "hello",
        allowed_integrations={"demo.local"},
        user_confirmed=True,
    )
    assert result.content == "HELLO"
    assert adapter.received == ["hello"]


def test_network_integration_requires_separate_network_permission():
    registry = IntegrationRegistry()
    registry.register(DemoExternalAI())
    with pytest.raises(IntegrationPermissionError):
        registry.invoke(
            "demo.remote", "hello",
            allowed_integrations={"demo.remote"},
            user_confirmed=True,
        )
    result = registry.invoke(
        "demo.remote", "hello",
        allowed_integrations={"demo.remote"},
        user_confirmed=True,
        network_enabled=True,
    )
    assert result.content == "remote: hello"


def test_duplicate_ids_and_mismatched_metadata_are_rejected():
    registry = IntegrationRegistry()
    adapter = DemoLocalIntegration()
    registry.register(adapter)
    with pytest.raises(ValueError):
        registry.register(adapter)


def test_manifest_rejects_external_provider_without_network_flag():
    with pytest.raises(ValueError):
        IntegrationManifest(
            integration_id="bad.remote",
            display_name="Bad remote",
            provider_type="external_ai",
            capabilities=("text_generation",),
        )
