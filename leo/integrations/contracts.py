"""Stable contracts for optional local and external integrations.

Integrations are tools around Leo, not replacements for Leo's model or weights.
They receive only the explicit input supplied by the caller.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class IntegrationManifest:
    integration_id: str
    display_name: str
    provider_type: str
    capabilities: tuple[str, ...]
    requires_network: bool = False

    def __post_init__(self):
        if not self.integration_id or any(ch.isspace() for ch in self.integration_id):
            raise ValueError("integration_id must be a non-empty token")
        if self.provider_type not in {"local", "external_ai", "utility"}:
            raise ValueError("provider_type must be local, external_ai, or utility")
        if not self.capabilities:
            raise ValueError("an integration must declare at least one capability")
        if self.provider_type == "external_ai" and not self.requires_network:
            raise ValueError("external_ai integrations must declare network access")


@dataclass(frozen=True)
class IntegrationResult:
    integration_id: str
    content: str
    metadata: dict[str, str] | None = None


@runtime_checkable
class Integration(Protocol):
    """An adapter that performs one explicitly requested, bounded operation."""

    @property
    def manifest(self) -> IntegrationManifest:
        ...

    def invoke(self, input_text: str) -> IntegrationResult:
        ...
