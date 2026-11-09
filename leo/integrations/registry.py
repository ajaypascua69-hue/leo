"""Permission-gated registry for optional Leo integrations."""
from .contracts import Integration, IntegrationResult


class IntegrationPermissionError(PermissionError):
    """Raised when a connector has not been explicitly authorized."""


class IntegrationRegistry:
    """Stores adapters without enabling them or granting them implicit access.

    The caller must separately:
      1. allow-list the integration ID,
      2. confirm this particular invocation, and
      3. enable network access for network-requiring integrations.

    The registry never passes chat history, files, credentials, or model weights
    to an adapter. Only the input_text argument is provided.
    """

    def __init__(self):
        self._integrations: dict[str, Integration] = {}

    def register(self, integration: Integration) -> None:
        manifest = integration.manifest
        integration_id = manifest.integration_id
        if integration_id in self._integrations:
            raise ValueError(f"integration already registered: {integration_id}")
        self._integrations[integration_id] = integration

    def available(self) -> tuple[dict[str, object], ...]:
        """Return public metadata only; never expose adapter secrets."""
        return tuple(
            {
                "integration_id": item.manifest.integration_id,
                "display_name": item.manifest.display_name,
                "provider_type": item.manifest.provider_type,
                "capabilities": item.manifest.capabilities,
                "requires_network": item.manifest.requires_network,
            }
            for item in self._integrations.values()
        )

    def invoke(
        self,
        integration_id: str,
        input_text: str,
        *,
        allowed_integrations: set[str] | frozenset[str] = frozenset(),
        user_confirmed: bool = False,
        network_enabled: bool = False,
    ) -> IntegrationResult:
        if integration_id not in self._integrations:
            raise KeyError(f"unknown integration: {integration_id}")
        if integration_id not in allowed_integrations:
            raise IntegrationPermissionError("integration is not on the user's allow-list")
        if not user_confirmed:
            raise IntegrationPermissionError("this invocation needs user confirmation")

        adapter = self._integrations[integration_id]
        if adapter.manifest.requires_network and not network_enabled:
            raise IntegrationPermissionError("network access is disabled")

        if not isinstance(input_text, str):
            raise TypeError("input_text must be a string")
        if not input_text.strip():
            raise ValueError("input_text must not be empty")

        result = adapter.invoke(input_text)
        if result.integration_id != integration_id:
            raise ValueError("adapter returned a mismatched integration ID")
        return result
