"""Optional integrations kept separate from Leo's independently trained model core."""

from .contracts import Integration, IntegrationManifest, IntegrationResult
from .registry import IntegrationRegistry, IntegrationPermissionError

__all__ = [
    "Integration",
    "IntegrationManifest",
    "IntegrationResult",
    "IntegrationRegistry",
    "IntegrationPermissionError",
]
