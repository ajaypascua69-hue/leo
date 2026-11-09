# Leo optional AI and tool integrations

## Design rule

Leo's own model remains the independent brain: original architecture, randomly
initialized weights, and training from scratch. Integrations are optional
services around that core. Connecting an external AI service must never replace
Leo's model, silently fine-tune it, or make the external service mandatory.

## What this first implementation provides

The first integration layer adds:
- A stable manifest describing each connector and its capabilities.
- A registry for local tools, utility tools, and external AI services.
- A user allow-list for each integration.
- Per-invocation confirmation.
- A separate network permission for connectors that need internet access.
- A narrow input contract: an adapter receives only the text explicitly passed
  to that invocation, not Leo's full conversation, files, secrets, or weights.

No third-party AI provider is enabled by default. The included tests use mock
adapters and make no network requests.

## Connecting services later

A connector can be written for:
- Local AI runtimes that run on the user's computer or phone.
- External AI APIs, if the user supplies their own credentials and approves
  possible provider charges.
- Speech-to-text and text-to-speech engines.
- Translation, OCR, vision, retrieval, and other specialized tools.

Each connector must declare its capabilities, whether it needs network access,
and its provider type. Credentials must be supplied at runtime through a
secure secret store or environment configuration; never commit API keys,
passwords, tokens, or private chat logs to Git.

## Free-first policy

The connector framework itself uses Python's standard library and has no
mandatory paid service or API dependency. Local integrations can be free when
the runtime and model are freely available and compatible with the device.
External APIs may have free tiers, limits, or charges; Leo must show the user
which provider is being called and must not assume an API is free.

## Current limitations

This is the integration foundation, not a finished app marketplace or a set of
working provider connectors. It does not yet connect to named AI apps, manage
credentials, perform network calls, or expose a user interface. Those should be
added one connector at a time with tests and explicit permission screens.

## Example

```python
from leo.integrations import (
    IntegrationManifest, IntegrationRegistry, IntegrationResult
)

class MyLocalTool:
    manifest = IntegrationManifest(
        integration_id="my.local_tool",
        display_name="My local tool",
        provider_type="local",
        capabilities=("text_processing",),
    )

    def invoke(self, input_text):
        return IntegrationResult("my.local_tool", input_text.upper())

registry = IntegrationRegistry()
registry.register(MyLocalTool())

result = registry.invoke(
    "my.local_tool",
    "hello Leo",
    allowed_integrations={"my.local_tool"},
    user_confirmed=True,
)
print(result.content)
```

This example demonstrates the permission contract only; it does not connect an
external provider.
