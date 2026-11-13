# FreeLLMAPI integration for Leo

Leo keeps its original from-scratch Transformer and training pipeline. FreeLLMAPI is an optional, separately maintained OpenAI-compatible inference router; it does not become Leo's model weights or replace Leo's core.

## Included

- `integrations/freellmapi` is a Git submodule pinned to the upstream commit recorded in this repository. It preserves upstream history and its MIT license without copying hundreds of files into Leo's own source tree.
- `leo/providers/freellmapi.py` is a small Python client for the router's OpenAI-compatible `/v1/chat/completions` endpoint.
- This guide documents the free-only setup safeguards.

Upstream: https://github.com/tashfeenahmed/freellmapi

## Cost and provider safeguards

**This integration does not make inference or hosting universally free.** Upstream providers set their own quotas, availability, privacy terms, and usage conditions. Some providers or features may be paid. The upstream project also advertises a paid live-catalog option; Leo does not require it.

To keep this setup from intentionally using paid services:

1. Run FreeLLMAPI locally or on infrastructure whose zero-cost terms you have verified. Do not add a payment card or enable a paid hosting plan.
2. In the FreeLLMAPI dashboard, add only provider credentials whose current terms explicitly show a free tier you are eligible to use.
3. Do not add paid-provider keys, paid custom endpoints, or paid fallback chains. Disable/remove any paid providers from routing.
4. Use only models/providers visibly marked free in the current catalog, and re-check their quotas and terms before use.
5. Do not buy the optional premium/live catalog. The free install's catalog snapshot is sufficient for testing.
6. Expect rate limits, outages, changing quotas, and possible pauses. No free tier guarantees 24/7 uptime or unlimited tokens.

If you cannot verify that a provider or deployment is free, leave it disabled. A free API key does not guarantee that every model exposed by a router is free.

## Configure Leo's client

Set these variables only in the backend/server environment:

- `LEO_FREELLMAPI_BASE_URL`: base URL ending in `/v1`, for example `http://127.0.0.1:3001/v1` for a local router.
- `LEO_FREELLMAPI_API_KEY`: the unified key created in your FreeLLMAPI instance.
- `LEO_FREELLMAPI_MODEL`: an exact model ID that you verified is free and enabled in your router.

Then call:

```python
from leo.providers.freellmapi import chat

answer = chat(
    [{"role": "user", "content": "Help me debug this Python function."}],
    task="coding",
)
print(answer)
```

Or pass an explicit verified model ID:

```python
answer = chat(
    [{"role": "user", "content": "Explain this code."}],
    model="YOUR_VERIFIED_FREE_MODEL_ID",
)
```

The client deliberately has no hard-coded model ID and no automatic paid fallback. It sends only the model you explicitly configure or pass. FreeLLMAPI's own router configuration must still be restricted to free providers.

## Security

- Keep the unified key in a server-side secret store or an untracked local environment file.
- Never put it in a browser bundle, Android APK, GitHub commit, public issue, or log.
- Do not send private or sensitive user content to an upstream provider unless you have reviewed its privacy and retention terms.
- Put authentication, per-user quotas, request-size limits, and abuse protection in front of any publicly reachable router.

## Updating the upstream submodule

The submodule is pinned so Leo's build does not silently change when upstream changes. To inspect it after cloning:

```bash
git submodule update --init --recursive
```

Review upstream releases, dependencies, security advisories, and provider terms before updating the pinned commit.

## License and attribution

FreeLLMAPI is distributed under the MIT License. Its copyright and license notice are retained in the submodule. Leo's own files remain governed by Leo's repository license and notices.
