# Approved open-weight model integrations

Leo's original model and training pipeline remain independent. These are optional
hosted inference connections; **the weights are not copied into this repository**.

The initial allow-list uses model endpoints whose published terms identify a
permissive license:

| Task | Model | License | OpenRouter model ID |
|---|---|---|---|
| General assistant, reasoning, multilingual and multimodal input | Google Gemma 4 31B Instruct | Apache-2.0 | `google/gemma-4-31b-it:free` |
| Coding and software-engineering tasks | Cohere North Mini Code | Apache-2.0 | `cohere/north-mini-code:free` |

Sources checked 10 November 2026:
- [Gemma 4 31B (free) model and license](https://openrouter.ai/google/gemma-4-31b-it:free)
- [North Mini Code (free) model and license](https://openrouter.ai/cohere/north-mini-code:free)
- [OpenRouter free-model catalogue and current limits](https://openrouter.ai/collections/free-models)

## Important distinctions

- "Open-weight" does not automatically mean fully open-source. Leo only enables
  models here when the listed model endpoint publishes a permissive license.
- Free API endpoints are quota-limited and can change availability. They are
  suitable for development and testing, not a guarantee of unlimited 24/7 use.
- The provider receives the prompts sent to its endpoint. Review its privacy and
  retention terms before sending user data.
- Model terms apply to the model weights; OpenRouter terms apply to API access.
- Do not claim that Leo was trained on these models. They are optional external
  inference engines, not Leo's own trained weights.

## Configure

1. Create an OpenRouter API key.
2. Set `OPENROUTER_API_KEY` as a secret in the backend/server environment.
3. Optionally set `LEO_PUBLIC_URL` to the deployed Leo website URL.
4. Call `leo.providers.openrouter.chat(messages, task="general")` or use
   `task="coding"` for software-development requests.

Example:

```python
from leo.providers.openrouter import chat

answer = chat(
    [{"role": "user", "content": "Explain recursion with a Python example."}],
    task="coding",
)
print(answer)
```

Never expose the API key in browser code, mobile apps, GitHub commits, or logs.
The backend must authenticate Leo users, enforce per-user rate limits and usage
budgets, validate message sizes, and handle provider errors before going public.
