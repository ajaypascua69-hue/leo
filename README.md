# Leo — an independent AI brain

Leo is an original, small decoder-only Transformer trained from random initialization. It does **not** load Gemma, Llama, Gemini, ChatGPT, or any other pretrained weights.

> **Honest status:** this repository is a research prototype and training foundation, not yet a capable general-purpose assistant. A useful language model requires substantial, carefully licensed training text, compute, evaluation, and repeated improvements. The 4 GB RAM phone is suitable for testing small exported models, not for pretraining a strong model.

## What is included

- A byte-level tokenizer (no downloaded vocabulary or pretrained tokenizer).
- A causal Transformer language model initialized from scratch.
- A training loop with validation loss, gradient clipping, checkpoints, and reproducible seeds.
- A text-generation script.
- A dataset intake policy and source register. No dataset is bundled or silently downloaded.

## Quick start (Linux, Windows, or cloud notebook)

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for training.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Leo is intended to learn **English and Filipino/Tagalog** from the beginning. Keep both languages represented in the source collection and record each document's language (`en`, `fil`, `tl`, or `mixed`). Do not assume a dataset is approved just because it is public.\n\nFor a safer corpus workflow, place source text files under `data/` and create `data/manifest.jsonl` using the example at [`examples/manifest.example.jsonl`](examples/manifest.example.jsonl). Review each source and set `approved` to `true` only after checking its exact license, terms, provenance, privacy, and intended use. Then prepare document-level train/validation/test splits (exact duplicates are removed) with:\n\n```bash\npython prepare_data.py --manifest data/manifest.jsonl --data-dir data --out-dir data/processed\n```\n\nThis creates `data/processed/train.txt`, `valid.txt`, `test.txt`, and `corpus_report.json`. Train using the prepared train and validation files. Keep raw source files, manifests, and corpus reports out of public Git unless redistribution is explicitly allowed.

Train a small first model:

```bash
python train.py --train data/processed/train.txt --valid data/processed/valid.txt --steps 5000 --batch-size 16 --block-size 256 --device auto
```

Generate text:

```bash
python generate.py --checkpoint checkpoints/leo.pt --prompt "Once upon a time" --tokens 120
```

Start with a small configuration. Increase model size only after the data pipeline and evaluation are working. Training from scratch on a small corpus will produce limited language ability; more parameters alone do not fix poor data.

## Data and legal safeguards

See [DATA_POLICY.md](DATA_POLICY.md) and [DATA_SOURCES.md](DATA_SOURCES.md). Every dataset must be checked at the individual dataset/version level for:
1. the exact license and whether commercial use is permitted;
2. attribution and ShareAlike obligations;
3. privacy, personal data, and sensitive content;
4. source provenance, quality, duplicates, and contamination;
5. whether redistribution of the raw data is allowed.

A public download URL is **not** proof that content is free to use. Keep a record of the dataset card, license text, version/date, source URL, and preprocessing changes. Do not use scraped books, paid datasets, private messages, or material with unclear rights.

## Roadmap

1. Verify data sources and build a clean, deduplicated corpus.
2. Train a small byte-level baseline and record validation loss.
3. Add held-out tests for factual recall, instruction following, reasoning, and Tagalog/English performance.
4. Improve tokenizer, architecture, data mixture, and training schedule based on measured results.
5. Add retrieval and tool-use as separate modules; these improve usefulness without pretending the base model learned facts it has not seen.
6. Quantize and benchmark an inference build for Android after model quality is adequate.

## Independence and attribution

The model architecture and training code in this repository are original project code. PyTorch is an external training dependency; see its license and the licenses of all transitive dependencies before redistribution. Dataset licenses remain separate from the code license.

This project does not claim to reproduce the capabilities of large commercial models. Progress will be reported using reproducible training settings and held-out evaluations.


## Optional hosted open-weight model providers

Leo's original from-scratch model remains the independent core. Optional hosted
models can be called through the server-side OpenRouter adapter; their weights
are not stored in Git. The initial Apache-2.0 allow-list includes Gemma 4 31B
Instruct for general tasks and Cohere North Mini Code for coding. See
[`MODEL_PROVIDERS.md`](MODEL_PROVIDERS.md) and
[`leo/providers/openrouter.py`](leo/providers/openrouter.py) for setup.

Hosted free endpoints have quotas and availability can change. Configure
`OPENROUTER_API_KEY` as a backend secret only. Never place it in frontend code
or commit it to this repository. This integration is optional and does not
replace Leo's own architecture or training pipeline.


## Optional FreeLLMAPI multi-provider router

Leo can optionally connect to a separately maintained FreeLLMAPI instance through
an OpenAI-compatible client. The upstream project is pinned as a Git submodule at
[`integrations/freellmapi`](integrations/freellmapi), and Leo's small Python
adapter is [`leo/providers/freellmapi.py`](leo/providers/freellmapi.py).
See [`docs/FREELLMAPI_INTEGRATION.md`](docs/FREELLMAPI_INTEGRATION.md) for setup.

This is an optional inference service, not Leo's own model. **Free does not mean
unlimited or guaranteed 24/7:** provider quotas and terms vary, and hosting can
cost money. Use only verified free providers, keep paid providers and paid
fallbacks disabled, and do not purchase the upstream premium catalog. Never
commit API keys or place them in client-side/mobile code.
