# Leo — an independent, scalable AI brain

Leo is an original AI research project. Its model weights are initialized randomly and trained from scratch. The project does **not** use pretrained Gemma, Llama, Qwen, Gemini, ChatGPT, or other model weights as its foundation.

> **Honest status:** Leo is an early research prototype, not yet a capable general-purpose assistant. The current compact Transformer is a baseline for experiments. It has not completed a meaningful large-scale pretraining run. Model quality depends on data quality, compute, architecture, training, and evaluation—not on branding or ambition.

## Project mission

Build one independent general-purpose AI brain that can grow across hardware generations and support:

- Multilingual understanding and generation, with a long-term target of 50+ languages.
- Conversation, writing, translation, reasoning, mathematics, and learning.
- Coding and software development, including ordinary consumer AI applications—not only developer tools.
- Document and data analysis.
- Tool use, retrieval, and task execution through separately engineered, permission-controlled systems.
- Mobile, desktop, and laptop deployment.

The initial language priorities include English and Filipino/Tagalog, alongside Japanese, Korean, Spanish, Portuguese, French, German, Chinese, Arabic, Hindi, Indonesian, and additional languages. This is a roadmap, not a claim that the current prototype already understands them.

## Hardware independence

The model design must not be capped by the owner's current 4 GB RAM phone.

- **Small devices:** later, offer carefully measured quantized builds and short-context settings.
- **More capable phones/tablets:** offer larger compatible Leo variants where memory and runtime permit.
- **Laptops/desktops/workstations:** support larger models, longer contexts, development tools, and higher-throughput inference.
- **Training:** use available free compute when possible; never assume the phone is the pretraining machine.

A small deployment is a resource-scaled build, not the definition or maximum size of Leo. Preserve architecture/configuration metadata and versioned checkpoint formats so future releases can support larger variants. Compatibility and export must be tested on each target runtime.

## Current research prototype

- A byte-level UTF-8 tokenizer baseline. It can represent any Unicode script without downloading a pretrained vocabulary, but is inefficient for long multilingual sequences.
- A compact causal Transformer initialized from scratch.
- A training loop with validation loss, gradient clipping, checkpoints, and reproducible seeds.
- A generation script.
- A provenance-aware data intake workflow and tests.

The byte tokenizer is deliberately a baseline. A future tokenizer should be trained from scratch on the approved multilingual corpus, with Unicode/script coverage tests; it must not import a pretrained vocabulary or model tokenizer.

## Quick start (Linux, Windows, macOS, or cloud notebook)

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for meaningful training.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Place reviewed source text files under `data/` and prepare `data/manifest.jsonl` using `examples/manifest.example.jsonl`. Each document must have a language tag (ISO 639-1/639-2/639-3 code or BCP 47 form such as `en`, `fil`, `ja`, `ko`, `pt-BR`, `zh-Hans`, or `sr-Latn`; use `mixed` for genuinely mixed-language documents). Set `approved` to `true` only after checking the exact dataset/version license, terms, provenance, privacy, and intended use.

```bash
python prepare_data.py --manifest data/manifest.jsonl --data-dir data --out-dir data/processed
```

The script creates train/validation/test text files and a corpus report with per-language counts. Each document is prefixed with a readable language marker to make its declared language visible during training. Keep raw source files, manifests, and reports out of public Git unless redistribution is explicitly allowed.

Train a small baseline:

```bash
python train.py --train data/processed/train.txt --valid data/processed/valid.txt --steps 5000 --batch-size 16 --block-size 256 --device auto
```

Generate text:

```bash
python generate.py --checkpoint checkpoints/leo.pt --prompt "Hello, I am Leo." --tokens 120
```

Start with a small configuration. Training from scratch on a small corpus produces limited ability; more parameters alone do not fix poor data or insufficient training.

## Data and legal safeguards

See [DATA_POLICY.md](DATA_POLICY.md) and [DATA_SOURCES.md](DATA_SOURCES.md). A public URL is not proof that content is free to use. Review each exact dataset revision, license, attribution and ShareAlike terms, privacy implications, source provenance, and intended deployment. Do not ingest arbitrary web crawls, private conversations, paid datasets, or material with unclear rights. Keep a record of licenses and preprocessing. Obtain qualified legal review before commercial release where needed.

## Development roadmap

1. Expand the data manifest and quality controls to handle many languages without assuming every language is equally represented.
2. Build a legally reviewed, balanced, deduplicated multilingual corpus with held-out evaluation data.
3. Train and evaluate a small random-init baseline; publish actual loss curves and language-specific results.
4. Develop a multilingual tokenizer from scratch and compare it fairly with the byte baseline.
5. Improve architecture, data mixture, training efficiency, and evaluation based on measured evidence.
6. Add reasoning, math, coding, instruction-following, retrieval, and permission-controlled tools as measurable capabilities.
7. Build shared inference interfaces and separate Android, desktop, and laptop user interfaces.
8. Add quantization and platform runtimes only after quality and compatibility tests.

## Independence and attribution

The model architecture and training code in this repository are original project code. PyTorch and other dependencies are external software; review their licenses before redistribution. Dataset rights are separate from code rights. Do not use another pretrained model's weights or tokenizer as Leo's foundation.

Leo's goal is ambitious, but claims of capability must follow reproducible training runs and held-out evaluations.
