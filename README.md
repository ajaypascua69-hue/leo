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

Put approved plain-text training material in `data/train.txt` and validation material in `data/valid.txt`. Keep raw source files and license records outside Git unless redistribution is explicitly allowed.

Train a small first model:

```bash
python train.py --train data/train.txt --valid data/valid.txt --steps 5000 --batch-size 16 --block-size 256 --device auto
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
