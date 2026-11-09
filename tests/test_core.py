import json
from pathlib import Path

import torch

from leo.tokenizer import encode, decode
from leo.model import LeoConfig, LeoLanguageModel
from prepare_data import is_language_tag, load_documents, render_document


def test_byte_tokenizer_round_trip():
    text = "Leo — hello, नमस्ते! こんにちは 안녕하세요 مرحبا"
    assert decode(encode(text)) == text


def test_language_tags_accept_many_scripts_and_regions():
    for tag in ("en", "fil", "tl", "ja", "ko", "es", "pt-BR", "fr", "de", "zh-Hans", "sr-Latn", "ar", "hi", "id", "mixed"):
        assert is_language_tag(tag)
    for tag in ("", "english", "en_US", "pt-BR-extra-tag-too-long"):
        assert not is_language_tag(tag)


def test_manifest_loads_multilingual_metadata_and_renders_language_marker(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    for i, (lang, sample) in enumerate((("en", "Hello"), ("ja", "こんにちは"), ("fil", "Kamusta"))):
        rel = f"source_{i}.txt"
        (data_dir / rel).write_text(sample, encoding="utf-8")
        rows.append({
            "path": rel,
            "approved": True,
            "license": "CC0-1.0",
            "license_url": "https://example.org/license",
            "source_url": f"https://example.org/{i}",
            "language": lang,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    records = load_documents(manifest, data_dir)
    assert [record["language"] for record in records] == ["en", "ja", "fil"]
    assert render_document(records[1]).startswith("[LEO_LANGUAGE=ja]\n")


def test_model_logits_and_loss():
    config = LeoConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = LeoLanguageModel(config)
    x = torch.randint(0, 256, (2, 12))
    y = torch.randint(0, 256, (2, 12))
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, 256)
    assert loss is not None and torch.isfinite(loss)


def test_causal_model_generation_length():
    config = LeoConfig(block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = LeoLanguageModel(config)
    prompt = torch.tensor([[72, 105]])
    result = model.generate(prompt, max_new_tokens=3, top_k=5)
    assert result.shape == (1, 5)
