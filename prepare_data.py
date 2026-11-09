"""Prepare a provenance-checked, document-level corpus for Leo.

Input:
  data/manifest.jsonl
Each line is JSON with: path, approved, license, license_url, source_url, language.
Only explicitly approved entries with traceable license/source URLs are included.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ALLOWED_LANGUAGES = {"en", "fil", "tl", "mixed"}

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_documents(manifest_path: Path, root: Path):
    records = []
    seen = set()
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on manifest line {line_no}: {exc}") from exc
        required = ("path", "approved", "license", "license_url", "source_url", "language")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Manifest line {line_no} missing: {', '.join(missing)}")
        if item["approved"] is not True:
            continue
        if not all(isinstance(item[k], str) and item[k].strip() for k in
                   ("path", "license", "license_url", "source_url", "language")):
            raise ValueError(f"Manifest line {line_no}: required metadata must be non-empty strings")
        if item["language"] not in ALLOWED_LANGUAGES:
            raise ValueError(f"Manifest line {line_no}: language must be one of {sorted(ALLOWED_LANGUAGES)}")
        source = (root / item["path"]).resolve()
        try:
            source.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Manifest line {line_no}: path escapes data directory") from exc
        if not source.is_file():
            raise FileNotFoundError(f"Manifest line {line_no}: file not found: {source}")
        raw = source.read_bytes()
        digest = sha256(raw)
        if digest in seen:
            continue
        seen.add(digest)
        text = raw.decode("utf-8", errors="strict").strip()
        if text:
            records.append({
                "path": item["path"],
                "text": text,
                "sha256": digest,
                "license": item["license"],
                "license_url": item["license_url"],
                "source_url": item["source_url"],
                "language": item["language"],
            })
    if len(records) < 3:
        raise ValueError("Need at least 3 unique approved non-empty documents for train/valid/test splits")
    return records

def split_documents(records, seed, valid_fraction, test_fraction):
    if valid_fraction <= 0 or test_fraction <= 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("valid/test fractions must be positive and sum to less than 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_valid = max(1, round(n * valid_fraction))
    n_test = max(1, round(n * test_fraction))
    if n_valid + n_test >= n:
        raise ValueError("Not enough documents for requested split fractions")
    return shuffled[n_valid+n_test:], shuffled[:n_valid], shuffled[n_valid:n_valid+n_test]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(args.data_dir)
    records = load_documents(Path(args.manifest), root)
    train, valid, test = split_documents(records, args.seed, args.valid_fraction, args.test_fraction)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, split in (("train", train), ("valid", valid), ("test", test)):
        (out / f"{name}.txt").write_text("\n\n".join(x["text"] for x in split) + "\n", encoding="utf-8")

    report = {
        "seed": args.seed,
        "documents": len(records),
        "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
        "items": [{k: v for k, v in x.items() if k != "text"} for x in records],
        "note": "Metadata records are not a legal determination. Human review is required before setting approved=true."
    }
    (out / "corpus_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} unique approved documents: train={len(train)}, valid={len(valid)}, test={len(test)}")
    print(f"Outputs: {out.resolve()}")

if __name__ == "__main__":
    main()
