"""Convert reviewed teacher-generated lessons into Leo's existing corpus manifest.

Input is JSONL. Every accepted row must be explicitly reviewed and approved.
This tool records provenance; it does not determine whether a license permits use.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

LANGUAGE_TAG_RE = re.compile(r"^(?:mixed|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$", re.IGNORECASE)


def required_text(item, key, line_no):
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_no}: '{key}' must be a non-empty string")
    return value.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="teacher_lessons.jsonl")
    parser.add_argument("--data-dir", default="data/teacher_lessons")
    parser.add_argument("--manifest", default="data/teacher_manifest.jsonl")
    args = parser.parse_args()

    source = Path(args.input)
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    accepted, seen = [], set()
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        if item.get("reviewed") is not True or item.get("approved_for_training") is not True:
            continue

        language = required_text(item, "language", line_no)
        if not LANGUAGE_TAG_RE.fullmatch(language):
            raise ValueError(f"Line {line_no}: invalid language tag {language!r}")
        topic = required_text(item, "topic", line_no)
        question = required_text(item, "question", line_no)
        answer = required_text(item, "answer", line_no)
        license_name = required_text(item, "license", line_no)
        license_url = required_text(item, "license_url", line_no)
        source_url = required_text(item, "source_url", line_no)

        # A plain-text instruction/answer format for next-token training.
        document = (
            f"[LEO_LANGUAGE={language}]\n"
            f"[LEO_TOPIC={topic}]\n"
            f"User: {question}\n"
            f"Leo: {answer}\n"
        )
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        filename = f"lesson_{digest[:16]}.txt"
        (data_dir / filename).write_text(document, encoding="utf-8")
        accepted.append({
            "path": str(Path(args.data_dir) / filename),
            "approved": True,
            "license": license_name,
            "license_url": license_url,
            "source_url": source_url,
            "language": language,
        })

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Prepared {len(accepted)} unique reviewed lessons.")
    print(f"Manifest: {manifest_path}")
    print("Next: run prepare_data.py with this manifest and its data directory.")
    print("Human review of rights, accuracy, privacy, and safety remains required.")


if __name__ == "__main__":
    main()
