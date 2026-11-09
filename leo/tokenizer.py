"""UTF-8 byte tokenizer; no external or pretrained vocabulary."""
from typing import List

VOCAB_SIZE = 256

def encode(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="replace"))

def decode(tokens) -> str:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return bytes(int(t) & 255 for t in tokens).decode("utf-8", errors="replace")
