"""Evaluate a Leo checkpoint on a held-out UTF-8 text file."""
import argparse
import torch
from leo.model import LeoConfig, LeoLanguageModel
from leo.tokenizer import encode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True, help="Held-out plain-text evaluation file")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = LeoConfig(**state["config"])
    model = LeoLanguageModel(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    with open(args.text, "r", encoding="utf-8") as f:
        data = torch.tensor(encode(f.read()), dtype=torch.long)
    size = config.block_size
    if len(data) <= size:
        raise ValueError("Evaluation text must be longer than block_size")
    losses = []
    with torch.no_grad():
        for start in range(0, len(data) - size - 1, size):
            x = data[start:start+size].unsqueeze(0).to(device)
            y = data[start+1:start+size+1].unsqueeze(0).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    print(f"validation_loss={mean_loss:.4f}")
    print(f"perplexity={torch.exp(torch.tensor(mean_loss)).item():.4f}")

if __name__ == "__main__":
    main()
