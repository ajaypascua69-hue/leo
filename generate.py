"""Generate text from a Leo checkpoint trained by train.py."""
import argparse
import torch
from leo.model import LeoConfig, LeoLanguageModel
from leo.tokenizer import encode, decode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/leo.pt")
    p.add_argument("--prompt", default="Hello, I am Leo.")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = LeoConfig(**state["config"])
    model = LeoLanguageModel(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    prompt = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    result = model.generate(prompt, args.tokens, args.temperature, args.top_k)
    print(decode(result[0].tolist()))

if __name__ == "__main__":
    main()
