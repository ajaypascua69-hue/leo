"""Train Leo from approved local plain-text files; no pretrained weights are loaded."""
import argparse
import os
import random
import torch
from leo.model import LeoConfig, LeoLanguageModel
from leo.tokenizer import encode

def read_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        values = encode(f.read())
    if len(values) < 2:
        raise ValueError(f"{path} must contain at least two UTF-8 bytes")
    return torch.tensor(values, dtype=torch.long)

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("Dataset is too short for block_size; reduce block_size.")
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, valid_data, batch_size, block_size, device, batches=20):
    model.eval()
    scores = {}
    for name, data in (("train", train_data), ("valid", valid_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        scores[name] = sum(losses) / len(losses)
    model.train()
    return scores

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.txt")
    p.add_argument("--valid", default="data/valid.txt")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", default="checkpoints/leo.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_data, valid_data = read_tokens(args.train), read_tokens(args.valid)
    config = LeoConfig(block_size=args.block_size, n_layer=args.layers,
                       n_head=args.heads, n_embd=args.width)
    model = LeoLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = estimate_loss(model, train_data, valid_data, args.batch_size,
                                     args.block_size, device)
            print(f"step {step}: train_loss={metrics['train']:.4f} valid_loss={metrics['valid']:.4f}")
            torch.save({"model": model.state_dict(), "config": vars(config),
                        "step": step, "seed": args.seed}, args.out)
    print(f"Saved checkpoint: {args.out}")

if __name__ == "__main__":
    main()
