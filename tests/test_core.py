import torch
from leo.tokenizer import encode, decode
from leo.model import LeoConfig, LeoLanguageModel

def test_byte_tokenizer_round_trip():
    text = "Leo — hello, नमस्ते!"
    assert decode(encode(text)) == text

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
