import torch
import json
import math
import torch.nn as nn
from transformers import AutoTokenizer
from train import TranslationModel


def greedy_decode(model, src_ids, tokenizer, bos_id, eos_id, max_len=60, device="cpu"):
    model.eval()

    # Encode
    with torch.no_grad():
        src_embed = model.embedding(src_ids) * math.sqrt(model.d_model)
        src_embed = model.positional(src_embed)
        memory = model.transformer.encoder(src_embed) 

    # Start with BOS
    ys = torch.tensor([[bos_id]], device=device)

    for _ in range(max_len):
        tgt_embed = model.embedding(ys) * math.sqrt(model.d_model)
        tgt_embed = model.positional(tgt_embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            ys.size(1)
        ).to(device)

        with torch.no_grad():
            out = model.transformer.decoder(
                tgt_embed,
                memory,
                tgt_mask=tgt_mask
            )
            logits = model.out(out[:, -1, :])
            next_tok = torch.argmax(logits, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)

        if next_tok == eos_id:
            break

    return tokenizer.decode(ys[0], skip_special_tokens=True)


def translate(model_dir, sentence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    # Load config
    with open(f"{model_dir}/config.json", "r") as f:
        cfg = json.load(f)

    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]
    pad_id = cfg["pad_id"]
    bos_id = cfg["bos_id"]
    eos_id = tokenizer.eos_token_id

    # Build model
    model = TranslationModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        pad_id=pad_id
    ).to(device)

    # Load weights
    state = torch.load(f"{model_dir}/model.pt", map_location=device)
    model.load_state_dict(state)

    # Tokenize input
    src_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)

    # Translate
    return greedy_decode(model, src_ids, tokenizer, bos_id, eos_id, device=device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate German → English")
    parser.add_argument("--model_dir", type=str, default="output",
                        help="Directory containing model.pt and config.json")
    parser.add_argument("--text", type=str, required=True,
                        help="German text to translate")

    args = parser.parse_args()

    result = translate(args.model_dir, args.text)
    print("Translation:", result)
