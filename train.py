import argparse
import math
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


# ------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ------------------------------------------------------------
# Transformer Model
# ------------------------------------------------------------
class TranslationModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):

        # Masks BEFORE embedding
        src_key_padding_mask = (src_ids == self.pad_id)
        tgt_key_padding_mask = (tgt_ids == self.pad_id)

        # Embed
        src = self.positional(self.embedding(src_ids) * math.sqrt(self.d_model))
        tgt = self.positional(self.embedding(tgt_ids) * math.sqrt(self.d_model))

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)

        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )

        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.out(out)


# ------------------------------------------------------------
# Warmup LR
# ------------------------------------------------------------
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        factor = min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [(self.d_model ** -0.5) * factor for _ in self.base_lrs]


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, labels):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.labels = labels

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.enc_inputs[idx]),
            "tgt": torch.tensor(self.dec_inputs[idx]),
            "label": torch.tensor(self.labels[idx])
        }


# ------------------------------------------------------------
# Collate
# ------------------------------------------------------------
def collate_fn(batch, pad_id):
    src = nn.utils.rnn.pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=pad_id)
    tgt = nn.utils.rnn.pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=pad_id)
    lbl = nn.utils.rnn.pad_sequence([b["label"] for b in batch], batch_first=True, padding_value=pad_id)
    return src, tgt, lbl


# ------------------------------------------------------------
# Train loop
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    tot = 0

    for src, tgt, lbl in loader:
        src, tgt, lbl = src.to(device), tgt.to(device), lbl.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), lbl[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        tot += loss.item()

    return tot / len(loader)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    pad_id = tokenizer.pad_token_id
    bos = tokenizer.eos_token_id  # Marian uses EOS as BOS also

    ds = load_dataset("wmt17", "de-en", split="train")
    if args.subset_size > 0:
        ds = ds.select(range(args.subset_size))

    def preprocess(batch):
        src_texts = [ex["de"] for ex in batch["translation"]]
        tgt_texts = [ex["en"] for ex in batch["translation"]]
        s = tokenizer(src_texts, truncation=True, max_length=args.max_length)
        t = tokenizer(tgt_texts, truncation=True, max_length=args.max_length)
        return {"src_ids": s["input_ids"], "tgt_ids": t["input_ids"]}

    ds = ds.map(preprocess, batched=True)

    enc_inputs = ds["src_ids"]
    dec_inputs = [[bos] + x for x in ds["tgt_ids"]]
    labels = [x + [bos] for x in ds["tgt_ids"]]

    train_ds = TranslationDataset(enc_inputs, dec_inputs, labels)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))

    model = TranslationModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        pad_id=pad_id
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupScheduler(optimizer, args.d_model, args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, criterion, device)
        losses.append(loss)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
    json.dump({
        **vars(args),
        "pad_id": pad_id,
        "bos_id": bos
    }, open(f"{args.output_dir}/config.json", "w"), indent=2)

    torch.save({"loss": losses}, f"{args.output_dir}/stats.pt")
    print("Training complete.")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    main(args)
