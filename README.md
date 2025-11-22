
# **Translation Transformer (WMT17: German → English)**

This repository contains a minimal yet complete implementation of a Transformer-based translation model, following the architecture described in **“Attention Is All You Need”**.

It includes:

### Custom attention components (implemented from scratch)

* Scaled Dot-Product Attention
* Multi-Head Attention
* Both validated through deterministic `pytest` tests

###End-to-end translation model

* Encoder–decoder Transformer using PyTorch `nn.Transformer`
* WMT17 German→English dataset
* Configurable model size & dataset subset

### Full training CLI

* Supports data loading, tokenization, batching, masking, warmup LR schedule
* Saves model weights + training statistics

---

## **Installation**

```bash
conda create -n llm_env python=3.10 -y
conda activate llm_env

pip install -r requirements.txt
```

---

## **Running Unit Tests**

The test suite checks:

* Scaled dot-product attention implementation
* Multi-head attention (Q/K/V projections + head combining)
* Padding masks & causal future-masking
* Exact numerical correctness using predefined tensors

Run:

```bash
pytest -q
```

---

## **Training the Translation Model**

Example command (recommended configuration):

```bash
python train.py \
    --subset_size 50000 \
    --epochs 20 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 3 \
    --batch_size 64
```

This will:

* Load & tokenize WMT17 (DE→EN)
* Train a small Transformer encoder–decoder model
* Save:

  * `output/model.pt`
  * `output/stats.pt` (epoch loss values)
  * `output/config.json`

---

## **Translating a Sentence**

After training:

```bash
python translate.py --model_dir output --text "Guten Abend, ich hoffe, du hattest einen angenehmen und produktiven Tag."
```

---

## **Project Structure**

```
translation-transformer-from-scratch/
│
├── modelling/
│   └── attention.py          # Scaled Dot-Product + Multi-Head Attention
│
├── train.py                  # Training pipeline (tokenization, batching, model)
├── translate.py              # Greedy decoding translation script
│
├── tests/
│   ├── test_attention.py     # Unit tests for attention
│   └── test_MHA.py           # Unit tests for multi-head attention
│
├── requirements.txt
└── README.md
```

---

## Notes

* `.gitignore` excludes `__pycache__` and other generated files
* The training outputs are not included in the repository to keep it lightweight
* This implementation focuses on correctness, readability, and passing unit tests—not state-of-the-art translation quality

---
