# **Translation Transformer (WMT17 DE→EN)**

This project contains:

1. **Custom implementation of Scaled Dot-Product Attention**
2. **Custom implementation of Multi-Head Attention**

   * Both tested using `pytest`
3. **A complete Transformer-based German→English translation model**

   * Built using PyTorch’s `nn.Transformer`
4. **A training CLI using WMT17**
5. **Training statistics export (`stats.pt`)**

---

## Installation

```bash
conda create -n llm_env python=3.10 -y
conda activate llm_env

pip install -r requirements.txt
```

---

## Running Unit Tests

The included tests verify:

- Scaled dot-product attention  
- Multi-head attention projections  
- Padding masks and causal masks  
- Exact numerical correctness of outputs  

Run:

```bash
pytest -q
```

---

## Training the Translation Model

Example:

```bash
python train.py \
    --subset_size 50000 \
    --epochs 20 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 3 \
    --batch_size 64
```
---

## Project Structure

```
project/
│
├── modelling/
│   ├── attention.py        # attention + MHA (unit-tested)
│
├── train.py                # Training script
├── translate.py            # Translation Demo
├── README.md
├── requirements.txt
│
└── tests/
    ├── test_attention.py
    └── test_MHA.py
```

---
