import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask):
        # q: [B, Lq, D], k/v: [B, Lk, D]
        B, Lq, D = query.shape
        _, Lk, _ = key.shape

        # 1) Scaled dot-product attention
        scale = D ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale   # [B, Lq, Lk]

        # 2) Apply padding mask
        if attention_mask is not None:
            # attention_mask: [B, Lk]
            mask = attention_mask.unsqueeze(1).expand(-1, Lq, -1)  # [B, Lq, Lk]
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3) Apply future mask
        if self.mask_future:
            future_mask = torch.triu(torch.ones(Lq, Lk), diagonal=1).bool()
            scores = scores.masked_fill(future_mask.to(scores.device), -1e9)

        # 4) Softmax
        attn = F.softmax(scores, dim=-1)

        # 5) Weighted sum
        output = torch.bmm(attn, value)  # [B, Lq, D]

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask_future=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mask_future = mask_future

        # weight names must match test state_dict
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform   = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(mask_future)

    def split_heads(self, x):
        B, L, D = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, L, Hd]

    def combine_heads(self, x):
        B, H, L, Hd = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return x

    def forward(self, q, k, v, attention_mask):
        B = q.shape[0]

        Q = self.split_heads(self.query_transform(q))
        K = self.split_heads(self.key_transform(k))
        V = self.split_heads(self.value_transform(v))

        # Compute per-head attention
        outputs = []
        for h in range(self.num_heads):
            out = self.attention(
                Q[:, h],  # [B, Lq, Hd]
                K[:, h],  # [B, Lk, Hd]
                V[:, h],  # [B, Lk, Hd]
                attention_mask
            )
            outputs.append(out)

        # Stack heads: [B, H, Lq, Hd]
        out = torch.stack(outputs, dim=1)

        # Combine heads → [B, Lq, d_model]
        out = self.combine_heads(out)

        # Final linear
        out = self.output_transform(out)
        return out
