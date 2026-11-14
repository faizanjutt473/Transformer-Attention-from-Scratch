"""Positional encoding, feed-forward, transformer block and the full small model."""
import math
import torch
import torch.nn as nn
from attention import MultiHeadSelfAttention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Attention -> Add&Norm -> FeedForward -> Add&Norm (post-LN, as in the original paper)."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, weights = self.attn(x, mask)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, weights


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.1, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, mask=None):
        x = self.pos(self.embed(tokens) * math.sqrt(self.d_model))
        all_weights = []
        for block in self.blocks:
            x, w = block(x, mask)
            all_weights.append(w)
        return self.out(x), all_weights  # logits (b, s, vocab), list of (b, h, s, s)
