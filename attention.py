"""Scaled dot-product attention and multi-head self-attention, written from scratch.
Only nn.Linear, torch.matmul and softmax are used (no nn.MultiheadAttention)."""
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    q, k, v : (batch, heads, seq, d_k)
    mask    : broadcastable to (batch, heads, seq_q, seq_k); 0/False = blocked
    returns : output (batch, heads, seq, d_k), weights (batch, heads, seq, seq)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


def causal_mask(seq_len, device=None):
    """Lower-triangular mask: position i may only attend to positions <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def padding_mask(lengths, max_len):
    """lengths: (batch,) -> mask (batch, 1, 1, max_len), 0 where token is padding."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2).float()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)  # (b, h, s, d_k)

    def forward(self, x, mask=None):
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        b, _, s, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)  # concat heads
        return self.w_o(out), weights


if __name__ == "__main__":
    x = torch.randn(2, 6, 32)
    mha = MultiHeadSelfAttention(32, 4)
    out, w = mha(x)
    assert out.shape == (2, 6, 32) and w.shape == (2, 4, 6, 6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-5)
    _, wc = mha(x, causal_mask(6))
    assert torch.all(torch.triu(wc, diagonal=1) == 0)
    print("attention.py checks passed")
