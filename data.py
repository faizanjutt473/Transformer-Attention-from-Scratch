"""Toy task: sequence reversal.  'hello' -> 'olleh'."""
import string
import torch
from torch.utils.data import TensorDataset

CHARS = list(string.ascii_lowercase)
stoi = {c: i for i, c in enumerate(CHARS)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(CHARS)


def make_dataset(n_samples, seq_len=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB_SIZE, (n_samples, seq_len), generator=g)
    y = torch.flip(x, dims=[1])
    return TensorDataset(x, y)


def decode(tokens):
    return "".join(itos[int(t)] for t in tokens)


def encode(text):
    return torch.tensor([stoi[c] for c in text]).unsqueeze(0)
