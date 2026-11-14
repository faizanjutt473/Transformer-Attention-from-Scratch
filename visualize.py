"""Visualize attention weights of the trained model (one heatmap per layer/head)."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data import encode, decode, VOCAB_SIZE
from transformer import SmallTransformer

TEXT = "abcdefgh"

model = SmallTransformer(VOCAB_SIZE, d_model=64, n_heads=4, d_ff=128, n_layers=2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

x = encode(TEXT)
with torch.no_grad():
    logits, weights = model(x)
pred = decode(logits.argmax(-1)[0])
print(f"input: {TEXT} | model output: {pred} | expected: {TEXT[::-1]}")

n_layers, n_heads = len(weights), weights[0].size(1)
fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 3.8 * n_layers))
for l in range(n_layers):
    for h in range(n_heads):
        ax = axes[l][h]
        sns.heatmap(weights[l][0, h].numpy(), ax=ax, cmap="viridis", vmin=0, vmax=1,
                    xticklabels=list(TEXT), yticklabels=list(TEXT), cbar=(h == n_heads - 1))
        ax.set_title(f"Layer {l+1} - Head {h+1}")
        ax.set_xlabel("key (attended to)"); ax.set_ylabel("query")
plt.suptitle(f"Attention weights  |  input '{TEXT}'  ->  output '{pred}'")
plt.tight_layout(); plt.savefig("images/attention_heatmaps.png", dpi=150)

# Average over heads, per layer
fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5))
for l in range(n_layers):
    sns.heatmap(weights[l][0].mean(0).numpy(), ax=axes[l], cmap="magma", vmin=0, vmax=1,
                xticklabels=list(TEXT), yticklabels=list(TEXT))
    axes[l].set_title(f"Layer {l+1} (mean of heads)")
plt.tight_layout(); plt.savefig("images/attention_mean_per_layer.png", dpi=150)

# Quantify: how much attention lands on the mirror position (anti-diagonal)?
L = len(TEXT)
for l in range(n_layers):
    w = weights[l][0]
    mirror = torch.stack([w[:, i, L - 1 - i] for i in range(L)], dim=1).mean(1)
    print(f"Layer {l+1}: mean attention on mirror position per head =", [round(v, 3) for v in mirror.tolist()])
print("Saved images/attention_heatmaps.png and images/attention_mean_per_layer.png")
