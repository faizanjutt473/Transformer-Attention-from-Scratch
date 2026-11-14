"""Train the small transformer on sequence reversal."""
import torch
import os
os.makedirs("images", exist_ok=True)
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import make_dataset, VOCAB_SIZE
from transformer import SmallTransformer

SEQ_LEN, EPOCHS, BATCH, LR = 8, 30, 128, 2e-3
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dl = DataLoader(make_dataset(20000, SEQ_LEN, seed=1), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(make_dataset(2000, SEQ_LEN, seed=2), batch_size=BATCH)

model = SmallTransformer(VOCAB_SIZE, d_model=64, n_heads=4, d_ff=128, n_layers=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)[0].argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


train_losses, val_accs = [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    train_losses.append(running / len(train_dl))
    val_accs.append(evaluate())
    print(f"Epoch {epoch:02d} | loss {train_losses[-1]:.4f} | val acc {val_accs[-1]:.4f}")

torch.save(model.state_dict(), "model.pt")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(train_losses); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(val_accs); ax[1].set_title("Validation token accuracy"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.savefig("images/training_curves.png", dpi=150)
print("Saved model.pt and images/training_curves.png")
