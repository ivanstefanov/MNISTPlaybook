import argparse
import subprocess
import sys
import os

# install torch CPU
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch==2.1.2+cpu",
    "--extra-index-url", "https://download.pytorch.org/whl/cpu",
])

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--epochs", type=int, required=True)
args = parser.parse_args()

# synthetic dataset (same as pipeline)
rng = np.random.default_rng(42)
X = rng.random((2000, 10), dtype=np.float32)
y = (X.sum(axis=1) > 5).astype(np.int64)

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
Xva = torch.tensor(Xva, dtype=torch.float32)
yva = torch.tensor(yva, dtype=torch.long)

net = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

net.train()
for _ in range(args.epochs):
    optimizer.zero_grad()
    logits = net(Xtr)
    loss = loss_fn(logits, ytr)
    loss.backward()
    optimizer.step()

net.eval()
with torch.no_grad():
    preds = torch.argmax(net(Xva), dim=1)

acc = (preds == yva).float().mean().item()

# IMPORTANT: Katib reads metric from stdout
print(f"accuracy={acc}")
