"""
Scorer v2 (overwrite baseline):
- Keep all samples (including activity=0)
- DataLoader uses WeightedRandomSampler: reduce zero-sample sampling rate, increase non-zero sample sampling rate
- Data augmentation: do reverse-complement with p=0.5
- Model: two small conv layers + GELU + Dropout (more stable)
"""

import os, random, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DATA_DIR = os.path.expanduser('~/CRE-seq/data/sharpr_mpra')
CSV_PATH = os.path.join(DATA_DIR, 'k562_train.csv')

# -------------------- Data --------------------
df = pd.read_csv(CSV_PATH)
print(f"Total samples (keep zeros): {len(df)}")

# one-hot + reverse-complement
base2idx = {'A':0, 'C':1, 'G':2, 'T':3}
comp = str.maketrans({'A':'T','T':'A','C':'G','G':'C'})
def rc(seq):  # reverse complement
    return seq.upper().translate(comp)[::-1]

def onehot(seq):
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i,b in enumerate(seq.upper()):
        if b in base2idx: arr[base2idx[b], i] = 1.0
    return arr

class SeqDataset(Dataset):
    def __init__(self, df, augment=False):
        self.seqs = df['sequence'].tolist()
        self.y = df['activity'].values.astype(np.float32)
        self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        s = self.seqs[i]
        if self.augment and random.random() < 0.5:
            s = rc(s)
        x = torch.tensor(onehot(s))
        y = torch.tensor(self.y[i])
        return x, y

# split
tr_df, va_df = train_test_split(df, test_size=0.1, random_state=1)

# -------- Core: weighted sampling --------
act = tr_df['activity'].values
is_zero = np.isclose(act, 0.0, atol=1e-9)

w0 = 0.1     # Base weight for zero samples (smaller → less likely to be sampled, but still retained)
w1 = 1.0     # Base weight for non-zero samples
scale = 1.0  # Scaling coefficient for non-zero sample weights by |activity| (log1p smoothing)
weights = np.where(is_zero, w0, w1 * (np.log1p(np.abs(act)) * scale + 1.0))
weights = weights * (len(weights) / weights.sum())  # Normalize for numerical stability

sampler = WeightedRandomSampler(
    weights=torch.tensor(weights, dtype=torch.double),
    num_samples=len(weights),
    replacement=True
)

train_ds = SeqDataset(tr_df, augment=True)
val_ds   = SeqDataset(va_df, augment=False)

train_dl = DataLoader(train_ds, batch_size=128, sampler=sampler)
val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False)

# -------------------- Model --------------------
class CNNScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=8),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNScorer().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
loss_fn = nn.MSELoss()

# -------------------- Eval --------------------
def evaluate():
    model.eval()
    with torch.no_grad():
        preds, trues = [], []
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds.append(model(xb))
            trues.append(yb)
        preds = torch.cat(preds).cpu().numpy()
        trues = torch.cat(trues).cpu().numpy()
        r = np.corrcoef(preds, trues)[0,1]
        return r

# -------------------- Train --------------------
best_r, best_state = -1.0, None
for epoch in range(8):
    model.train()
    total = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
    train_loss = total / len(train_ds)
    r = evaluate()
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | val_r={r:.3f}")
    if r > best_r:
        best_r, best_state = r, {k:v.cpu() for k,v in model.state_dict().items()}

# Save best
if best_state is not None:
    model.load_state_dict(best_state)
out_path = os.path.join(DATA_DIR, 'scorer_cnn_weighted.pt')
torch.save(model.state_dict(), out_path)
print(f"Saved best model with val_r={best_r:.3f} -> {os.path.basename(out_path)}")
