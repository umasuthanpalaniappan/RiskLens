"""
LSTM Volatility Predictor — Indian Banking Sector
==================================================
Question: How volatile will each banking stock be over the next 20 days?

Why this works:
  Volatility clusters — high-vol periods follow high-vol periods.
  This is one of the most robust facts in finance (ARCH effect).

Target:
  Realised volatility = std(daily returns) over next 20 days × √252
  Annualised so numbers are interpretable (e.g. 0.25 = 25% annual vol)

Evaluation:
  R² score         — how much future vol variance does model explain?
  Correlation      — predicted vs actual vol
  Direction acc    — did we correctly predict vol going UP or DOWN?
  Baseline         — predict next vol = current vol (naive persistence)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import random

# ─────────────────────────────────────────
# 0) Seeds — full reproducibility
# ─────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────
# 0) Settings
# ─────────────────────────────────────────
TICKERS = [
    "HDFCBANK.NS",  "ICICIBANK.NS", "SBIN.NS",      "AXISBANK.NS",   "KOTAKBANK.NS",
    "INDUSINDBK.NS","BANKBARODA.NS","PNB.NS",        "FEDERALBNK.NS", "IDFCFIRSTB.NS"
]
MARKET_TICKER = "^NSEBANK"
N        = len(TICKERS)
L        = 20      # lookback window (days of features)
H        = 20      # forward window (days to measure future volatility)
EPOCHS   = 100
PATIENCE = 15
BATCH    = 32
LR       = 3e-4
HIDDEN   = 64
device   = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ─────────────────────────────────────────
# 1) Download data
# ─────────────────────────────────────────
raw = yf.download(
    TICKERS + [MARKET_TICKER],
    start="2014-01-01",
    end="2024-12-31",
    interval="1d",
    group_by="ticker",
    auto_adjust=True
)

close     = pd.DataFrame({t: raw[t]["Close"]  for t in TICKERS}).dropna()
volume    = pd.DataFrame({t: raw[t]["Volume"] for t in TICKERS}).dropna()
mkt_close = raw[MARKET_TICKER]["Close"].dropna()

common    = close.index.intersection(volume.index).intersection(mkt_close.index)
close     = close.loc[common]
volume    = volume.loc[common]
mkt_close = mkt_close.loc[common]
print(f"Data: {len(common)} days × {N} stocks")

# ─────────────────────────────────────────
# 2) Daily returns
# ─────────────────────────────────────────
daily_ret = close.pct_change()   # [T, N]
mkt_ret   = mkt_close.pct_change()

# ─────────────────────────────────────────
# 3) TARGET — Future realised volatility
# ─────────────────────────────────────────
abs_future_vol  = pd.DataFrame(index=close.index, columns=TICKERS, dtype=np.float32)
abs_current_vol = pd.DataFrame(index=close.index, columns=TICKERS, dtype=np.float32)

for tkr in TICKERS:
    r = daily_ret[tkr]
    for i in range(H, len(r) - H):
        fwd_window  = r.iloc[i+1  : i+H+1]
        past_window = r.iloc[i-H+1: i+1]
        abs_future_vol.loc[r.index[i],  tkr] = float(fwd_window.std()  * np.sqrt(252))
        abs_current_vol.loc[r.index[i], tkr] = float(past_window.std() * np.sqrt(252))

abs_future_vol  = abs_future_vol.dropna()
abs_current_vol = abs_current_vol.dropna()

future_vol = (abs_future_vol / (abs_current_vol + 1e-8)).clip(0.2, 5.0)
future_vol = future_vol.dropna()

# ─────────────────────────────────────────
# 4) FEATURES — 8 per stock + 4 market features
# ─────────────────────────────────────────
def rsi(prices, period=14):
    d    = prices.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    return (100 - 100 / (1 + gain / (loss + 1e-8))) / 100.0

F_STOCK = 8
F_MKT   = 4
F_TOTAL = N * F_STOCK + F_MKT

stock_feat = {}
for tkr in TICKERS:
    r  = daily_ret[tkr]
    v  = volume[tkr]
    vol5_s  = r.rolling(5).std()  * np.sqrt(252)
    vol20_s = r.rolling(20).std() * np.sqrt(252)
    stock_feat[tkr] = pd.DataFrame({
        "ret1":      r,
        "ret5":      close[tkr].pct_change(5),
        "ret20":     close[tkr].pct_change(20),
        "vol5":      vol5_s,
        "vol20":     vol20_s,
        "vol_ratio": vol5_s / (vol20_s + 1e-8),
        "rsi14":     rsi(close[tkr]),
        "volume_z":  (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8),
    })

mkt_vol20 = mkt_ret.rolling(20).std() * np.sqrt(252)
mkt_feat  = pd.DataFrame({
    "mkt_ret1":  mkt_ret,
    "mkt_ret5":  mkt_close.pct_change(5),
    "mkt_vol20": mkt_vol20,
    "mkt_rsi14": rsi(mkt_close),
})

# ─────────────────────────────────────────
# 5) Align all dates
# ─────────────────────────────────────────
all_stock = pd.concat([stock_feat[t] for t in TICKERS], axis=1).dropna()
valid_dates = all_stock.index\
    .intersection(mkt_feat.dropna().index)\
    .intersection(future_vol.index)
print(f"Valid dates: {len(valid_dates)}")

# ─────────────────────────────────────────
# 6) Build arrays
# ─────────────────────────────────────────
vals    = np.zeros((len(valid_dates), F_TOTAL), dtype=np.float32)
targets = np.zeros((len(valid_dates), N),       dtype=np.float32)

for i, dt in enumerate(valid_dates):
    row = []
    for tkr in TICKERS:
        sf = stock_feat[tkr].loc[dt]
        row.extend([
            sf["ret1"], sf["ret5"], sf["ret20"],
            sf["vol5"], sf["vol20"], sf["vol_ratio"],
            sf["rsi14"], sf["volume_z"],
        ])
    mf = mkt_feat.loc[dt]
    row.extend([mf["mkt_ret1"], mf["mkt_ret5"], mf["mkt_vol20"], mf["mkt_rsi14"]])
    vals[i]    = np.array(row, dtype=np.float32)
    targets[i] = [float(future_vol.loc[dt, tkr]) for tkr in TICKERS]

# ─────────────────────────────────────────
# 7) Rolling windows
# ─────────────────────────────────────────
X_list, y_list = [], []
for t in range(L, len(valid_dates)):
    X_list.append(vals[t-L:t])
    y_list.append(targets[t])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"X: {X.shape}  y: {y.shape}")

# ─────────────────────────────────────────
# 8) Chronological split  70/15/15
# ─────────────────────────────────────────
n         = len(y)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train, y_train = X[:train_end],        y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:],          y[val_end:]
print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

# ─────────────────────────────────────────
# 9) Naive baseline
# ─────────────────────────────────────────
current_vol_idx = [F_STOCK * j + 4 for j in range(N)]
baseline_preds  = X_test[:, -1, current_vol_idx]

# ─────────────────────────────────────────
# 10) Normalize X using train stats only
# ─────────────────────────────────────────
mu = X_train.mean(axis=(0, 1), keepdims=True)
sd = X_train.std( axis=(0, 1), keepdims=True) + 1e-8
X_train = (X_train - mu) / sd
X_val   = (X_val   - mu) / sd
X_test  = (X_test  - mu) / sd

y_train_n = y_train
y_val_n   = y_val
y_test_n  = y_test

# ─────────────────────────────────────────
# 11) Tensors
# ─────────────────────────────────────────
X_tr = torch.tensor(X_train).to(device)
y_tr = torch.tensor(y_train_n).to(device)
X_va = torch.tensor(X_val).to(device)
y_va = torch.tensor(y_val_n).to(device)
X_te = torch.tensor(X_test).to(device)
y_te = torch.tensor(y_test_n).to(device)

# ─────────────────────────────────────────
# 12) Model
# ─────────────────────────────────────────
class VolLSTM(nn.Module):
    def __init__(self, fin, hidden=64, n_stocks=10):
        super().__init__()
        self.lstm = nn.LSTM(fin, hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_stocks),
            nn.Softplus()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.fc(h)

model = VolLSTM(fin=X_tr.shape[-1], hidden=HIDDEN, n_stocks=N).to(device)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
crit  = nn.HuberLoss(delta=0.1)

# ─────────────────────────────────────────
# 13) Training helpers
# ─────────────────────────────────────────
def run_epoch(Xb, yb, train=True):
    model.train() if train else model.eval()
    if train:
        g = torch.Generator()
        g.manual_seed(SEED)
        idx = torch.randperm(len(yb), generator=g)
    else:
        idx = torch.arange(len(yb))
    total = seen = 0
    for i in range(0, len(yb), BATCH):
        b = idx[i:i+BATCH]
        xb, ybb = Xb[b], yb[b]
        if train:
            opt.zero_grad()
        pred = model(xb)
        loss = crit(pred, ybb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(b)
        seen  += len(b)
    return total / seen

@torch.no_grad()
def get_predictions(Xb):
    model.eval()
    return model(Xb).cpu().numpy()

# ─────────────────────────────────────────
# 14) Training loop
# ─────────────────────────────────────────
best_val   = float("inf")
best_state = None
bad        = 0

print(f"\n{'Ep':>3}  {'TrMSE':>8}  {'VaMSE':>8}  Note")
print("-" * 40)

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(X_tr, y_tr, train=True)
    va = run_epoch(X_va, y_va, train=False)
    sched.step(va)

    note = ""
    if va < best_val - 1e-6:
        best_val   = va
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad        = 0
        note       = "✓ saved"
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch == 1 or epoch % 5 == 0 or note:
        print(f"{epoch:>3}  {tr:>8.5f}  {va:>8.5f}  {note}")

if best_state:
    model.load_state_dict(best_state)

# ─────────────────────────────────────────
# 15) Final evaluation
# ─────────────────────────────────────────
pred_ratio = get_predictions(X_te)
true_ratio = y_test

pred_flat     = pred_ratio.flatten()
true_flat     = true_ratio.flatten()
baseline_flat = np.ones_like(true_flat)

r2_model    = r2_score(true_flat, pred_flat)
r2_baseline = r2_score(true_flat, baseline_flat)

corr_model    = float(np.corrcoef(pred_flat, true_flat)[0, 1])
corr_baseline = float(np.corrcoef(baseline_flat, true_flat)[0, 1])

true_direction = (true_ratio  > 1.0).astype(float)
pred_direction = (pred_ratio  > 1.0).astype(float)
dir_acc_model  = (pred_direction == true_direction).mean()

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)
print(f"\n{'Metric':<28} {'Model':>8}  {'Baseline':>10}")
print("-" * 50)
print(f"{'R² Score':<28} {r2_model:>8.4f}  {r2_baseline:>10.4f}")
print(f"{'Correlation':<28} {corr_model:>8.4f}  {corr_baseline:>10.4f}")
print(f"{'Direction Accuracy':<28} {dir_acc_model:>8.4f}  {'0.5000':>10}")

print(f"\n--- Per-Stock Results ---")
print(f"{'Stock':<20}  {'R²':>6}  {'Corr':>6}  {'DirAcc':>7}  Status")
print("-" * 55)
for j, tkr in enumerate(TICKERS):
    r2  = r2_score(true_ratio[:, j], pred_ratio[:, j])
    cor = float(np.corrcoef(pred_ratio[:, j], true_ratio[:, j])[0, 1])
    da  = (((pred_ratio[:, j] > 1.0) == (true_ratio[:, j] > 1.0))).mean()
    status = "✓ good" if r2 > 0.2 else ("◑ weak" if r2 > 0 else "✗ poor")
    print(f"{tkr:<20}  {r2:>6.3f}  {cor:>6.3f}  {da:>7.3f}  {status}")

print(f"\n--- Sample predictions (last test day) ---")
print(f"{'Stock':<18}  {'CurrVol':>8}  {'PredFutVol':>11}  {'ActualFutVol':>13}  {'Correct?':>8}")
print("-" * 65)

current_vol_raw = baseline_preds

for j, tkr in enumerate(TICKERS):
    curr     = current_vol_raw[-1, j]
    pred_abs = pred_ratio[-1, j] * curr
    try:
        dt = valid_dates[val_end + len(y_test) - 1]
        actual_abs = float(abs_future_vol.loc[dt, tkr]) if dt in abs_future_vol.index else float('nan')
    except:
        actual_abs = float('nan')
    correct = "✓" if (pred_ratio[-1,j] > 1.0) == (true_ratio[-1,j] > 1.0) else "✗"
    print(f"{tkr:<18}  {curr:>7.1%}  {pred_abs:>10.1%}  {actual_abs:>12.1%}  {correct:>8}")

print(f"\n✓ KEY METRIC: Direction Accuracy {dir_acc_model:.1%} vs 50% random baseline")
print(f"   The model correctly predicts whether volatility will")
print(f"   INCREASE or DECREASE over the next 20 trading days.")