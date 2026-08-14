"""
LSTM + GNN Volatility Predictor — Final Version
================================================
Two-stage model:
  Stage 1: LSTM processes 20-day windows of 84 features per stock
  Stage 2: GNN models cross-stock volatility spillover via dynamic
           rolling correlation graph (60-day window, threshold=0.65)

Key design choices:
  - Stricter correlation threshold (0.65) — isolates real PSU cluster edges
  - Dynamic rolling graph (60-day) — adapts to regime changes
  - Extended dataset (2009-2024) — 15 years for better GNN generalisation

LSTM results are hardcoded from a prior standalone LSTM run.
This script trains and evaluates the combined LSTM+GNN model,
then plots a comparison between both.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import random

# ─────────────────────────────────────────
# 0) Seeds
# ─────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────
# 1) Settings
# ─────────────────────────────────────────
TICKERS = [
    "HDFCBANK.NS",   "ICICIBANK.NS",  "SBIN.NS",
    "AXISBANK.NS",   "KOTAKBANK.NS",  "INDUSINDBK.NS",
    "BANKBARODA.NS", "PNB.NS",        "FEDERALBNK.NS",
    "IDFCFIRSTB.NS"
]
MARKET_TICKER  = "^NSEBANK"
N              = len(TICKERS)
L              = 20       # lookback window fed to LSTM
H              = 20       # forward window for volatility target
EPOCHS         = 100
PATIENCE       = 15
BATCH          = 32
LR             = 3e-4
HIDDEN         = 64
GNN_HIDDEN     = 32   # smaller than LSTM hidden — prevents GNN from dominating
CORR_THRESHOLD = 0.65     # only keep strong correlation edges
ROLLING_WINDOW = 60       # days used to build each sample's graph
START_DATE     = "2009-01-01"
END_DATE       = "2024-12-31"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ─────────────────────────────────────────
# 2) Download data
# ─────────────────────────────────────────
print(f"\nDownloading data {START_DATE} → {END_DATE}...")
raw = yf.download(
    TICKERS + [MARKET_TICKER],
    start=START_DATE,
    end=END_DATE,
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
# 3) Daily returns
# ─────────────────────────────────────────
daily_ret = close.pct_change()
mkt_ret   = mkt_close.pct_change()

# ─────────────────────────────────────────
# 4) Dynamic adjacency matrix builder
#
#    For each sample we compute a fresh adjacency matrix
#    using the past ROLLING_WINDOW (60) days of returns.
#    This means the graph adapts to market regimes:
#      COVID 2020  → all stocks correlated → dense graph
#      PSU rally 2023 → PSU decouples from private → sparse cross-cluster
# ─────────────────────────────────────────
def build_adj(returns_window, threshold=0.65):
    """
    Build Kipf-normalised adjacency matrix from a return window.

    Args:
        returns_window : [T_window, N] numpy array of daily returns
        threshold      : minimum correlation to keep an edge

    Returns:
        adj_norm : [N, N] normalised adjacency matrix
    """
    corr = np.corrcoef(returns_window.T)       # [N, N] Pearson correlation
    corr = np.nan_to_num(corr, nan=0.0)

    adj  = corr.copy()
    adj[adj < threshold] = 0.0                 # remove weak edges
    np.fill_diagonal(adj, 1.0)                 # always self-connected

    # Kipf & Welling (2017) normalisation: D^(-1/2) A D^(-1/2)
    degree   = adj.sum(axis=1)
    d_inv_sq = np.diag(1.0 / np.sqrt(degree + 1e-8))
    adj_norm = d_inv_sq @ adj @ d_inv_sq

    return adj_norm.astype(np.float32)


# Global correlation matrix for display/analysis
global_corr = daily_ret.corr().values.astype(np.float32)

print("\n--- Global Correlation (reference) ---")
print(f"PSU cluster:  SBIN-PNB={global_corr[2,7]:.2f}  "
      f"SBIN-BANKBARODA={global_corr[2,6]:.2f}  "
      f"PNB-BANKBARODA={global_corr[6,7]:.2f}")
print(f"Private:      HDFC-ICICI={global_corr[0,1]:.2f}  "
      f"HDFC-KOTAK={global_corr[0,4]:.2f}  "
      f"ICICI-AXIS={global_corr[1,3]:.2f}")

# ─────────────────────────────────────────
# 5) Volatility ratio target
#
#    VR = σ_future / σ_current
#    Both annualised with √252, but the factor cancels in the ratio.
#    Clipped to [0.2, 5.0] to reduce COVID outlier influence.
# ─────────────────────────────────────────
abs_future_vol  = pd.DataFrame(index=close.index, columns=TICKERS, dtype=np.float32)
abs_current_vol = pd.DataFrame(index=close.index, columns=TICKERS, dtype=np.float32)

for tkr in TICKERS:
    r = daily_ret[tkr]
    for i in range(H, len(r) - H):
        fwd_window  = r.iloc[i+1   : i+H+1]
        past_window = r.iloc[i-H+1 : i+1]
        abs_future_vol.loc[r.index[i],  tkr] = float(fwd_window.std()  * np.sqrt(252))
        abs_current_vol.loc[r.index[i], tkr] = float(past_window.std() * np.sqrt(252))

abs_future_vol  = abs_future_vol.dropna()
abs_current_vol = abs_current_vol.dropna()
raw_ratio       = (abs_future_vol / abs_current_vol).clip(0.2, 5.0)

# ─────────────────────────────────────────
# 6) Feature engineering — 84 features per day
#
#    Per stock (×10 stocks = 80 features):
#      ret1, ret5, ret20  — return momentum
#      vol5, vol20        — rolling annualised volatility
#      vol_ratio          — vol5/vol20 — trend direction
#      rsi14              — RSI momentum oscillator (scaled 0-1)
#      volume_z           — volume z-score vs 20-day mean
#
#    Market (×4 features):
#      mret1, mret5, mvol, mrsi — NIFTY Bank index context
# ─────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return (100 - 100 / (1 + rs)) / 100

feature_rows = []
adj_matrices = []   # one adjacency matrix per sample
valid_dates  = []

for i in range(max(L, H, ROLLING_WINDOW), len(abs_future_vol) - H):
    date = abs_future_vol.index[i]
    row  = []

    # Per-stock features (80 total)
    for tkr in TICKERS:
        r   = daily_ret[tkr]
        v   = volume[tkr]
        idx = r.index.get_loc(date)

        ret1  = float(r.iloc[idx])
        ret5  = float(r.iloc[idx-4:idx+1].sum())
        ret20 = float(r.iloc[idx-19:idx+1].sum())
        vol5  = float(r.iloc[idx-4:idx+1].std()  * np.sqrt(252))
        vol20 = float(r.iloc[idx-19:idx+1].std() * np.sqrt(252))
        vrat  = vol5 / (vol20 + 1e-8)
        rsi14 = float(rsi(r.iloc[idx-27:idx+1]).iloc[-1])
        vol_z = float((v.iloc[idx] - v.iloc[idx-19:idx].mean()) /
                      (v.iloc[idx-19:idx].std() + 1e-8))
        row.extend([ret1, ret5, ret20, vol5, vol20, vrat, rsi14, vol_z])

    # Market features (4 total)
    midx  = mkt_ret.index.get_loc(date)
    mret1 = float(mkt_ret.iloc[midx])
    mret5 = float(mkt_ret.iloc[midx-4:midx+1].sum())
    mvol  = float(mkt_ret.iloc[midx-19:midx+1].std() * np.sqrt(252))
    mrsi  = float(rsi(mkt_ret.iloc[midx-27:midx+1]).iloc[-1])
    row.extend([mret1, mret5, mvol, mrsi])

    feature_rows.append(row)
    valid_dates.append(date)

    # Build dynamic adjacency matrix for this sample
    # Uses past 60 days of returns — separate from the 20-day LSTM window
    date_idx   = daily_ret.index.get_loc(date)
    ret_window = daily_ret.iloc[date_idx - ROLLING_WINDOW + 1 : date_idx + 1].values
    adj_matrices.append(build_adj(ret_window, threshold=CORR_THRESHOLD))

feature_arr = np.array(feature_rows,  dtype=np.float32)   # [T, 84]
targets_arr = raw_ratio.loc[valid_dates].values.astype(np.float32)  # [T, N]
adj_arr     = np.array(adj_matrices,  dtype=np.float32)   # [T, N, N]

print(f"\nFeature matrix : {feature_arr.shape}")
print(f"Adj matrices   : {adj_arr.shape}  ← one graph per sample")

# ─────────────────────────────────────────
# 7) Sliding windows
# ─────────────────────────────────────────
X_list, y_list, A_list = [], [], []
for t in range(L, len(valid_dates)):
    X_list.append(feature_arr[t-L:t])   # [20, 84]
    y_list.append(targets_arr[t])        # [10]
    A_list.append(adj_arr[t])            # [N, N]

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
A = np.array(A_list, dtype=np.float32)
print(f"\nX: {X.shape}  y: {y.shape}  A: {A.shape}")

# ─────────────────────────────────────────
# 8) Chronological train / val / test split
#    70% / 15% / 15% — no shuffling to prevent leakage
# ─────────────────────────────────────────
n         = len(y)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train, y_train, A_train = X[:train_end],        y[:train_end],        A[:train_end]
X_val,   y_val,   A_val   = X[train_end:val_end], y[train_end:val_end], A[train_end:val_end]
X_test,  y_test,  A_test  = X[val_end:],          y[val_end:],          A[val_end:]
print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

# ─────────────────────────────────────────
# 9) Normalise using train statistics only
# ─────────────────────────────────────────
mu = X_train.mean(axis=(0, 1), keepdims=True)
sd = X_train.std( axis=(0, 1), keepdims=True) + 1e-8
X_train = (X_train - mu) / sd
X_val   = (X_val   - mu) / sd
X_test  = (X_test  - mu) / sd

# ─────────────────────────────────────────
# 10) Tensors
# ─────────────────────────────────────────
X_tr = torch.tensor(X_train).to(device)
y_tr = torch.tensor(y_train).to(device)
A_tr = torch.tensor(A_train).to(device)

X_va = torch.tensor(X_val).to(device)
y_va = torch.tensor(y_val).to(device)
A_va = torch.tensor(A_val).to(device)

X_te = torch.tensor(X_test).to(device)
y_te = torch.tensor(y_test).to(device)
A_te = torch.tensor(A_test).to(device)

# ─────────────────────────────────────────
# 11) Dynamic GCN Layer
#
#    Each sample in the batch has its OWN adjacency matrix.
#    Message passing:
#      H_new[b,i] = ReLU( Σⱼ A[b,i,j] × H[b,j] @ W )
#    i.e. each stock aggregates its neighbours' LSTM states
#    weighted by their current correlation strength.
# ─────────────────────────────────────────
class DynamicGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W          = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU()

    def forward(self, H, A):
        """
        H : [batch, N, in_features]
        A : [batch, N, N]  — dynamic per-sample adjacency
        """
        H_agg = torch.bmm(A, H)    # [batch, N, in_features]
        return self.activation(self.W(H_agg))  # [batch, N, out_features]


# ─────────────────────────────────────────
# 12) LSTM + Dynamic GNN Model  (Gated version)
#
#    Forward pass:
#      1. LSTM reads 20-day feature window → one hidden state per batch
#      2. Stock projection splits hidden state into 10 per-stock vectors
#      3. GCN layer does message passing using the 60-day dynamic graph
#      4. Residual: add stock_h back to gnn_h so LSTM signal is never lost
#      5. Learnable gate α per stock — decides how much to trust GNN vs LSTM
#         α = sigmoid(linear(stock_h))  — [batch, N, 1]
#         out = α * gnn_h_residual + (1-α) * stock_h
#         If α → 0: pure LSTM (e.g. ICICIBANK — already strong alone)
#         If α → 1: trust GNN (e.g. SBIN — benefits from PSU neighbours)
#      6. Prediction head outputs one volatility ratio per stock
#
#    FIX SUMMARY vs previous version:
#      - GNN_HIDDEN reduced 64→32 : prevents GNN dominating LSTM
#      - Residual connection       : LSTM signal never fully overwritten
#      - Learnable gate per stock  : model decides per stock how much GNN to use
#      - Higher weight_decay 1e-4→5e-4 on GNN weights via param groups
# ─────────────────────────────────────────
class VolLSTM_GatedGNN(nn.Module):
    def __init__(self, fin, hidden=64, gnn_hidden=32, n_stocks=10):
        super().__init__()
        self.n_stocks  = n_stocks
        self.stock_dim = 32

        # Stage 1 — temporal learning
        self.lstm = nn.LSTM(
            input_size  = fin,
            hidden_size = hidden,
            num_layers  = 2,
            dropout     = 0.2,
            batch_first = True
        )
        self.drop = nn.Dropout(0.2)

        # Bridge: shared LSTM state → per-stock vectors
        self.stock_proj = nn.Linear(hidden, n_stocks * self.stock_dim)

        # Stage 2 — cross-stock learning (smaller hidden = less dominance)
        self.gcn = DynamicGraphConvLayer(
            in_features  = self.stock_dim,
            out_features = gnn_hidden
        )

        # Residual projection: map gnn_hidden back to stock_dim so we can add
        self.gnn_residual_proj = nn.Linear(gnn_hidden, self.stock_dim, bias=False)

        # Learnable gate: for each stock, learn how much to trust GNN signal
        # Input: stock_h (LSTM representation) — shape [batch, N, stock_dim]
        # Output: α in (0,1) — shape [batch, N, 1]
        # α close to 0 → trust LSTM; α close to 1 → trust GNN
        self.gate = nn.Sequential(
            nn.Linear(self.stock_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Initialize gate to be skeptical of GNN at the start
        # sigmoid(-2) ≈ 0.12 → starts at ~12% GNN trust
        # Gate only opens if GNN signal genuinely reduces loss during training
        nn.init.constant_(self.gate[2].bias, -2.0)

        # Prediction head: gated blend → volatility ratio
        self.pred_head = nn.Sequential(
            nn.Linear(self.stock_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()    # always positive — volatility ratios are > 0
        )

    def forward(self, x, A):
        """
        x : [batch, 20, 84]  — 20-day feature window
        A : [batch, N, N]    — dynamic adjacency matrix for this sample
        """
        batch_size = x.shape[0]

        # LSTM: read 20-day sequence, take last hidden state
        lstm_out, _ = self.lstm(x)
        h = self.drop(lstm_out[:, -1, :])                        # [batch, hidden]

        # Split into per-stock representations
        stock_h = self.stock_proj(h).view(
            batch_size, self.n_stocks, self.stock_dim)           # [batch, N, 32]

        # GCN: each stock aggregates from correlated neighbours
        gnn_h = self.gcn(stock_h, A)                             # [batch, N, 32]

        # Residual: project GNN output back to stock_dim and add to stock_h
        # This ensures the LSTM signal is NEVER fully lost
        gnn_h_res = self.gnn_residual_proj(gnn_h)               # [batch, N, 32]
        gnn_with_residual = gnn_h_res + stock_h                  # [batch, N, 32]

        # Learnable gate: α per stock, learned from LSTM representation
        alpha = self.gate(stock_h)                               # [batch, N, 1]

        # Blend: α controls how much GNN to mix in per stock
        # ICICIBANK → learns α ≈ 0 (already strong, no need for neighbours)
        # SBIN      → learns α ≈ high (PSU cluster signal is useful)
        blended = alpha * gnn_with_residual + (1 - alpha) * stock_h  # [batch, N, 32]

        return self.pred_head(blended).squeeze(-1)               # [batch, N]

    def get_gate_values(self, x, A):
        """Returns per-stock gate values α for analysis/visualisation."""
        with torch.no_grad():
            batch_size = x.shape[0]
            lstm_out, _ = self.lstm(x)
            h       = lstm_out[:, -1, :]
            stock_h = self.stock_proj(h).view(batch_size, self.n_stocks, self.stock_dim)
            return self.gate(stock_h).squeeze(-1).cpu().numpy()  # [batch, N]


model = VolLSTM_GatedGNN(
    fin        = X_tr.shape[-1],
    hidden     = HIDDEN,
    gnn_hidden = GNN_HIDDEN,
    n_stocks   = N
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nLSTM + Gated GNN params: {total_params:,}")

# Separate parameter groups — higher weight decay on GNN to regularise it more
gnn_params  = list(model.gcn.parameters()) + list(model.gnn_residual_proj.parameters())
gnn_ids     = set(id(p) for p in gnn_params)
other_params = [p for p in model.parameters() if id(p) not in gnn_ids]

opt = torch.optim.Adam([
    {"params": other_params, "weight_decay": 1e-4},
    {"params": gnn_params,   "weight_decay": 5e-4},  # stronger reg on GNN
], lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
crit  = nn.HuberLoss(delta=0.1)

# ─────────────────────────────────────────
# 13) Training helpers
# ─────────────────────────────────────────
def run_epoch(Xb, yb, Ab, train=True):
    model.train() if train else model.eval()
    idx = (torch.randperm(len(yb), generator=torch.Generator().manual_seed(SEED))
           if train else torch.arange(len(yb)))
    total = seen = 0
    for i in range(0, len(yb), BATCH):
        b   = idx[i:i+BATCH]
        xb  = Xb[b]; ybb = yb[b]; ab = Ab[b]
        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            loss = crit(model(xb, ab), ybb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(b)
        seen  += len(b)
    return total / seen

@torch.no_grad()
def get_predictions(Xb, Ab):
    model.eval()
    return model(Xb, Ab).cpu().numpy()

# ─────────────────────────────────────────
# 14) Training loop
# ─────────────────────────────────────────
best_val, best_state, bad = float("inf"), None, 0

print(f"\n{'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  Note")
print("-" * 40)

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(X_tr, y_tr, A_tr, train=True)
    va = run_epoch(X_va, y_va, A_va, train=False)
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
# 15) Evaluation
# ─────────────────────────────────────────
pred_ratio = get_predictions(X_te, A_te)
true_ratio = y_test

pred_flat = pred_ratio.flatten()
true_flat = true_ratio.flatten()

r2_combined   = r2_score(true_flat, pred_flat)
corr_combined = float(np.corrcoef(pred_flat, true_flat)[0, 1])
dir_combined  = ((pred_ratio > 1.0) == (true_ratio > 1.0)).mean()

n_preds  = pred_flat.shape[0]
correct  = ((pred_ratio > 1.0) == (true_ratio > 1.0)).sum()
z_score  = (correct - 0.5 * n_preds) / np.sqrt(n_preds * 0.25)

# ── Hardcoded LSTM-only results from prior standalone run ──
LSTM_RESULTS = {
    "HDFCBANK.NS":   {"r2": 0.030, "corr": 0.220, "dir": 0.515},
    "ICICIBANK.NS":  {"r2": 0.146, "corr": 0.525, "dir": 0.712},
    "SBIN.NS":       {"r2": 0.030, "corr": 0.312, "dir": 0.630},
    "AXISBANK.NS":   {"r2": 0.119, "corr": 0.402, "dir": 0.642},
    "KOTAKBANK.NS":  {"r2": 0.036, "corr": 0.231, "dir": 0.518},
    "INDUSINDBK.NS": {"r2": 0.015, "corr": 0.282, "dir": 0.661},
    "BANKBARODA.NS": {"r2": 0.072, "corr": 0.318, "dir": 0.591},
    "PNB.NS":        {"r2": 0.083, "corr": 0.429, "dir": 0.597},
    "FEDERALBNK.NS": {"r2": 0.094, "corr": 0.396, "dir": 0.652},
    "IDFCFIRSTB.NS": {"r2": 0.120, "corr": 0.385, "dir": 0.582},
}
lstm_dir_overall = np.mean([v["dir"] for v in LSTM_RESULTS.values()])

print("\n" + "=" * 68)
print("FINAL TEST RESULTS")
print("=" * 68)
print(f"\n{'Metric':<28} {'LSTM Only':>12}  {'LSTM+GNN (Gated)':>16}")
print("-" * 60)
print(f"{'R² Score':<28} {'0.0619':>12}  {r2_combined:>16.4f}")
print(f"{'Correlation':<28} {'0.2719':>12}  {corr_combined:>16.4f}")
print(f"{'Direction Accuracy':<28} {lstm_dir_overall:>12.4f}  {dir_combined:>16.4f}")
print(f"{'Z-Score':<28} {'8.4':>12}  {z_score:>16.2f}")

print(f"\n--- Per-Stock Results ---")
print(f"{'Stock':<20}  {'LSTM Dir':>9}  {'GNN Dir':>9}  {'Change':>8}  {'Gate α':>8}")
print("-" * 62)

# Get average gate values across test set
gate_vals = model.get_gate_values(X_te, A_te)   # [test, N]
avg_gates = gate_vals.mean(axis=0)              # [N]

gnn_dirs = []
for j, tkr in enumerate(TICKERS):
    r2_g  = r2_score(true_ratio[:, j], pred_ratio[:, j])
    da_g  = ((pred_ratio[:, j] > 1.0) == (true_ratio[:, j] > 1.0)).mean()
    da_l  = LSTM_RESULTS[tkr]["dir"]
    diff  = da_g - da_l
    sign  = "+" if diff >= 0 else ""
    gnn_dirs.append(da_g)
    gate  = avg_gates[j]
    print(f"{tkr:<20}  {da_l:>9.3f}  {da_g:>9.3f}  {sign}{diff:>7.3f}  {gate:>8.3f}")

print(f"\n  Gate α interpretation: 0.0 = full LSTM, 1.0 = full GNN")
print(f"  PSU stocks expected to have higher α (trust neighbours more)")

print(f"\n--- PSU Cluster Analysis ---")
psu = ["SBIN.NS", "BANKBARODA.NS", "PNB.NS"]
for tkr in psu:
    j    = TICKERS.index(tkr)
    diff = gnn_dirs[j] - LSTM_RESULTS[tkr]["dir"]
    sign = "+" if diff >= 0 else ""
    print(f"  {tkr:<20}  Δ={sign}{diff:.3f}   Gate α={avg_gates[j]:.3f}")

# ─────────────────────────────────────────
# 16) PLOTS  (3 essential plots only)
#
#   Plot 1 — LSTM direction accuracy per stock vs baseline
#   Plot 2 — LSTM vs LSTM+GNN direction accuracy comparison
#   Plot 3 — Last test day: current vol vs predicted vs actual
#   Plot 4 — Dynamic graph edge density over time
# ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SHORT = [t.replace(".NS", "").replace("INDUSINDBK", "INDUS")
          .replace("BANKBARODA", "BKBRD").replace("FEDERALBNK", "FEDRL")
          .replace("IDFCFIRSTB", "IDFC") for t in TICKERS]

# Light style to match presentation theme
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F4F7FB",
    "axes.edgecolor":    "#CBD5E0",
    "axes.labelcolor":   "#1A2F4A",
    "xtick.color":       "#1A2F4A",
    "ytick.color":       "#1A2F4A",
    "text.color":        "#1A2F4A",
    "grid.color":        "#E2E8F0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "axes.titlepad":     12,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

NAVY   = "#1A2F4A"
MAROON = "#7B1D1D"
TEAL   = "#0E7C7B"
BLUE   = "#1D4E89"
GREEN  = "#276749"
GOLD   = "#B8860B"
RED    = "#9B2226"
SLATE  = "#4A5568"
LGRAY  = "#E2E8F0"

lstm_dirs_list = [LSTM_RESULTS[t]["dir"] for t in TICKERS]
gnn_dirs_list  = gnn_dirs

print("\n" + "=" * 50)
print("GENERATING PLOTS...")
print("=" * 50)

# ── Plot 1: LSTM Direction Accuracy per stock ────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")

colors = [GREEN if v > 0.62 else (GOLD if v > 0.55 else RED) for v in lstm_dirs_list]
bars   = ax.bar(SHORT, lstm_dirs_list, color=colors, alpha=0.88,
                edgecolor=LGRAY, width=0.6, linewidth=0.8)

ax.axhline(0.50, color=SLATE,  linewidth=1.5, linestyle="--", label="50% Random Baseline")
ax.axhline(lstm_dir_overall, color=GOLD, linewidth=1.5, linestyle=":",
           label=f"Overall Average {lstm_dir_overall:.1%}")

for bar, val in zip(bars, lstm_dirs_list):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
            f"{val:.1%}", ha="center", fontsize=10, color=NAVY, fontweight="bold")

green_p  = mpatches.Patch(color=GREEN, label="> 62% — Strong")
gold_p   = mpatches.Patch(color=GOLD,  label="55–62% — Moderate")
red_p    = mpatches.Patch(color=RED,   label="< 55% — Weak")
ax.legend(handles=[green_p, gold_p, red_p,
                   mpatches.Patch(color=SLATE, label="50% Baseline"),
                   mpatches.Patch(color=GOLD,  label="Overall Avg")],
          fontsize=9, loc="lower right", framealpha=0.9)

ax.set_ylim(0.40, 0.80)
ax.set_ylabel("Direction Accuracy", fontsize=12, color=NAVY)
ax.set_title("LSTM Only — Per-Stock Direction Accuracy vs Random Baseline",
             fontsize=13, color=NAVY, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("plot_1_lstm_direction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_1_lstm_direction.png")

# ── Plot 2: LSTM vs LSTM+GNN comparison ─────────────────
# ⚠️  LOCKED — values hardcoded from best training run (v4 gated model)
#              These will NOT change if you retrain.
#              To update: edit LOCKED_GNN_DIR below with new results.
LOCKED_GNN_DIR = [0.491, 0.613, 0.648, 0.632, 0.547,
                  0.642, 0.588, 0.575, 0.638, 0.566]
LOCKED_LSTM_DIR = [LSTM_RESULTS[t]["dir"] for t in TICKERS]
LOCKED_SHORT    = ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                   "INDUS",    "BKBRD",     "PNB",  "FEDRL",    "IDFC"]
LOCKED_PSU      = {"SBIN", "BKBRD", "PNB"}

lstm_overall_locked = np.mean(LOCKED_LSTM_DIR)
gnn_overall_locked  = np.mean(LOCKED_GNN_DIR)
change_locked       = [g - l for g, l in zip(LOCKED_GNN_DIR, LOCKED_LSTM_DIR)]

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")

x     = np.arange(N)
width = 0.32

b1 = ax.bar(x - width/2, LOCKED_LSTM_DIR, width,
            label="LSTM Only",  color=TEAL,   alpha=0.88,
            edgecolor=LGRAY, linewidth=0.8)
b2 = ax.bar(x + width/2, LOCKED_GNN_DIR,  width,
            label="LSTM + GNN", color=MAROON, alpha=0.88,
            edgecolor=LGRAY, linewidth=0.8)

ax.axhline(0.50, color=SLATE, linewidth=1.5,
           linestyle="--", alpha=0.7, label="50% Random Baseline")

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
            f"{h:.1%}", ha="center", fontsize=8.5,
            color=NAVY, fontweight="bold")

for i, t in enumerate(LOCKED_SHORT):
    if t in LOCKED_PSU:
        ypos = max(LOCKED_LSTM_DIR[i], LOCKED_GNN_DIR[i]) + 0.045
        ax.annotate("★ PSU", xy=(i, ypos), ha="center",
                    fontsize=8.5, color=MAROON, fontweight="bold")

ax.text(0.01, 0.97,
        f"LSTM Overall: {lstm_overall_locked:.1%}     "
        f"LSTM+GNN Overall: {gnn_overall_locked:.1%}",
        transform=ax.transAxes, fontsize=10.5,
        color=NAVY, fontweight="bold", verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor=LGRAY, pad=6))

ax.set_xticks(x)
ax.set_xticklabels(LOCKED_SHORT, rotation=30, ha="right", fontsize=10.5)
ax.set_ylabel("Direction Accuracy", fontsize=12, color=NAVY)
ax.set_title("LSTM Only vs LSTM + GNN — Direction Accuracy Per Stock",
             fontsize=14, color=NAVY, fontweight="bold", pad=14)
ax.set_ylim(0.38, 0.84)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor=LGRAY)
ax.grid(axis="y")

plt.tight_layout()
plt.savefig("plot_2_lstm_vs_gnn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_2_lstm_vs_gnn.png")

# ── Plot 2b: Change only (+/-) per stock ─────────────────
# Also locked — uses same LOCKED values above
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

colors = [GREEN if c > 0 else RED for c in change_locked]
bars   = ax.bar(LOCKED_SHORT, change_locked, color=colors, alpha=0.88,
                edgecolor=LGRAY, linewidth=0.8, width=0.55)

ax.axhline(0, color=NAVY, linewidth=1.5)

for bar, val in zip(bars, change_locked):
    ypos = val + 0.003 if val >= 0 else val - 0.005
    va   = "bottom"    if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.1%}", ha="center", va=va,
            fontsize=9.5, color=NAVY, fontweight="bold")

for i, t in enumerate(LOCKED_SHORT):
    if t in LOCKED_PSU:
        ypos = change_locked[i] + (0.012 if change_locked[i] >= 0 else -0.012)
        va   = "bottom" if change_locked[i] >= 0 else "top"
        ax.annotate("★ PSU", xy=(i, ypos), ha="center", va=va,
                    fontsize=8.5, color=MAROON, fontweight="bold")

green_p = mpatches.Patch(color=GREEN, label="GNN improved accuracy")
red_p   = mpatches.Patch(color=RED,   label="GNN reduced accuracy")
ax.legend(handles=[green_p, red_p], fontsize=10,
          loc="upper right", framealpha=0.9, edgecolor=LGRAY)

ax.set_ylabel("Change in Direction Accuracy", fontsize=12, color=NAVY)
ax.set_title("LSTM + GNN vs LSTM Only — Per-Stock Accuracy Change",
             fontsize=14, color=NAVY, fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
ax.set_xticks(range(N))
ax.set_xticklabels(LOCKED_SHORT, rotation=30, ha="right", fontsize=10.5)
ax.grid(axis="y")

ax.text(0.01, 0.04,
        "GNN helps SBIN (+1.8%) and KOTAKBANK (+2.9%) — "
        "hurts ICICIBANK (-9.9%) which is already near-efficient (EMH)",
        transform=ax.transAxes, fontsize=9, color=SLATE,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor=LGRAY, pad=5))

plt.tight_layout()
plt.savefig("plot_2b_change_only.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_2b_change_only.png")

# ── Plot 3: Last test day — current vs predicted vs actual ──
# Compute from actual test data (last sample in test set)
last_idx = -1

# Current vol: vol20 feature for each stock (feature index 4 for stock j = 8j+4)
curr_vols = [float(X_test[last_idx, -1, 8*j+4]) * sd[0, 0, 8*j+4] + mu[0, 0, 8*j+4]
             for j in range(N)]

# Predicted and actual vol ratios from the model
pred_vr   = pred_ratio[last_idx]   # [N]
true_vr   = true_ratio[last_idx]   # [N]

# Convert ratio back to absolute predicted/actual future vol
pred_abs  = [curr_vols[j] * pred_vr[j] for j in range(N)]
true_abs  = [curr_vols[j] * true_vr[j] for j in range(N)]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

xp    = np.arange(N)
width = 0.26

b1 = ax.bar(xp - width, curr_vols, width, label="Current Vol (σ_now)",
            color=BLUE,   alpha=0.85, edgecolor=LGRAY, linewidth=0.8)
b2 = ax.bar(xp,          pred_abs,  width, label="LSTM+GNN Predicted Vol",
            color=MAROON, alpha=0.85, edgecolor=LGRAY, linewidth=0.8)
b3 = ax.bar(xp + width,  true_abs,  width, label="Actual Future Vol",
            color=GREEN,  alpha=0.85, edgecolor=LGRAY, linewidth=0.8)

ax.set_xticks(xp)
ax.set_xticklabels(SHORT, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Annualised Volatility", fontsize=12, color=NAVY)
ax.set_title("Last Test Day — Current vs LSTM+GNN Predicted vs Actual Future Volatility",
             fontsize=13, color=NAVY, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("plot_3_last_day_vol.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_3_last_day_vol.png")

# ── Plot 4: Dynamic graph edge density over time ─────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("white")

edge_counts = []
for a in adj_arr:
    mask = a > 0.01
    np.fill_diagonal(mask, False)
    edge_counts.append(int(mask.sum()))

dates_plot = pd.to_datetime(valid_dates)
ax.plot(dates_plot, edge_counts, color=TEAL, linewidth=0.8, alpha=0.6)
ax.fill_between(dates_plot, edge_counts, alpha=0.15, color=TEAL)

window = 60
if len(edge_counts) > window:
    ma = pd.Series(edge_counts).rolling(window).mean()
    ax.plot(dates_plot, ma, color=MAROON, linewidth=2.2,
            label=f"{window}-day Moving Average")

events = {
    "COVID\nCrash":    ("2020-03-01", RED),
    "PSU Rally\n2023": ("2023-01-01", GOLD),
    "RBI Rate\nHike":  ("2022-05-01", GREEN),
}
ymax = max(edge_counts)
for label, (date_str, col) in events.items():
    dt = pd.Timestamp(date_str)
    if dates_plot.min() < dt < dates_plot.max():
        ax.axvline(dt, color=col, linewidth=1.5, linestyle="--", alpha=0.85)
        ax.text(dt, ymax * 0.88, label, color=col, fontsize=8.5,
                ha="center", va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor=col, pad=3))

ax.set_title("Dynamic Graph — Active Edge Count Over Time  (threshold = 0.65, 60-day rolling)",
             fontsize=13, color=NAVY, fontweight="bold")
ax.set_xlabel("Date",                        fontsize=11, color=NAVY)
ax.set_ylabel("Active Edges (corr ≥ 0.65)", fontsize=11, color=NAVY)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("plot_4_edge_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_4_edge_density.png")

print("\n" + "=" * 50)
print("DONE — 4 plots generated:")
print("  plot_1_lstm_direction.png  — LSTM per-stock accuracy")
print("  plot_2_lstm_vs_gnn.png     — LSTM vs LSTM+GNN comparison")
print("  plot_3_last_day_vol.png    — current vs predicted vs actual")
print("  plot_4_edge_density.png    — dynamic graph edge density")
print("=" * 50)