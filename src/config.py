"""Central configuration for the RiskLens volatility pipeline.

All paths, hyperparameters, and the investable universe are defined here so
that data.py, features.py, graph.py, dataset.py, models.py, train.py,
portfolio.py and regime.py all agree on a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "prices.csv")
RAW_VOLUME_PATH = os.path.join(DATA_DIR, "raw", "volume.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats.json")
LSTM_RESULTS_PATH = os.path.join(RESULTS_DIR, "lstm_results.json")
GNN_RESULTS_PATH = os.path.join(RESULTS_DIR, "gnn_results.json")
PORTFOLIO_RESULTS_PATH = os.path.join(RESULTS_DIR, "portfolio_weights.json")

LSTM_WEIGHTS_PATH = os.path.join(MODELS_DIR, "lstm.pt")
GNN_WEIGHTS_PATH = os.path.join(MODELS_DIR, "gnn.pt")

# NSE-listed large caps (Yahoo Finance ".NS" suffix). 18 names across
# banking, IT, energy, FMCG, telecom, pharma, auto, cement, metals and
# power keep the correlation graph diverse across sectors while still
# training fast.
TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "MARUTI.NS",
    "LT.NS",
    "ULTRACEMCO.NS",
    "TATASTEEL.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
]

START_DATE = "2011-01-01"  # ~15 years of history
# End date is left as None -> "today" at data-download time.
END_DATE = None

SEED = 42

# --- Feature / target windows ---
RETURN_WINDOW = 60          # trading days of history fed to the models
VOL_WINDOW = 5               # rolling window (days) used to realise volatility
HORIZON = 5                  # days ahead the target volatility ratio looks
MOMENTUM_WINDOW = 20          # trading days of cumulative return used as a momentum feature
VOLUME_ZSCORE_WINDOW = 20     # rolling window used to z-score traded volume
RSI_WINDOW = 14                # standard Wilder RSI lookback

# --- Train/val/test split (chronological) ---
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remaining fraction is the test split

# --- Model hyperparameters ---
LSTM_HIDDEN = 48
LSTM_LAYERS = 1
GNN_HIDDEN = 48
GNN_LAYERS = 2
DROPOUT = 0.15

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# --- Graph construction ---
GRAPH_WINDOW = 60            # trading days used to estimate the correlation graph
GRAPH_CORR_THRESHOLD = 0.0   # edges below this absolute correlation are zeroed

# --- Regime classification ---
# Quantiles of trailing realised market volatility (equal-weight average
# across the universe) used to bucket "calm" / "normal" / "crisis".
REGIME_CALM_QUANTILE = 0.33
REGIME_CRISIS_QUANTILE = 0.85
REGIME_LOOKBACK = 252 * 3    # ~3 years of history to build the quantile bands

# --- Portfolio optimisation ---
PORTFOLIO_LONG_ONLY = True
PORTFOLIO_MAX_WEIGHT = 0.35   # single-name cap
PORTFOLIO_RIDGE = 1e-6        # numerical stabiliser added to the diagonal of Sigma

FEATURE_COLUMNS: List[str] = [
    "log_return", "realized_vol", "momentum", "volume_zscore", "rsi", "macd_hist",
]

# Sector labels, used only for dashboard grouping/display -- no model or
# optimizer logic depends on these.
SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking",
    "SUNPHARMA.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "MARUTI.NS": "Auto",
    "LT.NS": "Capital Goods",
    "ULTRACEMCO.NS": "Cement",
    "TATASTEEL.NS": "Metals",
    "NTPC.NS": "Power",
    "NESTLEIND.NS": "FMCG",
}


@dataclass
class Paths:
    root: str = ROOT_DIR
    data_dir: str = DATA_DIR
    raw_prices: str = RAW_DATA_PATH
    results_dir: str = RESULTS_DIR
    models_dir: str = MODELS_DIR


def ensure_dirs() -> None:
    """Create data/results/models directories if they don't exist yet."""
    for d in (os.path.join(DATA_DIR, "raw"), RESULTS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
