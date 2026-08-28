# RiskLens

Stock volatility forecasting with an LSTM baseline and a dynamic graph
neural network (GNN), feeding a daily minimum-variance portfolio and a
Streamlit dashboard.

## What it does

1. **Forecast volatility.** For each stock in the universe, predict a
   *volatility ratio* — realised volatility over the next `HORIZON` trading
   days, relative to trailing realised volatility. Two models are trained
   and compared:
   - `LSTMVolModel` — a single-stock LSTM. Each stock is scored from its own
     return/volatility history only.
   - `GNNVolModel` — the same LSTM encoder, refined by a graph convolution
     over a **dynamic graph** that is re-estimated on a rolling 60-day
     window (so the graph structure itself evolves over time). Edges come
     from cross-stock **realised-volatility correlation**, not return
     correlation -- vol clustering/contagion is the structure that's
     actually relevant to a model predicting where volatility goes next,
     whereas return correlation mostly captures a different thing (shared
     market/sector beta). A learnable per-stock gate (`alpha`, in [0, 1])
     blends the graph-refined prediction with the plain LSTM prediction, so
     the model can fall back to the univariate signal for stocks the graph
     doesn't help.
2. **Build a portfolio.** Turn today's predicted volatilities into a
   long-only **minimum-variance** portfolio (see below for why not full
   Markowitz).
3. **Classify the regime.** Bucket today's market-wide volatility into
   calm / normal / crisis based on trailing quantiles.
4. **Show it.** A single-page Streamlit dashboard reads the JSON results and
   renders weights, per-stock detail, and a model-comparison chart.
5. **Automate it.** A daily GitHub Actions workflow re-runs the live
   portfolio step after the NSE market closes and commits any changed
   results.

## Why minimum-variance, not Markowitz

Classic Markowitz mean-variance optimisation needs a forecast of *expected
returns* as well as risk. RiskLens's models are trained to predict
**volatility**, not returns — there's no honest signal here for expected
return, and fabricating one just to run a "real" Markowitz optimiser would
mean optimizing against a made-up number. Instead, `portfolio.py` builds the
predicted covariance matrix Sigma from predicted volatilities and the
recent correlation structure, and solves for the long-only weights that
minimise portfolio variance `w' Sigma w` subject to full investment and a
per-name cap. A volatility model has no honest way to prefer one stock's
*return* over another's — it can only tell you which combination of stocks
is currently the least risky, which is exactly what min-variance computes.
The regime classification is used only as a light risk overlay (blending
the optimiser's output toward equal weight when the market looks crisis-y),
not a second forecasting model.

## Architecture

```
src/
  config.py     # single source of truth: universe, paths, hyperparameters
  data.py       # yfinance download + local CSV cache (data/raw/prices.csv)
  features.py   # log returns, realised vol, forward vol-ratio target
  graph.py      # rolling correlation adjacency, graph Laplacian
  dataset.py    # windowing, chronological train/val/test split, norm stats
  models.py     # LSTMVolModel, GNNVolModel (+ GraphConvLayer, gate alpha)
  train.py      # training loops shared by both scripts; saves weights + results
  evaluate.py   # MSE / direction-accuracy metrics (shared, no duplication)
  portfolio.py  # predicted_covariance(), optimize() — min-variance QP
  regime.py     # current_regime() — calm/normal/crisis quantile classifier

scripts/
  run_lstm.py       # trains LSTMVolModel -> models/lstm.pt, results/lstm_results.json
  run_gnn.py        # trains GNNVolModel  -> models/gnn.pt,  results/gnn_results.json,
                     #                       results/norm_stats.json (mu/sd for live inference)
  run_portfolio.py  # live daily inference: loads gnn.pt + norm_stats.json, pulls the
                     # latest prices, predicts vol, builds Sigma + the graph Laplacian
                     # from the most recent 60-day window, classifies the regime, solves
                     # the portfolio, writes results/portfolio_weights.json

app/
  dashboard.py  # Streamlit dashboard reading the three results JSON files

.github/workflows/
  daily_update.yml  # cron @ 12:30 UTC (18:00 IST) -> run_portfolio.py -> commit results/
```

Nothing is duplicated between scripts: all data prep, normalisation, model
definitions and metrics live in `src/`, and each entry-point script is a
thin wrapper around it.

## Data

`src/data.py` downloads daily adjusted-close prices *and* traded volume for
an 18-stock NSE universe (`config.TICKERS`, spanning banking, IT, energy,
FMCG, telecom, pharma, auto, cement, metals and power — see
`config.SECTOR_MAP`) via `yfinance` in a single request, caching each to
`data/raw/prices.csv` / `data/raw/volume.csv`. `run_lstm.py`/`run_gnn.py`
use the cache when present; `run_portfolio.py` always refreshes it, since
it needs today's close.

Each stock is represented by four features (`config.FEATURE_COLUMNS`):
daily log return, trailing realised volatility, `MOMENTUM_WINDOW`-day
cumulative return (trend), and a rolling z-score of traded volume (unusual
activity). All four feed both the LSTM and the GNN encoder.

### The committed dataset

`data/raw/` is a gitignored *live cache* — it exists so repeated local runs
don't re-hit the network, not as a versioned artifact. For a fixed,
inspectable copy of the actual data the currently-committed `models/*.pt`
were trained on, see [`dataset/`](dataset/): `dataset/prices.csv` and
`dataset/volume.csv` are a snapshot of the real 18-stock, ~15-year NSE
history, regenerated and committed automatically every time
`train_models.yml` runs (see `dataset/README.md` for the exact date range).

History spans `config.START_DATE` (2011-01-01, ~15 years) to today.
`run_portfolio.py`'s daily refresh always re-downloads that full span (not
just a short live window) before persisting the cache, specifically so the
dashboard's Market Overview tab keeps its full multi-year history instead
of being truncated down to a few hundred days every time the daily job
runs — it just returns the trailing window it actually needs to the caller.

If a ticker in `config.TICKERS` is ever delisted, renamed, or otherwise
stops resolving on Yahoo Finance (this happens — corporate actions like
mergers/demergers retire old symbols), `data.py` raises a clear error
naming the ticker rather than failing downstream with a cryptic empty-data
crash. Fix it by swapping the symbol in `config.TICKERS`/`config.SECTOR_MAP`
and retraining.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/run_lstm.py        # trains the LSTM baseline
python scripts/run_gnn.py         # trains the GNN, saves norm_stats.json
python scripts/run_portfolio.py   # live daily inference -> portfolio_weights.json
streamlit run app/dashboard.py    # view the dashboard
```

Run them in that order the first time — `run_portfolio.py` needs the GNN
weights and normalisation stats that `run_gnn.py` produces, and the
dashboard needs all three results files to exist.

> **Note on this repository's committed model weights.** The
> `models/*.pt` and `results/*.json` currently checked in were produced end
> to end, but with a *synthetic* price series rather than live NSE data —
> the sandbox this project was assembled in has outbound network access to
> Yahoo Finance blocked by policy, so live data could not be pulled from
> there. The full pipeline (data → features → graph → training → portfolio
> → dashboard) was verified to run cleanly with that synthetic data. Before
> trusting the numbers for anything real, re-run `run_lstm.py` and
> `run_gnn.py` locally (or let a manual Actions run do it) so the models are
> trained on real market history — `data/raw/` is gitignored precisely so
> the first real run pulls fresh data rather than reusing a stale cache.

## Daily automation

`.github/workflows/daily_update.yml` runs on a cron of `30 12 * * *` (UTC),
i.e. 18:00 IST, after the NSE market close. It installs
`requirements.txt`, runs `scripts/run_portfolio.py` (inference only — it
does not retrain the models), and commits `results/` back to the repo as
the `github-actions[bot]` identity, but only if something actually
changed. It does not retrain `models/*.pt`; retraining is a separate,
manual step (`run_lstm.py` / `run_gnn.py`) since it's compute-heavier and
the universe/graph structure doesn't need to be re-learned every day.

`.github/workflows/train_models.yml` is that manual retraining step, run as
a GitHub Actions job (`workflow_dispatch` only — no schedule) rather than
locally, since GitHub's runners have real internet access that some
development environments don't. It installs `requirements.txt`, runs
`scripts/run_lstm.py` then `scripts/run_gnn.py`, and commits
`models/*.pt` + `results/{lstm,gnn}_results.json` + `results/norm_stats.json`
as `github-actions[bot]`, only if anything changed. Trigger it from the
Actions tab whenever you want the models retrained on the latest history
(e.g. after adding tickers, changing features/hyperparameters, or simply to
refresh on more recent data).

## Dashboard

`app/dashboard.py` is a single-file Streamlit app with no browser storage —
every reload re-reads the on-disk results. It has three tabs:

- **Portfolio** — RiskLens title with a calm/normal/crisis regime badge, KPI
  tiles, portfolio weights (bar or pie, colour-coded by sector), a sector
  allocation chart, and a per-stock table (predicted volatility, portfolio
  weight, GNN gate alpha).
- **Model Performance** — LSTM vs. LSTM+GNN direction-accuracy comparison,
  a breakdown of the GNN's internal LSTM-only/GNN-only/blended heads, and
  the per-stock gate alpha.
- **Market Overview** — a correlation heatmap of the same rolling window
  the GNN's graph and the portfolio's Sigma are built from, a price-history
  and realised-vol trend chart per stock with a selectable lookback (1Y /
  3Y / 5Y / 10Y / Max, up to the full ~15-year cache), and a sortable
  universe snapshot table (last close + trailing vol for every stock, by
  sector). This tab reads the local `data/raw/` price/volume cache in
  addition to the three results JSONs, and degrades to a friendly message
  if that cache isn't present yet.

A last-updated timestamp is shown at the bottom of every tab.

## Regime classification

`regime.current_regime()` computes the equal-weight average trailing
realised volatility across the universe and buckets the latest value
against its own trailing-3-year quantiles: below the 33rd percentile is
"calm", above the 85th is "crisis", otherwise "normal". It's intentionally
simple and auditable rather than a latent-state model.
