# RiskLens training dataset

Real daily close price and traded volume for the 18-stock NSE universe
(`config.TICKERS`), pulled from Yahoo Finance. This is the exact data
the currently-committed `models/*.pt` were trained on -- regenerated and
committed here each time `train_models.yml` runs (unlike `data/raw/`,
which is a gitignored live cache, not a versioned artifact).

- `prices.csv` / `volume.csv`: date-indexed, one column per ticker
- Range: 2011-01-03 -> 2026-08-27 (3864 trading days)
- Regenerated: 2026-08-28 10:37 UTC
