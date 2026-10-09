# RiskLens — Model Metrics (MAE / MSE / Direction Accuracy)

**Generated:** 2026-10-09, from GitHub Actions run [`train_models.yml` #6](https://github.com/umasuthanpalaniappan/RiskLens/actions/runs/37892161914)
**Trained on:** real NSE data, 2011-01-03 → 2026-10-08 (3,889 trading days)
**Source files:** [`results/lstm_results.json`](https://github.com/umasuthanpalaniappan/RiskLens/blob/main/results/lstm_results.json), [`results/gnn_results.json`](https://github.com/umasuthanpalaniappan/RiskLens/blob/main/results/gnn_results.json)

All error values are in **vol-ratio units** — the prediction target is `forward_5day_realised_vol / trailing_realised_vol`, not a percentage or a price. A value centered around 1.0 means "volatility unchanged"; the model is scored on how far off its ratio prediction is from the actual ratio.

## LSTM baseline

| Metric | Value |
|---|---|
| Test MAE | **0.5342** |
| Test MSE | **0.7392** |
| Test direction accuracy | 71.3% |
| Test samples | 10,422 |

## Dynamic GNN (three internal heads)

| Head | Test MAE | Test MSE | Direction accuracy |
|---|---|---|---|
| **Blended (deployed)** | **0.5294** | **0.7572** | **70.9%** |
| LSTM-only (inside GNN) | 0.7832 | 1.4129 | 68.7% |
| Graph-only | 0.6794 | 1.4230 | 50.3% (≈coin flip) |

Test samples: 579.

## Reading MAE vs. MSE together

MSE is consistently higher than MAE² for every row above (e.g. blended: MSE 0.757 vs. MAE² ≈ 0.280) — that gap is the signature of a **fat-tailed error distribution**: the model is usually off by a moderate amount (what MAE reports), but occasionally misses by a lot (likely during sharp volatility spikes), and those rare large misses get squared and dominate MSE. This matches what you'd expect from a target built on realised volatility, which is well known to spike during market stress.

## Why did the numbers move since the last run (Aug 28)?

Two genuine, verifiable reasons — not noise and not a regression:

1. **The dataset grew.** This run trained on data through **2026-10-08** instead of 2026-08-27 — 111 more real trading days. Since the train/val/test split is chronological, the test window itself shifted to a different, more recent slice of the market. Comparing MSE/MAE/accuracy across two runs is partly comparing performance on two different test periods, not just two different models.
2. **Model weights aren't seeded.** `src/train.py` doesn't call `torch.manual_seed()`, so each retrain starts from different random initial weights (confirmed by checking the code — this is a real gap, not a guess). Some run-to-run variation in the final numbers is expected from that alone.

Net effect this time: LSTM baseline **improved** (MSE 0.848 → 0.739, MAE now measured at 0.534, accuracy 70.1% → 71.3%). GNN blended MSE also **improved** (0.845 → 0.757) and MAE is now measured (0.529), though blended direction accuracy dipped slightly (72.1% → 70.9%). Nothing here indicates a bug — this is normal run-to-run variation on a genuinely growing, continuously-updating real dataset.

## Honest takeaway

The graph-only head still scores ~50% direction accuracy — essentially a coin flip on its own. The blended model's edge over a coin flip, and its slight edge over the LSTM-only internal head, is coming from the learned gate leaning on the LSTM signal for most stocks while selectively trusting the graph for a few. See the live [Model Performance tab](https://risk-lens-orcin.vercel.app/#dashboard) and the per-stock gate alpha chart for the current breakdown — these numbers refresh automatically every time the pipeline retrains.

---
*This file is a point-in-time snapshot. For always-current numbers, check the live dashboard or `results/*.json` on `main` directly — both update automatically.*
