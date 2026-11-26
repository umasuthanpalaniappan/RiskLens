# RiskLens roadmap and working agreement

A running list of worthwhile improvements, worked through a few at a time by
small, verified commits. Newest done items go at the top of **Done**.

## Working agreement (for anyone, human or automated, making changes)

- Make **real improvements only**: correctness, tests, reproducibility, clarity, the dashboard, honest documentation. No padding commits (whitespace churn, empty commits, renaming for its own sake).
- **One focused change per commit**, with a message that says what changed and why. Run `python -m pytest` before every commit; for workflow or web changes also parse the YAML / load the page and check the console.
- **Never edit what the daily bot owns**: `results/`, `web/data.json`, `web/status.json`, `dataset/*.csv`, `models/*.pt`. Do not retrain or regenerate numbers to change what the project reports.
- **Never fabricate or tune results.** Documentation states what the code and the latest `docs/metrics.md` actually show, including the unflattering parts (the blend is comparable to the LSTM, not better).
- Keep the daily workflow healthy: it must keep passing after any change to `src/`, `scripts/`, `requirements.txt` or `.github/workflows/`.
- Pull `main` (the bot commits daily) and rebase before pushing; never force-push `main`.

## Backlog (rough priority order)

1. **Walk-forward backtest** of the minimum-variance portfolio against equal weight and the NIFTY 50, after costs. The biggest open gap: forecasts have not been shown to lower realised risk.
2. **Multi-seed evaluation**: a script that retrains with several seeds and reports mean ± spread for direction accuracy / MAE / MSE, so small differences can be judged against noise. Add a bootstrap confidence interval for direction accuracy to `evaluate.py`.
3. **Dashboard freshness banner**: read `web/status.json` and warn when the last run failed or the data is older than the last NSE session.
4. **Tests for `dataset.py`** (chronological split has no overlap; normalisation statistics come from the training span only; LSTM and GNN windows line up), a synthetic smoke test of `run_portfolio`, and a schema test for `export_dashboard_data`.
5. **Regenerate `docs/metrics.md` from `results/*.json`** with a script, so the numbers cannot drift from the files.
6. **Download resilience**: if Yahoo is down, fall back to the cached history with a clear status instead of failing; unit-test it.
7. **Lint and typing**: add ruff and a pre-commit config; tighten type hints; remove duplicated code between `train_lstm` and `train_gnn`.
8. **Replace deprecated `datetime.utcnow()`** in `src/train.py` (and anywhere else) with timezone-aware time.
9. **Docs**: fix the `graph_laplacian` docstring (an isolated node keeps a 1 on the diagonal of L, so it receives nothing from neighbours); add a Mermaid architecture diagram in the README showing the graph and the LSTM as parallel branches; add a short FAQ and glossary.
10. **Dashboard quality**: keyboard navigation and ARIA labels, a mobile-layout check with Playwright, and a CPU-light version of the network animation.
11. **Experiment (research)**: directed volatility-spillover graphs (Diebold–Yilmaz) in place of undirected correlation; report honestly whether they help.

## Done

- Tests for the optimiser, regime classifier, features (with a look-ahead guard), graph, metrics, models and data dates; CI on every push and pull request.
- Seeded training (`train.set_seed`) so retrains reproduce.
- Correct network caption on the dashboard (return correlation, not volatility correlation).
- Daily workflow: CPU-only torch, failure alert issue, status commit on every run, latest finished trading day included.
