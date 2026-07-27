#!/usr/bin/env python3
"""Train the dynamic-graph GNN volatility model and save results/gnn.pt.

Also (re)writes results/norm_stats.json with the mu/sd used to normalise
features and targets, so scripts/run_portfolio.py can reproduce identical
scaling at inference time without touching training data again.

Usage: python scripts/run_gnn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import train


def main() -> None:
    results = train.train_gnn()
    print("GNN training complete.")
    print(f"  test MSE (blended):                {results['test_mse_blended']}")
    print(f"  test direction accuracy (blended):  {results['test_direction_accuracy_blended']}")
    print(f"  test direction accuracy (LSTM-only):{results['test_direction_accuracy_lstm_only']}")
    print(f"  weights saved to models/gnn.pt")
    print(f"  results saved to results/gnn_results.json")
    print(f"  norm stats saved to results/norm_stats.json")


if __name__ == "__main__":
    main()
