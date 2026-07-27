#!/usr/bin/env python3
"""Train the single-stock LSTM volatility baseline and save results/lstm.pt.

Usage: python scripts/run_lstm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import train


def main() -> None:
    results = train.train_lstm()
    print("LSTM training complete.")
    print(f"  test MSE:                {results['test_mse']}")
    print(f"  test direction accuracy: {results['test_direction_accuracy']}")
    print(f"  weights saved to models/lstm.pt")
    print(f"  results saved to results/lstm_results.json")


if __name__ == "__main__":
    main()
