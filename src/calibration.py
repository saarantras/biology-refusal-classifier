"""Quantify how confident the classifier's mistakes are.

A safety filter with an "escalate low-confidence cases to human review" band
only works if its errors are *uncertain*. This script measures whether they
are: confidence of errors vs. correct predictions, calibration error, and a
risk-coverage analysis (does abstaining on low-confidence inputs remove the
errors?).

Run:  python -m src.calibration
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODELS = [("baseline_embed_lr", "Embeddings + LR"), ("distilbert", "DistilBERT")]


def expected_calibration_error(conf, correct, n_bins=10):
    """ECE: average gap between confidence and accuracy, binned by confidence."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def analyze(name, title):
    df = pd.read_csv(os.path.join(OUT_DIR, f"{name}_predictions.csv"))
    p = df["refuse_score"].to_numpy()          # P(refuse)
    y = df["label"].to_numpy()
    pred = df["pred"].to_numpy()
    conf = np.maximum(p, 1 - p)                 # confidence in the chosen class
    correct = (pred == y)
    err = ~correct

    print(f"\n{'=' * 64}\n{title}  (n={len(df)}, errors={err.sum()})\n{'=' * 64}")
    print(f"Brier score (refuse prob):        {brier_score_loss(y, p):.4f}")
    print(f"Expected calibration error (ECE): {expected_calibration_error(conf, correct):.4f}")
    print(f"Mean confidence | correct:        {conf[correct].mean():.4f}")
    print(f"Mean confidence | error:          {conf[err].mean():.4f}")

    print("\nError confidence distribution:")
    for t in (0.90, 0.95, 0.99):
        n = (err & (conf >= t)).sum()
        print(f"  errors with confidence >= {t:.2f}:  {n:2d} / {err.sum()} "
              f"({100 * n / err.sum():.0f}%)")
    print(f"  error rate among conf>=0.90 predictions: "
          f"{err[conf >= 0.90].sum()}/{(conf >= 0.90).sum()} "
          f"({100 * err[conf >= 0.90].mean():.2f}%)")

    # Risk-coverage: abstain on the least-confident inputs, see how many errors
    # that removes. Report the coverage needed to drop each error count.
    order = np.argsort(conf)              # least confident first
    err_sorted = err[order]
    cum_err_removed = np.cumsum(err_sorted)
    print("\nAbstention (escalate least-confident inputs to review):")
    for target in (3, 6, 9, err.sum()):
        # smallest number of abstentions that removes >= target errors
        k = int(np.searchsorted(cum_err_removed, target) + 1)
        k = min(k, len(df))
        print(f"  to catch {target:2d}/{err.sum()} errors: abstain on "
              f"{k} inputs ({100 * k / len(df):.1f}% of traffic)")


if __name__ == "__main__":
    for name, title in MODELS:
        analyze(name, title)
