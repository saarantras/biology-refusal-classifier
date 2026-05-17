"""Shared evaluation: per-class metrics, confusion matrix, failure dump."""

import os

import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

LABEL_NAMES = {0: "don't refuse", 1: "refuse"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def report(name: str, df_test: pd.DataFrame, y_pred, y_score=None) -> dict:
    """Print and persist metrics for one model.

    df_test: test frame with columns text, label, source.
    y_pred:  predicted labels (0/1), aligned to df_test rows.
    y_score: optional probability of the refuse class, for failure analysis.
    """
    y_true = df_test["label"].to_numpy()
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(
        y_true, y_pred, target_names=["don't refuse", "refuse"], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix (rows=true, cols=pred):")
    print(f"               pred:don't  pred:refuse")
    print(f"  true:don't   {cm[0, 0]:>10}  {cm[0, 1]:>11}")
    print(f"  true:refuse  {cm[1, 0]:>10}  {cm[1, 1]:>11}")

    # Threshold-free ranking metrics (need scores, not just labels).
    if y_score is not None:
        print(f"\nAUROC:           {roc_auc_score(y_true, y_score):.4f}")
        print(f"AUPRC (refuse):  {average_precision_score(y_true, y_score):.4f}")

    rep = classification_report(
        y_true, y_pred, target_names=["don't refuse", "refuse"],
        digits=4, output_dict=True)

    # Persist full predictions (for figures) and misclassified rows (for
    # failure analysis).
    os.makedirs(OUT_DIR, exist_ok=True)
    preds = df_test.copy()
    preds["pred"] = y_pred
    if y_score is not None:
        preds["refuse_score"] = y_score
    preds.to_csv(os.path.join(OUT_DIR, f"{name}_predictions.csv"), index=False)

    wrong = preds[preds["label"] != preds["pred"]]
    path = os.path.join(OUT_DIR, f"{name}_misclassified.csv")
    wrong.to_csv(path, index=False)
    print(f"\n{len(wrong)} misclassified -> {os.path.relpath(path)}")
    return rep
