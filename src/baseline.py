"""Baseline: sentence-transformer embeddings + logistic regression.

Cheap, interpretable reference point. If a frozen general-purpose embedding
plus a linear classifier already separates the classes well, that tells us
how much signal is in surface semantics alone.

Run:  python -m src.baseline
"""

import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.evaluate import report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    encoder = SentenceTransformer(EMBED_MODEL)
    X_train = encoder.encode(train["text"].tolist(), show_progress_bar=True,
                             batch_size=64)
    X_test = encoder.encode(test["text"].tolist(), show_progress_bar=True,
                            batch_size=64)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train["label"])

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    report("baseline_embed_lr", test, y_pred, y_score)


if __name__ == "__main__":
    main()
