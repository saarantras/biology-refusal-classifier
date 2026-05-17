"""Build the biology refusal classification dataset.

Positive ("refuse") class: WMDP-bio question stems (cais/wmdp), answer choices
dropped so the classifier sees only the natural-language query.

Negative ("don't refuse") class: benign biology and medical exam questions
drawn from MMLU. These are deliberately *hard* negatives -- same exam-question
format and same broad domain as WMDP-bio -- so the classifier is forced to
separate hazardous biology from ordinary biology, rather than learning a
trivial "is this about biology / is this a question" shortcut.

Run:  python -m src.data
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# MMLU subjects used as benign negatives. Chosen to mirror WMDP-bio's domain:
# molecular/cellular biology, genetics, virology, anatomy, and clinical
# knowledge. All are everyday academic content with no operational uplift.
MMLU_NEG_SUBJECTS = [
    "high_school_biology",
    "college_biology",
    "anatomy",
    "medical_genetics",
    "virology",
    "college_medicine",
    "clinical_knowledge",
    "professional_medicine",
    "human_aging",
    "nutrition",
]

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_positives() -> pd.DataFrame:
    """WMDP-bio question stems -> refuse class (label 1)."""
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    questions = [q.strip() for q in ds["question"]]
    df = pd.DataFrame({"text": questions})
    df["label"] = 1
    df["source"] = "wmdp-bio"
    return df


def load_negatives() -> pd.DataFrame:
    """Benign MMLU biology/medical question stems -> don't-refuse class (label 0)."""
    frames = []
    for subject in MMLU_NEG_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        questions = [q.strip() for q in ds["question"]]
        sub = pd.DataFrame({"text": questions})
        sub["source"] = f"mmlu/{subject}"
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = 0
    return df


def build(out_dir: str = DATA_DIR, test_size: float = 0.2) -> None:
    pos = load_positives()
    neg = load_negatives()
    print(f"positives (wmdp-bio): {len(pos)}")
    print(f"negatives (mmlu bio/medical, raw): {len(neg)}")

    # Balance: subsample negatives to match the positive count so neither
    # class metric is dominated by prior. Sample the same fraction from every
    # subject so domain coverage across MMLU subjects is preserved.
    if len(neg) > len(pos):
        neg = neg.groupby("source", group_keys=False).sample(
            frac=len(pos) / len(neg), random_state=SEED
        )
    print(f"negatives (after balancing): {len(neg)}")

    df = pd.concat([pos, neg], ignore_index=True)
    # Drop empties and exact-duplicate stems.
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")

    train, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label"]
    )
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"\ntrain: {len(train)}  (refuse={train.label.sum()}, "
          f"benign={(train.label == 0).sum()})")
    print(f"test:  {len(test)}  (refuse={test.label.sum()}, "
          f"benign={(test.label == 0).sum()})")
    print(f"written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    build(test_size=args.test_size)
