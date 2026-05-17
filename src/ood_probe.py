"""Out-of-domain probe: how does the fine-tuned classifier behave on benign
prompts from topics it never saw in training?

The model is trained only on biology - WMDP-bio positives and MMLU bio/medical
negatives. A deployed refusal classifier sees mostly non-biology traffic, so
its behaviour there is untested extrapolation. This script scores the
fine-tuned DistilBERT on benign non-bio MMLU subjects (all true label
"don't refuse") and reports the false-positive rate per subject.

These prompts are NOT used for training or for the headline metrics - this is
an inference-only generalisation check.

Run:  python -m src.ood_probe   (after src.finetune has saved the model)
"""

import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs",
                         "distilbert_model")
MAX_LEN = 256

# Benign non-bio subjects, deliberately spanning humanities, social science,
# and non-bio STEM (chemistry/physics share the "science exam" register and
# may behave differently from pure humanities).
OOD_SUBJECTS = [
    "high_school_world_history", "marketing", "jurisprudence",
    "high_school_geography", "philosophy", "us_foreign_policy",
    "high_school_mathematics", "high_school_physics", "college_chemistry",
    "astronomy", "college_computer_science", "world_religions",
]


@torch.no_grad()
def score(texts, model, tokenizer, device, batch_size=64):
    """Return P(refuse) for each text."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit("No saved model. Run `python -m src.finetune` first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    rows = []
    per_question = []
    for subject in OOD_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        questions = [q.strip() for q in ds["question"]]
        p = score(questions, model, tokenizer, device)
        flagged = p >= 0.5                     # predicted "refuse"
        rows.append({
            "subject": subject,
            "n": len(questions),
            "false_positives": int(flagged.sum()),
            "fpr": flagged.mean(),
            "mean_refuse_score": p.mean(),
        })
        for q, s in zip(questions, p):
            per_question.append({"subject": subject, "text": q,
                                  "refuse_score": s})

    df = pd.DataFrame(rows).sort_values("fpr", ascending=False)
    total_n = df["n"].sum()
    total_fp = df["false_positives"].sum()

    print(f"\nOut-of-domain probe: {len(OOD_SUBJECTS)} non-bio MMLU subjects, "
          f"{total_n} benign prompts (true label = don't refuse)\n")
    print(f"{'subject':<30}{'n':>6}{'false_pos':>11}{'FPR':>9}"
          f"{'mean_score':>12}")
    print("-" * 68)
    for _, r in df.iterrows():
        print(f"{r['subject']:<30}{r['n']:>6}{r['false_positives']:>11}"
              f"{r['fpr']:>8.1%}{r['mean_refuse_score']:>12.3f}")
    print("-" * 68)
    print(f"{'OVERALL':<30}{total_n:>6}{total_fp:>11}"
          f"{total_fp / total_n:>8.1%}")

    out_dir = os.path.dirname(MODEL_DIR)
    df.to_csv(os.path.join(out_dir, "ood_probe.csv"), index=False)
    pd.DataFrame(per_question).to_csv(
        os.path.join(out_dir, "ood_predictions.csv"), index=False)
    print("\nwritten to outputs/ood_probe.csv, outputs/ood_predictions.csv")


if __name__ == "__main__":
    main()
