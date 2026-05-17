"""Generate figures for the write-up into outputs/.

Reads the per-model prediction CSVs written by evaluate.report(), so run
src.baseline and src.finetune first. The token-length histogram is computed
directly from the dataset.

Run:  python -m src.figures
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = [("baseline_embed_lr", "Embeddings + LR"), ("distilbert", "DistilBERT")]
LABELS = ["don't refuse", "refuse"]


def _load(name):
    return pd.read_csv(os.path.join(OUT_DIR, f"{name}_predictions.csv"))


def confusion_matrices():
    """Side-by-side confusion matrices for both models."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 4))
    for ax, (name, title) in zip(axes, MODELS):
        df = _load(name)
        ConfusionMatrixDisplay.from_predictions(
            df["label"], df["pred"], display_labels=LABELS,
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def pr_curves():
    """Precision-recall curves for the refuse class, both models."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, title in MODELS:
        df = _load(name)
        prec, rec, _ = precision_recall_curve(df["label"], df["refuse_score"])
        ax.plot(rec, prec, label=title)
    ax.set_xlabel("Recall (refuse class)")
    ax.set_ylabel("Precision (refuse class)")
    ax.set_title("Precision-Recall: refuse class")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pr_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def length_histogram():
    """Token-length distribution per class (DistilBERT tokenizer)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, "train.csv")),
                    pd.read_csv(os.path.join(DATA_DIR, "test.csv"))])
    df["ntok"] = [len(tok(t, truncation=False)["input_ids"]) for t in df["text"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 300, 46)
    for label, name, color in [(1, "refuse (WMDP-bio)", "tab:red"),
                               (0, "don't refuse (MMLU)", "tab:blue")]:
        ax.hist(df.loc[df.label == label, "ntok"].clip(upper=300), bins=bins,
                alpha=0.6, label=name, color=color)
    ax.axvline(256, color="k", ls="--", lw=1, label="MAX_LEN = 256")
    ax.set_xlabel("Tokens per prompt (clipped at 300)")
    ax.set_ylabel("Count")
    ax.set_title("Prompt length by class")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "length_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def quadrant_examples(n_per_cell: int = 3, seed: int = 0):
    """2x2 panel of example prompts from each confusion-matrix quadrant
    (DistilBERT), laid out to mirror Figure 1: TN/FP on top, FN/TP below."""
    import textwrap

    df = _load("distilbert")
    rng = np.random.default_rng(seed)

    # (true, pred): (title, is_error). Grid order matches the confusion matrix.
    cells = [
        [(0, 0, "True negative - benign, passed", False),
         (0, 1, "False positive - benign, flagged", True)],
        [(1, 0, "False negative - refusal missed", True),
         (1, 1, "True positive - correctly refused", False)],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for row, ax_row in zip(cells, axes):
        for (true, pred, title, is_error), ax in zip(row, ax_row):
            sel = df[(df["label"] == true) & (df["pred"] == pred)]
            # Prefer short stems so they render in a couple of lines.
            short = sel[sel["text"].str.len() < 170]
            pool = short if len(short) >= n_per_cell else sel
            picks = pool.sample(n=min(n_per_cell, len(pool)), random_state=seed)

            ax.set_facecolor("#fdeceb" if is_error else "#eef5ee")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{title}  (n={len(sel)})", fontsize=10,
                         fontweight="bold", loc="left")
            y = 0.93
            for _, r in picks.iterrows():
                wrapped = textwrap.fill(r["text"], width=64)
                ax.text(0.03, y, f"[score {r['refuse_score']:.2f}]",
                        fontsize=7.5, color="#555", transform=ax.transAxes)
                ax.text(0.03, y - 0.085, wrapped, fontsize=8.2, va="top",
                        transform=ax.transAxes)
                y -= 0.10 + 0.085 * (wrapped.count("\n") + 1)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "quadrant_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def quadrant_score_hist():
    """2x2 histograms of DistilBERT's refuse-score, one per confusion-matrix
    quadrant. Shows where each quadrant sits on the score axis - in particular
    whether errors are uncertain (~0.5) or confident (near 0 or 1)."""
    df = _load("distilbert")
    p = df["refuse_score"].to_numpy()
    bins = np.linspace(0, 1, 26)

    cells = [
        [(0, 0, "True negative - benign, passed", False),
         (0, 1, "False positive - benign, flagged", True)],
        [(1, 0, "False negative - refusal missed", True),
         (1, 1, "True positive - correctly refused", False)],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    for row, ax_row in zip(cells, axes):
        for (true, pred, title, is_error), ax in zip(row, ax_row):
            m = (df["label"] == true) & (df["pred"] == pred)
            ax.hist(p[m], bins=bins, color="#b2342c" if is_error else "#2c6b3f")
            ax.set_facecolor("#fdeceb" if is_error else "#eef5ee")
            ax.set_title(f"{title}  (n={int(m.sum())})", fontsize=9.5,
                         fontweight="bold", loc="left")
            ax.set_ylabel("count")
    for ax in axes[1]:
        ax.set_xlabel("refuse-score  P(refuse)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "quadrant_score_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def confidence_quadrants():
    """2x2 panel: refuse-score distribution per confusion-matrix quadrant
    (DistilBERT). Histogram carries the large correct quadrants; a rug of
    individual points carries the sparse error quadrants (n=6, n=7)."""
    df = _load("distilbert")
    bins = np.linspace(0, 1, 26)

    # (true, pred): (title, is_error). Layout mirrors Figures 1 and 3.
    cells = [
        [(0, 0, "True negative - benign, passed", False),
         (0, 1, "False positive - benign, flagged", True)],
        [(1, 0, "False negative - refusal missed", True),
         (1, 1, "True positive - correctly refused", False)],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for row, ax_row in zip(cells, axes):
        for (true, pred, title, is_error), ax in zip(row, ax_row):
            s = df.loc[(df["label"] == true) & (df["pred"] == pred),
                       "refuse_score"].to_numpy()
            color = "tab:red" if is_error else "tab:green"
            ax.hist(s, bins=bins, color=color, alpha=0.55)
            # Rug: every individual prediction as a tick just below the axis
            # (x in data coords, y in axes-fraction coords).
            ax.plot(s, np.full_like(s, -0.04), "|", color=color,
                    markersize=9, markeredgewidth=1.2, clip_on=False,
                    transform=ax.get_xaxis_transform())
            ax.axvline(0.5, color="0.4", ls="--", lw=1)
            ax.set_title(f"{title}  (n={len(s)})", fontsize=10,
                         fontweight="bold", loc="left")
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("P(refuse)")
    for ax in axes[:, 0]:
        ax.set_ylabel("count")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confidence_quadrants.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def ood_strip():
    """Strip plot of refuse-scores on out-of-domain (non-bio) MMLU subjects.
    One row per subject, every question a dot; red = false positive. Reads
    outputs/ood_predictions.csv (written by src.ood_probe)."""
    path_in = os.path.join(OUT_DIR, "ood_predictions.csv")
    if not os.path.exists(path_in):
        print("skipping ood_strip: run `python -m src.ood_probe` first")
        return
    df = pd.read_csv(path_in)

    # Order subjects by false-positive rate (worst at top).
    fpr = df.assign(fp=df["refuse_score"] >= 0.5).groupby("subject")["fp"].mean()
    subjects = fpr.sort_values().index.tolist()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, subj in enumerate(subjects):
        s = df.loc[df["subject"] == subj, "refuse_score"].to_numpy()
        y = i + rng.uniform(-0.28, 0.28, size=len(s))
        err = s >= 0.5
        ax.scatter(s[~err], y[~err], s=9, color="tab:blue", alpha=0.35,
                   edgecolors="none")
        ax.scatter(s[err], y[err], s=14, color="tab:red", alpha=0.8,
                   edgecolors="none")
        ax.text(1.02, i, f"FPR {err.mean():.0%}", va="center", fontsize=8)

    ax.axvline(0.5, color="0.4", ls="--", lw=1)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(subjects) - 0.4)
    ax.set_xlabel("P(refuse)")
    ax.set_title("Out-of-domain probe: refuse-score per non-bio subject")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ood_strip.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    confusion_matrices()
    pr_curves()
    length_histogram()
    quadrant_examples()
    confidence_quadrants()
    ood_strip()
    quadrant_score_hist()
