"""Dump every test prompt grouped by confusion-matrix quadrant, sorted by
the model's refuse-score, for manual browsing. Not a deliverable - an
exploration aid. Writes outputs/examples_by_quadrant.txt.

Run:  python -m src.dump_examples
"""

import os

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

QUADRANTS = [
    (1, 0, "FALSE NEGATIVE - refusal missed"),
    (0, 1, "FALSE POSITIVE - benign flagged"),
    (1, 1, "TRUE POSITIVE - correctly refused"),
    (0, 0, "TRUE NEGATIVE - benign passed"),
]


def main(model: str = "distilbert") -> None:
    df = pd.read_csv(os.path.join(OUT_DIR, f"{model}_predictions.csv"))
    lines = [f"# {model}: test prompts by confusion-matrix quadrant",
             "# sorted by refuse-score; [score] (source) text\n"]
    for true, pred, title in QUADRANTS:
        sub = df[(df["label"] == true) & (df["pred"] == pred)]
        sub = sub.sort_values("refuse_score", ascending=False)
        lines.append(f"\n{'=' * 78}\n{title}  (n={len(sub)})\n{'=' * 78}")
        for _, r in sub.iterrows():
            lines.append(f"[{r['refuse_score']:.3f}] ({r['source']}) {r['text']}")

    path = os.path.join(OUT_DIR, "examples_by_quadrant.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(path)}  ({len(df)} prompts)")


if __name__ == "__main__":
    main()
