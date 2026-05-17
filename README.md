# Biology Refusal Classifier

A binary text classifier that flags natural-language prompts as **refuse**
(biosecurity risk) or **don't refuse** (benign). Built for the SecureBio /
CBAI take-home task.

## Approach in one paragraph

Positives are WMDP-bio question stems (`cais/wmdp`), answer choices stripped.
Negatives are benign biology and medical exam questions from MMLU -- a
deliberately *hard* contrast: same exam-question format, same broad domain, so
the classifier must separate hazardous biology from ordinary biology rather
than learning a "is this biology / is this a question" shortcut. Two models are
compared: a frozen-embedding + logistic-regression baseline, and a fine-tuned
DistilBERT. See `WRITEUP.md` for full reasoning and results.

## Setup

```bash
conda create -n biorefuse python=3.11 -y
conda activate biorefuse
# torch must match your CUDA driver; the cu121 wheel works for driver >= 12.1.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets sentence-transformers scikit-learn pandas accelerate
```

Exact pinned versions used for the reported results are in `requirements.txt`.

WMDP is a gated dataset on Hugging Face. Accept the terms at
https://huggingface.co/datasets/cais/wmdp then authenticate:

```bash
huggingface-cli login
```

## Run

```bash
python -m src.data        # build data/train.csv and data/test.csv
python -m src.baseline    # embeddings + logistic regression
python -m src.finetune    # fine-tune DistilBERT (saves the model)
python -m src.calibration # confidence/calibration analysis (after the models)
python -m src.ood_probe   # out-of-domain probe on non-bio MMLU (after finetune)
python -m src.figures     # write-up figures (run after the steps above)
```

Metrics print to stdout. Each model writes full test predictions to
`outputs/<model>_predictions.csv` and misclassified rows to
`outputs/<model>_misclassified.csv` for failure analysis.

To render the write-up to PDF (no LaTeX needed):

```bash
conda install -c conda-forge pandoc weasyprint -y
pandoc WRITEUP.md -o WRITEUP.pdf --pdf-engine=weasyprint -c style.css
```

## Layout

| File | Purpose |
|------|---------|
| `src/data.py`     | Build and balance the dataset, stratified 80/20 split |
| `src/baseline.py` | all-MiniLM-L6-v2 embeddings + logistic regression |
| `src/finetune.py` | DistilBERT fine-tuning |
| `src/evaluate.py` | Shared metrics + prediction/failure dump |
| `src/calibration.py` | Confidence-of-errors, ECE, risk-coverage analysis |
| `src/ood_probe.py` | Out-of-domain false-positive rate on non-bio MMLU |
| `src/figures.py`  | All write-up figures |

## Hardware

Developed on a single GTX 1080 (8 GB). DistilBERT fine-tuning takes a few
minutes; the baseline runs in well under a minute. Runs on free-tier Colab
unchanged.
