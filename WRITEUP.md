# Biology Refusal Classifier - Write-up

## Data

**Positives ("refuse").** 1,273 WMDP-bio questions from `cais/wmdp`

**Negatives ("don't refuse").** I used
benign **MMLU biology and medical subsets** (`high_school_biology`,
`college_biology`, `anatomy`, `virology`, `medical_genetics`, plus five clinical
subjects). The idea is to make the model separate *hazardous* biology
from *ordinary* biology. 2,094 raw negatives were down-sampled to 1,273 (equal
fraction per subject, preserving coverage); stratified 80/20 split.

## Models and results

**Baseline: frozen embeddings + logistic regression.** Each prompt is encoded
once by `all-MiniLM-L6-v2` (a 6-layer, 22.7M-parameter sentence-transformer)
into a 384-d vector. The encoder is totally frozen, and an L2-penalised logistic regression
is fit on top: 385 trained parameters (384 weights + bias). This isolates how much signal is already linearly separable
in general-purpose embedding space.

**DistilBERT: full fine-tuning.** `distilbert-base-uncased` with a new
2-class head (a 768x768 `pre_classifier` plus a 768x2 `classifier`,
~0.6M new parameters on top of the first-token representation). *All* ~67M
parameters are updated - the 6 transformer layers, the embeddings, and the head;
nothing is frozen. With only ~2k training examples this is cheap (trained in ~75 s on my old desktop computer, which just has a
GTX 1080).
Training: 3 epochs, AdamW at lr 2e-5 with linear decay, batch size 16, weight
decay 0.01, max sequence length 256, fp32.
`enable_full_determinism` fixes the run.

::: {.metricbox}
| Model | Acc. | AUROC | AUPRC |
|---|---|---|---|
| Emb. + LR  | 0.952 | 0.986 | 0.985 |
| DistilBERT | 0.974 | 0.993 | 0.995 |
:::

Headline metrics are shown right; AUPRC is for the refuse (positive) class.
Per-class precision / recall / F1 at the 0.5 threshold - DistilBERT: refuse
0.976 / 0.973 / 0.975, don't-refuse 0.972 / 0.976 / 0.974; baseline: refuse
0.943 / 0.965 / 0.954, don't-refuse 0.963 / 0.940 / 0.951 (raw counts in the
Figure 1 confusion matrices). Fine-tuning buys ~2.2 accuracy points. The
threshold-free metrics matter more for a safety filter: DistilBERT's AUPRC of
0.995 means it holds near-perfect precision across almost the whole recall
range, so the high-recall operating point a filter wants is nearly free. Even
the frozen baseline reaches 95% accuracy against hard negatives - separable,
but not trivially so.

## Failure modes (DistilBERT, 13 errors)

Figure 3 shows sampled prompts from each confusion-matrix quadrant.

**False negatives - missed refusals (7).** WMDP stems that read as ordinary
biology out of context: *"Which of the following are considered gain of function
research?"*, *"How long can the paroxysmal coughing stage last?"* - arguably
label noise as much as model error.

**False positives - benign flagged (6).** Benign MMLU questions tripping on
hazard-adjacent vocabulary: *"Who discovered the first virus?"*, *"Nipah is a
zoonotic paramyxovirus; where did it originate?"* - the model over-weights
pathogen and recombinant-DNA terms, the exact tension the hard-negative design
was built to expose. These false positives are not a free trade for safety: a
filter that over-refuses benign biology makes the assistant unusable for
legitimate research and pushes users toward unfiltered alternatives - an
instance of AviD's Rule of Usability, "security at the expense of usability
comes at the expense of security."[^avid]

[^avid]: Avi Douglen, "AviD's Rule of Usability." Coined on the Information
Security Stack Exchange; see <https://xo.tc/avids-rule-of-usability.html>.

**Errors are often confident (Figure 4).** DistilBERT is well-calibrated in
aggregate (ECE 1.6%, Brier 0.021), but its mistakes split into two populations:
~half (6/13) sit near the 0.5 boundary, while the rest are confident - 5/13 at
>=0.99, mean error-confidence 0.87. A low-confidence "escalate to review" band
therefore catches 6/13 errors for only ~2% review load, then hits a wall: the
confident errors need better data, not a threshold.

One systematic confound (Figure 2): WMDP stems are uniformly short, while MMLU's
clinical-vignette questions give the benign class a long right tail, so prompt
length weakly correlates with the label and could be learned as a shortcut.

## Out-of-domain behaviour

The classifier only ever sees biology in training, but a deployed filter mostly
sees non-bio traffic. I scored it on 12 benign non-bio MMLU subjects (2,132
prompts, inference-only - not in training or the headline metrics; Figure 5).
Aggregate false-positive rate is a reassuring 2.3%, and pure humanities/STEM
(history, philosophy, maths, physics, CS) are ~0%. But it is very uneven:
`us_foreign_policy` is flagged at 27% and `college_chemistry` at 9%.
Foreign-policy questions about nuclear weapons, proliferation and rogue states
genuinely overlap WMDP's threat register (WMDP = *Weapons of Mass Destruction*
Proxy), and reactive-chemistry questions share a hazard vocabulary. The model
learned a fuzzier "biosecurity-risk-adjacent" concept than "hazardous biology" -
a scoping error only visible off-distribution.

## What I'd do next

- **Better positives.** I don't think the WMDP-bio question set represents a very realistic set of attacks. Many read pretty benign to me. 
- **Threshold + calibration.** Pick a high-recall operating point from the PR
  curve rather than the 0.5 default; calibrate the (often over-confident) scores.
  Could add a low-confidence "escalate to review"/"escalate to a heavier model".
- **Stricter evaluation.** Length-match the classes so prompt length can't be
  used as a shortcut; test on a held-out *different* benign corpus. I'm worried
  the model could just learn "MMLU vs WMDP" style.
- **Long inputs.** Inputs are unrealistically short. Maybe windowing + max-pool aggregation, but I suspect this would lose a lot of nuance. Maybe a bigger input?

## AI assistance

Claude Code under supervision helped quite a bit.

## Figures

![Figure 1. Confusion matrices on the held-out test set (n=503).](outputs/confusion_matrices.png){width=78%}

![Figure 2. Prompt-length distribution by class. WMDP-bio (refuse) is uniformly short; MMLU (don't refuse) has a long right tail.](outputs/length_histogram.png){width=62%}

![Figure 3. Sampled prompts from each confusion-matrix quadrant (DistilBERT), with the model's refuse-score. Green = correct, red = error.](outputs/quadrant_examples.png){width=92%}

![Figure 4. Refuse-score distribution per confusion-matrix quadrant (DistilBERT). Histogram bars carry the large correct quadrants; the rug below each axis marks every individual prediction. Error quadrants (red) show many predictions far from the 0.5 boundary - confident mistakes.](outputs/confidence_quadrants.png){width=80%}

![Figure 5. Out-of-domain probe: refuse-score for every benign prompt across 12 non-bio MMLU subjects (inference-only). Red = false positive. Subjects ordered by false-positive rate.](outputs/ood_strip.png){width=82%}
