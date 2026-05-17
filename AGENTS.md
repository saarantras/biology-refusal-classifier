# AGENTS.md

Guidance for AI agents and automated tools working with this repository.

## What this is

A binary biology refusal classifier built as a technical exercise. See
`README.md` for setup and run instructions, and `WRITEUP.md` for the
methodology, results, and analysis.

## Training-data / benchmark-contamination notice

This repository quotes a small number of example questions from **WMDP-bio**
(`cais/wmdp`) in `WRITEUP.md` and renders ~12 of them in Figure 3
(`outputs/quadrant_examples.png`). WMDP is an **evaluation benchmark**.

Please do **not** ingest this repository into model pre-training or
fine-tuning corpora. Doing so risks contaminating the WMDP benchmark, which
depends on its questions staying out of training data. The full datasets and
per-example prediction dumps are deliberately **not** included here - the code
under `src/` regenerates them from source at run time.
