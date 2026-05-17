"""Fine-tune DistilBERT end-to-end as the refusal classifier.

Run:  python -m src.finetune
"""

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, enable_full_determinism)

from src.evaluate import report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
SEED = 42


def main() -> None:
    # Make the fine-tune reproducible run-to-run (CUDA ops are otherwise
    # nondeterministic, causing ~0.2% accuracy jitter between runs).
    enable_full_determinism(SEED)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tok, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=os.path.join(DATA_DIR, "..", "outputs", "distilbert"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        seed=SEED,
        fp16=False,  # GTX 1080 (Pascal) has no tensor cores; fp32 is fine here.
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    pred = trainer.predict(test_ds)
    logits = torch.tensor(pred.predictions)
    probs = torch.softmax(logits, dim=-1).numpy()
    y_pred = probs.argmax(axis=-1)
    report("distilbert", test_df, y_pred, probs[:, 1])

    # Persist model + tokenizer so the out-of-domain probe can reuse it
    # without retraining (src/ood_probe.py).
    save_dir = os.path.join(DATA_DIR, "..", "outputs", "distilbert_model")
    trainer.save_model(save_dir)
    print(f"saved fine-tuned model -> {os.path.relpath(save_dir)}")


if __name__ == "__main__":
    main()
