#!/usr/bin/env python3
"""Train a claim verifier against dialogue-claim pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dummy/processed/verifier"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "validation.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def preprocess(batch):
        return tokenizer(
            batch["dialogue"],
            batch["claim"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(preprocess, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy_score = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels)
        return {
            "accuracy": round(accuracy_score["accuracy"], 4),
            "f1": round(f1_score["f1"], 4),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved verifier artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
