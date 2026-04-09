#!/usr/bin/env python3
"""Train a claim verifier against dialogue-claim pairs."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_LABEL_MAP = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
}


def infer_label_mapping(dataset) -> tuple[dict[int, str], dict[str, int]]:
    train_rows = dataset["train"]
    label_name_column = train_rows.column_names and "label_name" in train_rows.column_names
    if label_name_column:
        label_pairs = sorted(
            {(int(label), str(label_name)) for label, label_name in zip(train_rows["label"], train_rows["label_name"])},
            key=lambda item: item[0],
        )
        id2label = {label_id: label_name for label_id, label_name in label_pairs}
        label2id = {label_name: label_id for label_id, label_name in id2label.items()}
        return id2label, label2id
    unique_labels = sorted(set(int(label) for label in train_rows["label"]))
    if unique_labels == [0, 1, 2]:
        id2label = DEFAULT_LABEL_MAP.copy()
    else:
        id2label = {0: "unsupported", 1: "supported"}
    label2id = {label_name: label_id for label_id, label_name in id2label.items()}
    return id2label, label2id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dummy/processed/verifier"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "validation.jsonl"),
        },
    )
    id2label, label2id = infer_label_mapping(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

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
        average = "binary" if len(id2label) == 2 else "macro"
        return {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
            "precision": round(float(precision_score(labels, predictions, average=average, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, predictions, average=average, zero_division=0)), 4),
            "f1": round(float(f1_score(labels, predictions, average=average, zero_division=0)), 4),
        }

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": 5,
        "report_to": "none",
    }
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_arg] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    processing_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processing_arg] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "verifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved verifier artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
