#!/usr/bin/env python3
"""Train a small seq2seq model on the prepared summarization dataset."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dummy/processed/summarization"))
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "validation.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    rouge = evaluate.load("rouge")

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(predictions=decoded_predictions, references=decoded_labels)
        return {key: round(value, 4) for key, value in scores.items()}

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "predict_with_generate": True,
        "logging_steps": 5,
        "report_to": "none",
        "load_best_model_at_end": False,
    }
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_arg] = "epoch"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        "compute_metrics": compute_metrics,
    }
    processing_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processing_arg] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved summarizer artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
