#!/usr/bin/env python3
"""Train a summarizer with either a seq2seq baseline or a PEFT/QLoRA causal LM path."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.evaluation.pipeline_metrics import compute_text_overlap_metrics
from src.modeling.pipeline import build_summary_prompt, generate_summaries_batch
from src.utils.metadata import build_run_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mimiciii/processed/summarization"))
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--trainer-type", choices=("seq2seq", "causal"), default="seq2seq")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parse_args_with_optional_config(parser)
    if args.use_qlora and not args.use_peft:
        raise ValueError("--use-qlora requires --use-peft.")
    if args.use_qlora and not torch.cuda.is_available():
        raise ValueError("QLoRA training requires CUDA in this implementation.")
    return args


class CausalDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_data(data_dir: Path):
    return load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
        },
    )


def build_quantization_config(use_qlora: bool):
    if not use_qlora:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def configure_peft(model, args: argparse.Namespace, task_type: TaskType):
    if not args.use_peft:
        return model
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=task_type,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, peft_config)
    return model


def build_seq2seq_trainer(dataset, tokenizer, args: argparse.Namespace):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    model = configure_peft(model, args, TaskType.SEQ_2_SEQ_LM)

    def preprocess(batch):
        prompts = [build_summary_prompt(str(input_text)) for input_text in batch["input_text"]]
        inputs = tokenizer(
            prompts,
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
    training_kwargs = {
        "output_dir": str(args.output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
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
    }
    processing_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processing_arg] = tokenizer
    return Seq2SeqTrainer(**trainer_kwargs)


def build_causal_trainer(dataset, tokenizer, args: argparse.Namespace):
    quantization_config = build_quantization_config(args.use_qlora)
    model_kwargs = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = configure_peft(model, args, TaskType.CAUSAL_LM)

    def preprocess(batch):
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for input_text, target_text in zip(batch["input_text"], batch["target_text"]):
            prompt = build_summary_prompt(str(input_text))
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_source_length,
            )["input_ids"]
            target_ids = tokenizer(
                str(target_text) + (tokenizer.eos_token or ""),
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_target_length,
            )["input_ids"]
            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            rows["input_ids"].append(input_ids)
            rows["attention_mask"].append([1] * len(input_ids))
            rows["labels"].append(labels)
        return rows

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    training_kwargs = {
        "output_dir": str(args.output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": 5,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
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
        "data_collator": CausalDataCollator(tokenizer),
    }
    processing_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processing_arg] = tokenizer
    return Trainer(**trainer_kwargs)


def evaluate_generation(dataset, tokenizer, model, args: argparse.Namespace) -> Dict[str, object]:
    validation_inputs = [str(row["input_text"]) for row in dataset["validation"]]
    validation_references = [str(row["target_text"]) for row in dataset["validation"]]
    predictions = generate_summaries_batch(
        dialogues=validation_inputs,
        summarizer_tokenizer=tokenizer,
        summarizer_model=model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
    )
    return compute_text_overlap_metrics(predictions=predictions, references=validation_references)


def main() -> None:
    args = parse_args()
    dataset = load_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if args.trainer_type == "seq2seq":
        trainer = build_seq2seq_trainer(dataset, tokenizer, args)
    else:
        trainer = build_causal_trainer(dataset, tokenizer, args)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    generation_metrics = evaluate_generation(dataset, tokenizer, trainer.model, args)
    metrics_path = args.output_dir / "generation_metrics.json"
    write_json(metrics_path, generation_metrics)
    write_json(
        args.output_dir / "run_metadata.json",
        build_run_metadata(
            stage="summarizer_training",
            args=args,
            extra={
                "metrics_file": metrics_path,
                "train_examples": len(dataset["train"]),
                "validation_examples": len(dataset["validation"]),
            },
        ),
    )
    print(json.dumps(generation_metrics, indent=2))
    print(f"Saved summarizer artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
