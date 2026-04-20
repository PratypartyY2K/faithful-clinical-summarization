#!/usr/bin/env python3
"""Evaluate a trained summarizer without requiring a verifier model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.preprocessing.io import read_jsonl
from src.utils.metadata import build_run_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/mimiciii/raw/test.jsonl"))
    parser.add_argument("--summarizer-dir", type=Path, default=Path("artifacts/summarizer/flan_t5_small"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/summarizer"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parse_args_with_optional_config(parser)


def get_dialogue_text(example: dict[str, object]) -> str:
    if "dialogue" in example:
        return str(example["dialogue"])
    if "input_text" in example:
        return str(example["input_text"])
    raise KeyError("Expected example to contain either 'dialogue' or 'input_text'.")


def get_reference_summary(example: dict[str, object]) -> str:
    if "target_text" in example:
        return str(example["target_text"])
    return str(example.get("summary") or "")


def get_reference_summary_full(example: dict[str, object]) -> str | None:
    if "target_text_full" in example:
        return str(example["target_text_full"])
    summary = example.get("summary")
    return str(summary) if summary is not None else None


def main() -> None:
    from src.evaluation.pipeline_metrics import compute_text_overlap_metrics
    from src.modeling.pipeline import generate_summaries_batch, load_summarizer

    args = parse_args()
    examples = read_jsonl(args.input_file)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found in {args.input_file}")

    tokenizer, model = load_summarizer(args.summarizer_dir)
    dialogues = [get_dialogue_text(example) for example in examples]
    references = [get_reference_summary(example) for example in examples]
    predictions = generate_summaries_batch(
        dialogues=dialogues,
        summarizer_tokenizer=tokenizer,
        summarizer_model=model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    metrics = compute_text_overlap_metrics(predictions=predictions, references=references)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for example, prediction in zip(examples, predictions):
            handle.write(
                json.dumps(
                    {
                        "example_id": example["example_id"],
                        "generated_summary": prediction,
                        "reference_summary": get_reference_summary(example),
                        "reference_summary_full": get_reference_summary_full(example),
                        "metadata": example.get("metadata", {}),
                    }
                )
                + "\n"
            )

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "run_metadata.json",
        build_run_metadata(
            stage="summarizer_evaluation",
            args=args,
            extra={
                "num_examples": len(examples),
                "predictions_file": predictions_path,
            },
        ),
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions to {predictions_path}")


if __name__ == "__main__":
    main()
