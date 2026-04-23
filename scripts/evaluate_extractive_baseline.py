#!/usr/bin/env python3
"""Evaluate simple extractive baselines such as Lead-k sentences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.evaluation.pipeline_metrics import compute_text_overlap_metrics
from src.preprocessing.io import read_jsonl, take_first_sentences
from src.utils.metadata import build_run_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/mimiciii/raw/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/extractive_baseline"))
    parser.add_argument("--strategy", choices=("lead"), default="lead")
    parser.add_argument("--num-sentences", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parse_args_with_optional_config(parser)


def get_source_text(example: dict[str, object]) -> str:
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


def build_prediction(example: dict[str, object], strategy: str, num_sentences: int) -> str:
    source_text = get_source_text(example)
    if strategy == "lead":
        return take_first_sentences(source_text, max_sentences=num_sentences)
    raise ValueError(f"Unsupported strategy: {strategy}")


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.input_file)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found in {args.input_file}")

    predictions = [
        build_prediction(example, strategy=args.strategy, num_sentences=args.num_sentences)
        for example in examples
    ]
    references = [get_reference_summary(example) for example in examples]
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
            stage="extractive_baseline_evaluation",
            args=args,
            extra={
                "num_examples": len(examples),
                "predictions_file": predictions_path,
                "baseline": {
                    "strategy": args.strategy,
                    "num_sentences": args.num_sentences,
                },
            },
        ),
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions to {predictions_path}")


if __name__ == "__main__":
    main()
