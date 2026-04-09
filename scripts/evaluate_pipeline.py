#!/usr/bin/env python3
"""Evaluate the synthetic faithful summarization pipeline on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.cli import parse_args_with_optional_config
from src.evaluation.pipeline_metrics import run_full_evaluation, write_evaluation_report
from src.modeling.pipeline import load_summarizer, load_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-test-file", type=Path, default=Path("data/dummy/raw/test.jsonl"))
    parser.add_argument(
        "--verifier-test-file",
        type=Path,
        default=Path("data/dummy/processed/verifier/test.jsonl"),
    )
    parser.add_argument("--summarizer-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--verifier-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--verifier-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/evaluation_report.json"),
    )
    args = parse_args_with_optional_config(parser)

    summarizer_tokenizer, summarizer_model = load_summarizer(args.summarizer_dir)
    verifier_tokenizer, verifier_model = load_verifier(args.verifier_dir)
    report = run_full_evaluation(
        raw_examples_path=args.raw_test_file,
        verifier_examples_path=args.verifier_test_file,
        summarizer_tokenizer=summarizer_tokenizer,
        summarizer_model=summarizer_model,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
        max_new_tokens=args.max_new_tokens,
        verifier_batch_size=args.verifier_batch_size,
    )
    write_evaluation_report(report, args.output_file)
    print(json.dumps(report["generation_metrics"], indent=2))
    print(json.dumps(report["verifier_metrics"], indent=2))
    print(json.dumps(report["qualitative_error_analysis"], indent=2))


if __name__ == "__main__":
    main()
