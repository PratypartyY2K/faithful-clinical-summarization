#!/usr/bin/env python3
"""Run a lightweight generate-and-verify pipeline on one dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.cli import parse_args_with_optional_config
from src.modeling.pipeline import build_pipeline_report, load_summarizer, load_verifier
from src.preprocessing.io import read_first_jsonl_row
from src.utils.metadata import build_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/dummy/raw/test.jsonl"))
    parser.add_argument("--summarizer-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--verifier-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--verifier-batch-size", type=int, default=16)
    parser.add_argument("--output-file", type=Path, default=Path("artifacts/pipeline_report.json"))
    args = parse_args_with_optional_config(parser)

    example = read_first_jsonl_row(args.input_file)
    summarizer_tokenizer, summarizer_model = load_summarizer(args.summarizer_dir)
    verifier_tokenizer, verifier_model = load_verifier(args.verifier_dir)
    report = build_pipeline_report(
        example=example,
        summarizer_tokenizer=summarizer_tokenizer,
        summarizer_model=summarizer_model,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
        max_new_tokens=args.max_new_tokens,
        verifier_batch_size=args.verifier_batch_size,
    )
    report["run_metadata"] = build_run_metadata(
        stage="single_pipeline_run",
        args=args,
        extra={"input_example_id": example["example_id"]},
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
