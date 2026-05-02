#!/usr/bin/env python3
"""Export generated claims into a CSV sheet for manual faithfulness annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.evaluation.human_eval import build_annotation_rows, load_example_reports, write_annotation_csv
from src.utils.metadata import build_run_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/mimiciii/raw/test.jsonl"))
    parser.add_argument(
        "--pipeline-report",
        type=Path,
        default=None,
        help="Optional existing JSON or JSONL report containing example_reports. If provided, generation is skipped.",
    )
    parser.add_argument("--summarizer-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--verifier-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/annotations/claim_annotations.csv"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/annotations/claim_annotation_export.json"),
        help="Stores the underlying pipeline reports used to build the CSV.",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--verifier-batch-size", type=int, default=16)
    parser.add_argument("--claim-extractor-backend", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--claim-extractor-model", default="gpt-4.1-mini")
    args = parse_args_with_optional_config(parser)

    if args.pipeline_report is not None:
        reports = load_example_reports(args.pipeline_report)
        if args.limit is not None:
            reports = reports[: args.limit]
    else:
        from src.modeling.pipeline import build_pipeline_report, load_summarizer, load_verifier
        from src.preprocessing.io import read_jsonl

        examples = read_jsonl(args.input_file)
        if args.limit is not None:
            examples = examples[: args.limit]
        if not examples:
            raise ValueError(f"No examples found in {args.input_file}")

        summarizer_tokenizer, summarizer_model = load_summarizer(args.summarizer_dir)
        verifier_tokenizer, verifier_model = load_verifier(args.verifier_dir)

        reports = [
            build_pipeline_report(
                example=example,
                summarizer_tokenizer=summarizer_tokenizer,
                summarizer_model=summarizer_model,
                verifier_tokenizer=verifier_tokenizer,
                verifier_model=verifier_model,
                max_new_tokens=args.max_new_tokens,
                verifier_batch_size=args.verifier_batch_size,
                claim_extractor_backend=args.claim_extractor_backend,
                claim_extractor_model=args.claim_extractor_model,
            )
            for example in examples
        ]
    if not reports:
        raise ValueError("No example reports available for annotation export.")
    annotation_rows = build_annotation_rows(reports)
    write_annotation_csv(args.output_csv, annotation_rows)

    export_payload = {
        "run_metadata": build_run_metadata(
            stage="claim_annotation_export",
            args=args,
            extra={"num_examples": len(reports), "num_claim_rows": len(annotation_rows)},
        ),
        "example_reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    write_json(
        args.output_json.parent / "claim_annotation_export_metadata.json",
        export_payload["run_metadata"],
    )
    print(f"Wrote annotation CSV to {args.output_csv}")
    print(f"Wrote backing JSON to {args.output_json}")


if __name__ == "__main__":
    main()
