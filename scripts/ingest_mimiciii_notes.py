#!/usr/bin/env python3
"""Build raw train/validation/test JSONL files from MIMIC-III discharge summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.preprocessing.io import write_jsonl
from src.preprocessing.mimiciii_discharge import build_raw_example, stable_split_name
from src.utils.metadata import build_run_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/mimiciii/NOTEEVENTS.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mimiciii/raw"))
    parser.add_argument("--category", default="Discharge summary")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-source-chars", type=int, default=400)
    parser.add_argument("--min-target-chars", type=int, default=120)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parse_args_with_optional_config(parser)
    if args.train_fraction <= 0 or args.validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")
    return args


def main() -> None:
    args = parse_args()
    split_rows = {"train": [], "validation": [], "test": []}
    stats = Counter()

    with args.input_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["rows_seen"] += 1

            if row.get("CATEGORY") != args.category:
                stats["rows_skipped_wrong_category"] += 1
                continue
            if (row.get("ISERROR") or "").strip():
                stats["rows_skipped_iserror"] += 1
                continue

            example = build_raw_example(
                row,
                min_source_chars=args.min_source_chars,
                min_target_chars=args.min_target_chars,
            )
            if example is None:
                stats["rows_skipped_missing_sections"] += 1
                continue

            split = stable_split_name(
                str(example["metadata"].get("hadm_id") or example["example_id"]),
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )
            split_rows[split].append(example)
            stats["examples_written"] += 1
            stats[f"{split}_examples"] += 1

            if args.max_examples is not None and stats["examples_written"] >= args.max_examples:
                break

    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    metadata = build_run_metadata(
        stage="mimiciii_ingestion",
        args=args,
        extra={"stats": dict(stats)},
    )
    write_json(args.output_dir / "ingestion_metadata.json", metadata)

    print(json.dumps(dict(stats), indent=2))
    print(f"Wrote raw MIMIC-III splits to {args.output_dir}")


if __name__ == "__main__":
    main()
