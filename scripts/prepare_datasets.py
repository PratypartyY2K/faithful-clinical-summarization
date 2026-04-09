#!/usr/bin/env python3
"""Prepare summarization and verifier datasets from synthetic raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_summarization_rows(examples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example in examples:
        rows.append(
            {
                "example_id": example["example_id"],
                "input_text": example["dialogue"],
                "target_text": example["summary"],
            }
        )
    return rows


def build_verifier_rows(examples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example in examples:
        for claim in example["claims"]:
            rows.append(
                {
                    "example_id": example["example_id"],
                    "dialogue": example["dialogue"],
                    "claim": claim["claim"],
                    "label": claim["label"],
                }
            )
    return rows


def process_split(input_dir: Path, output_dir: Path, split: str) -> None:
    examples = read_jsonl(input_dir / f"{split}.jsonl")
    write_jsonl(
        output_dir / "summarization" / f"{split}.jsonl",
        build_summarization_rows(examples),
    )
    write_jsonl(
        output_dir / "verifier" / f"{split}.jsonl",
        build_verifier_rows(examples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/dummy/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dummy/processed"))
    args = parser.parse_args()

    for split in ("train", "validation", "test"):
        process_split(args.input_dir, args.output_dir, split)

    print(f"Prepared datasets under {args.output_dir}")


if __name__ == "__main__":
    main()
