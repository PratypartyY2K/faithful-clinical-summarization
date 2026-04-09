"""Shared JSONL helpers and dataset transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_first_jsonl_row(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON rows found in {path}")


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


def process_dataset_split(input_dir: Path, output_dir: Path, split: str) -> None:
    examples = read_jsonl(input_dir / f"{split}.jsonl")
    write_jsonl(
        output_dir / "summarization" / f"{split}.jsonl",
        build_summarization_rows(examples),
    )
    write_jsonl(
        output_dir / "verifier" / f"{split}.jsonl",
        build_verifier_rows(examples),
    )
