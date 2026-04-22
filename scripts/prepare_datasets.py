#!/usr/bin/env python3
"""Prepare summarization and verifier datasets from raw clinical examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.preprocessing.io import process_dataset_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mimiciii/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mimiciii/processed"))
    parser.add_argument(
        "--target-sentence-limit",
        type=int,
        default=None,
        help="Optionally keep only the first N sentences of the summarization target.",
    )
    parser.add_argument(
        "--narrative-only",
        action="store_true",
        help="Keep only cleaner narrative summarization targets using conservative heuristics.",
    )
    parser.add_argument("--min-target-words", type=int, default=40)
    parser.add_argument("--max-target-words", type=int, default=260)
    parser.add_argument("--min-target-sentences", type=int, default=2)
    parser.add_argument("--max-structured-markers", type=int, default=0)
    args = parse_args_with_optional_config(parser)

    for split in ("train", "validation", "test"):
        process_dataset_split(
            args.input_dir,
            args.output_dir,
            split,
            target_sentence_limit=args.target_sentence_limit,
            narrative_only=args.narrative_only,
            min_target_words=args.min_target_words,
            max_target_words=args.max_target_words,
            min_target_sentences=args.min_target_sentences,
            max_structured_markers=args.max_structured_markers,
        )

    print(f"Prepared datasets under {args.output_dir}")


if __name__ == "__main__":
    main()
