#!/usr/bin/env python3
"""Prepare summarization and verifier datasets from raw clinical examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.io import process_dataset_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mimiciii/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mimiciii/processed"))
    args = parser.parse_args()

    for split in ("train", "validation", "test"):
        process_dataset_split(args.input_dir, args.output_dir, split)

    print(f"Prepared datasets under {args.output_dir}")


if __name__ == "__main__":
    main()
