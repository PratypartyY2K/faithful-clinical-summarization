#!/usr/bin/env python3
"""Compare human claim-faithfulness annotations against overlap metrics and verifier outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.cli import parse_args_with_optional_config
from src.evaluation.human_eval import evaluate_human_annotations, read_annotation_csv
from src.utils.metadata import build_run_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/evaluation/human_annotations/human_annotation_report.json"),
    )
    args = parse_args_with_optional_config(parser)

    annotation_rows = read_annotation_csv(args.annotation_csv)
    report = {
        "run_metadata": build_run_metadata(
            stage="human_annotation_evaluation",
            args=args,
            extra={"num_rows": len(annotation_rows)},
        ),
        "metrics": evaluate_human_annotations(annotation_rows),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_json(args.output_file.parent / "human_annotation_eval_metadata.json", report["run_metadata"])
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
