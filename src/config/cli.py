"""CLI helpers for loading JSON config presets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args_with_optional_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file whose keys map to CLI argument names.",
    )
    preliminary_args, _ = parser.parse_known_args()
    if preliminary_args.config is not None:
        config_data = json.loads(preliminary_args.config.read_text(encoding="utf-8"))
        normalized = {key.replace("-", "_"): value for key, value in config_data.items()}
        parser.set_defaults(**normalized)
    return parser.parse_args()
