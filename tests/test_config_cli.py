from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.cli import parse_args_with_optional_config


class ConfigCliTest(unittest.TestCase):
    def test_config_defaults_are_loaded(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--value", type=int, default=1)
        parser.add_argument("--name", default="baseline")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({"value": 7, "name": "configured"}), encoding="utf-8")
            with patch("sys.argv", ["prog", "--config", str(config_path)]):
                args = parse_args_with_optional_config(parser)

        self.assertEqual(args.value, 7)
        self.assertEqual(args.name, "configured")


if __name__ == "__main__":
    unittest.main()
