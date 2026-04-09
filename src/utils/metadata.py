"""Helpers for writing reproducible run metadata."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def namespace_to_dict(args: Namespace) -> Dict[str, Any]:
    return {key: serialize_value(value) for key, value in vars(args).items()}


def build_run_metadata(stage: str, args: Namespace, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "created_at_utc": utc_timestamp(),
        "args": namespace_to_dict(args),
    }
    if extra:
        payload["extra"] = serialize_value(extra)
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
