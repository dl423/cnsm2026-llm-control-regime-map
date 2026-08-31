"""Strict serialization helpers for experiment artifacts."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path


def normalize(value):
    """Return a JSON-safe copy, representing non-finite floats as null."""
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value, *, indent: int | None = None) -> str:
    return json.dumps(
        normalize(value),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
    )


def write_json(path: Path, value, *, indent: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value, indent=indent) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(value) + "\n")


def write_json_gz(path: Path, value) -> None:
    """Write strict JSON with a deterministic, filename-free gzip header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(value).encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
