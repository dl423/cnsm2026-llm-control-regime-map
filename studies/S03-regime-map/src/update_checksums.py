#!/usr/bin/env python3
"""Regenerate the release checksum inventory from the artifact root."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {".git", ".venv", "reproduced", "runs", "__pycache__", ".DS_Store", "PRIVACY.md"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not path.name.endswith((".pyc", ".pyo"))
    )


def main() -> None:
    lines = []
    for path in sorted((path for path in ROOT.rglob("*") if included(path)),
                       key=lambda item: item.relative_to(ROOT).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {OUTPUT.name}")


if __name__ == "__main__":
    main()
