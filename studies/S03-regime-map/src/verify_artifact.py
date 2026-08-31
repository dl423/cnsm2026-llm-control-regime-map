#!/usr/bin/env python3
"""Offline integrity and completeness checks for the public artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
CHECKSUMS = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {".git", ".venv", "reproduced", "runs", "__pycache__", ".DS_Store", "PRIVACY.md"}

CONTENT_PATTERNS = {
    "absolute macOS home": re.compile(r"/" + r"Users/[^/\s]+"),
    "absolute Linux home": re.compile(r"/" + r"home/[^/\s]+"),
    "exact UTC timestamp": re.compile(
        r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b"
    ),
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
    ),
    "private IPv4 address": re.compile(
        r"(?<!\d)(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
        r"192\.168\.(?:\d{1,3}\.)\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})(?!\d)"
    ),
    "MAC address": re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "common API token": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|"
        r"github_pat_[A-Za-z0-9_]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|"
        r"AIza[0-9A-Za-z_-]{20,}|hf_[A-Za-z0-9]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    "API resource identifier": re.compile(
        r"\b(?:chatcmpl-|resp_|req_|msg_|org-|proj_)[A-Za-z0-9_-]{10,}"
    ),
    "JWT-like token": re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
}

ANALYSIS_FILES = {
    "blocks_EH.json", "cells.json", "failure_modes.json",
    "failure_modes_attempts.json", "margins.json",
    "oracle_static_I1000.json", "overload_fraction.json",
    "realised_I.json", "rq2_summary.json",
}
REGENERATED_ANALYSIS_FILES = ANALYSIS_FILES - {"oracle_static_I1000.json"}


def fail(message: str) -> None:
    raise AssertionError(message)


def strict_loads(text: str, source: Path | str):
    def invalid_constant(value: str):
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        return json.loads(text, parse_constant=invalid_constant)
    except Exception as error:
        raise AssertionError(f"invalid strict JSON in {source}: {error}") from error


def included_release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != CHECKSUMS
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not path.name.endswith((".pyc", ".pyo"))
    )


def verify_checksums() -> int:
    if not CHECKSUMS.is_file():
        fail("CHECKSUMS.sha256 is missing")
    expected: dict[str, str] = {}
    for number, line in enumerate(CHECKSUMS.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise AssertionError(f"malformed checksum line {number}") from error
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"malformed SHA-256 on line {number}")
        if relative in expected:
            fail(f"duplicate checksum path: {relative}")
        expected[relative] = digest

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if included_release_file(path)
    }
    if set(expected) != actual_paths:
        missing = sorted(actual_paths - set(expected))
        stale = sorted(set(expected) - actual_paths)
        fail(f"checksum inventory mismatch; unlisted={missing}, missing={stale}")

    for relative, expected_digest in expected.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if digest != expected_digest:
            fail(f"checksum mismatch: {relative}")
    return len(expected)


def verify_content() -> int:
    files = [path for path in ROOT.rglob("*") if included_release_file(path)]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            fail(f"symbolic links are not permitted in the release: {relative}")
        if path.suffix in {".log", ".tex", ".bib", ".pdf"}:
            fail(f"unexpected file type in release: {relative}")

        if path.name.endswith(".json.gz"):
            content = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        else:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
        for label, pattern in CONTENT_PATTERNS.items():
            match = pattern.search(content)
            if match:
                fail(f"{label} in {relative}: {match.group(0)!r}")
    return len(files)


def verify_environment_manifest() -> None:
    models_path = STUDY / "env" / "models.json"
    models = strict_loads(models_path.read_text(encoding="utf-8"), models_path)
    local_models = models.get("local_models", [])
    if len(local_models) != 3:
        fail(f"expected three local model records, got {len(local_models)}")
    required = {
        "alias", "filename", "source_repository", "source_url", "revision",
        "bytes", "sha256", "quantization", "upstream_license",
    }
    for model in local_models:
        if set(model) != required:
            fail(f"model provenance fields differ for {model.get('alias', '<unknown>')}")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            fail(f"invalid model revision for {model['alias']}")
        if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
            fail(f"invalid model SHA-256 for {model['alias']}")
        if not isinstance(model["bytes"], int) or model["bytes"] <= 0:
            fail(f"invalid model size for {model['alias']}")
    if len(models.get("hosted_models", [])) != 2:
        fail("expected two hosted model provenance records")

    system_path = STUDY / "env" / "system.json"
    system = strict_loads(system_path.read_text(encoding="utf-8"), system_path)
    if system.get("python", {}).get("version") != "3.12.13":
        fail("reference Python version is not pinned to 3.12.13")
    if system.get("cuda", {}).get("version") != "12.6":
        fail("reference CUDA version is not pinned to 12.6")
    if system.get("hardware", {}).get("power_mode_id") != 2:
        fail("reference Jetson power mode is not pinned to mode 2")
    flags = system.get("llama_cpp", {}).get("cmake_flags", {})
    if flags != {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "87"}:
        fail(f"unexpected llama.cpp build flags: {flags}")


def verify_json_and_gzip() -> tuple[int, int, int]:
    json_count = jsonl_rows = gzip_count = 0
    for path in ROOT.rglob("*.json"):
        strict_loads(path.read_text(encoding="utf-8"), path)
        json_count += 1
    for path in ROOT.rglob("*.jsonl"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            strict_loads(line, f"{path}:{number}")
            jsonl_rows += 1
    for path in ROOT.rglob("*.json.gz"):
        raw = path.read_bytes()
        if raw[:2] != b"\x1f\x8b":
            fail(f"bad gzip magic: {path}")
        if int.from_bytes(raw[4:8], "little") != 0:
            fail(f"non-deterministic gzip mtime: {path}")
        if raw[3] & 0x1C:
            fail(f"gzip includes extra/name/comment metadata: {path}")
        strict_loads(gzip.decompress(raw).decode("utf-8"), path)
        gzip_count += 1
    return json_count, jsonl_rows, gzip_count


def read_jsonl(path: Path) -> list[dict]:
    return [strict_loads(line, f"{path}:{number}")
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)]


def verify_result_inventory() -> dict[str, int]:
    sweep = read_jsonl(RESULTS / "sweep.jsonl")
    if len(sweep) != 1690:
        fail(f"sweep row count: expected 1690, got {len(sweep)}")
    if len({row["key"] for row in sweep}) != len(sweep):
        fail("sweep keys are not unique")
    blocks = Counter(row["block"] for row in sweep)
    if blocks != Counter({"B": 810, "C": 420, "E": 340, "H": 120}):
        fail(f"unexpected sweep block counts: {dict(blocks)}")
    conservation_failures = 0
    for row in sweep:
        metrics = row["metrics"]
        if metrics["offered"] != (
            metrics["completed"] + metrics["dropped"] + metrics["residual_queue"]
        ):
            conservation_failures += 1
    if conservation_failures:
        fail(f"request-conservation failures: {conservation_failures}")

    decision_files = list((RESULTS / "decisions").glob("*.json.gz"))
    if len(decision_files) != 700:
        fail(f"decision archive count: expected 700, got {len(decision_files)}")

    rq2 = read_jsonl(RESULTS / "rq2.jsonl")
    rq2_keys = {(row["config"], row["state_id"], row["repeat"]) for row in rq2}
    if len(rq2) != 4200 or len(rq2_keys) != 4200:
        fail(f"RQ2 expected 4200 unique rows, got {len(rq2)}/{len(rq2_keys)}")
    rq2_counts = Counter(row["config"] for row in rq2)
    expected_rq2 = Counter({
        "luna-seeded": 900,
        "luna-default": 900,
        "local-serial": 900,
        "local-batched": 900,
        "sonnet-default": 600,
    })
    if rq2_counts != expected_rq2:
        fail(f"unexpected RQ2 config counts: {dict(rq2_counts)}")

    layer_dir = RESULTS / "layer_a"
    raw_cells = [path for path in layer_dir.glob("*.json")
                 if path.name not in {"summary.json", "summary_slots.json"}]
    process_cells = [path for path in raw_cells if not path.name.startswith("slots_")]
    slots_cells = [path for path in raw_cells if path.name.startswith("slots_")]
    if (len(raw_cells), len(process_cells), len(slots_cells)) != (42, 21, 21):
        fail("Layer-A expected 42 raw cells (21 process, 21 slots), got "
             f"{len(raw_cells)} ({len(process_cells)}, {len(slots_cells)})")

    tuning = list((RESULTS / "tuning").glob("*.json"))
    if len(tuning) != 12:
        fail(f"tuning artifact count: expected 12, got {len(tuning)}")

    oracle = read_jsonl(RESULTS / "oracle_static.jsonl")
    oracle_cells = Counter(row["cell"] for row in oracle)
    if len(oracle) != 40 or oracle_cells != Counter({"I1000|c15": 20, "I1000|c60": 20}):
        fail(f"unexpected oracle-static inventory: {dict(oracle_cells)}")

    validation = {path.name for path in (RESULTS / "validation_F").glob("*.json")}
    expected_validation = {"heuristic_s13.json", "llm_s13.json", "llm_I1000_s8.json"}
    if validation != expected_validation:
        fail(f"unexpected validation inventory: {sorted(validation)}")

    analysis = {path.name for path in (RESULTS / "analysis").glob("*.json")}
    if analysis != ANALYSIS_FILES:
        fail(f"unexpected analysis inventory: {sorted(analysis)}")

    return {
        "sweep": len(sweep),
        "decisions": len(decision_files),
        "rq2": len(rq2),
        "layer_a": len(raw_cells),
        "tuning": len(tuning),
        "oracle": len(oracle),
        "validation": len(validation),
        "conservation_failures": conservation_failures,
    }


def compare_values(expected, actual, location: str = "root") -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            fail(f"analysis mismatch at {location}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            fail(f"analysis mismatch at {location}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            fail(f"analysis key mismatch at {location}")
        for key in expected:
            compare_values(expected[key], actual[key], f"{location}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            fail(f"analysis length mismatch at {location}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{location}[{index}]")
        return
    if expected != actual:
        fail(f"analysis mismatch at {location}: {expected!r} != {actual!r}")


def compare_analysis(directory: Path) -> None:
    for name in sorted(REGENERATED_ANALYSIS_FILES):
        expected_path = RESULTS / "analysis" / name
        actual_path = directory / name
        if not actual_path.is_file():
            fail(f"regenerated analysis missing {name}")
        expected = strict_loads(expected_path.read_text(encoding="utf-8"), expected_path)
        actual = strict_loads(actual_path.read_text(encoding="utf-8"), actual_path)
        compare_values(expected, actual, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-analysis", type=Path)
    args = parser.parse_args()
    checksum_count = verify_checksums()
    file_count = verify_content()
    verify_environment_manifest()
    json_count, jsonl_rows, gzip_count = verify_json_and_gzip()
    inventory = verify_result_inventory()
    if args.compare_analysis:
        compare_analysis(args.compare_analysis.resolve())

    print(f"checksums: PASS ({checksum_count} files)")
    print(f"content scan: PASS ({file_count} files)")
    print(f"strict JSON: PASS ({json_count} JSON files, {jsonl_rows} JSONL rows)")
    print(f"decision archives: {gzip_count}/700")
    print(f"sweep: {inventory['sweep']} rows; blocks B=810 C=420 E=340 H=120")
    print(f"request conservation failures: {inventory['conservation_failures']}")
    print(f"RQ2: {inventory['rq2']} unique rows")
    print("analysis semantic comparison: " +
          ("PASS" if args.compare_analysis else "not requested"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        sys.exit(1)
