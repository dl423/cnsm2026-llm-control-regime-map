#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY_DIR="$ROOT_DIR/studies/S03-regime-map"
RESULTS_DIR="$STUDY_DIR/results"
OUTPUT_DIR="${S03_REPRODUCED_DIR:-$ROOT_DIR/reproduced/analysis}"
PYTHON_BIN="${PYTHON:-python3}"

verify() {
  "$PYTHON_BIN" "$STUDY_DIR/src/verify_artifact.py"
}

analysis() {
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" "$STUDY_DIR/src/analyze.py" tables \
    --input-results "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR"
}

compare() {
  "$PYTHON_BIN" "$STUDY_DIR/src/verify_artifact.py" \
    --compare-analysis "$OUTPUT_DIR"
}

selfcheck() {
  "$PYTHON_BIN" "$STUDY_DIR/src/analyze.py" selfcheck
  "$PYTHON_BIN" "$STUDY_DIR/src/rq2.py" selfcheck
  "$PYTHON_BIN" "$STUDY_DIR/src/rq3.py" selfcheck
  "$PYTHON_BIN" "$STUDY_DIR/src/validate_replay.py" selfcheck
}

case "${1:-all}" in
  verify) verify ;;
  analysis) analysis ;;
  compare) compare ;;
  selfcheck) selfcheck ;;
  all)
    verify
    analysis
    compare
    ;;
  *)
    echo "usage: $0 [all|verify|analysis|compare|selfcheck]" >&2
    exit 2
    ;;
esac
