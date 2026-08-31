# LLM control-regime map: reproducibility artifact

This repository contains the code, fixed inputs, structured evidence, and
derived result tables for experiments on LLM-driven control of an edge
inference service. The hardware experiments used an NVIDIA Jetson AGX Orin.

The release is scoped to **experimental reproducibility**. It includes the
experiment, analysis, and verification code; the fixed design inputs; the
pinned environment and model provenance; the canonical structured evidence for
every experiment block; and the derived analysis tables the results are read
from.

## Quick verification

With Python 3.12 and the dependencies installed:

```bash
./run_all.sh
```

The default command is offline and makes no network requests. It verifies the
checksum inventory, strict JSON, gzip integrity, expected experiment counts,
and request conservation, and then regenerates and semantically compares the
derived analysis tables. See [REPRODUCING.md](REPRODUCING.md) for environment
setup and hardware/API rerun instructions.

## Repository map

- `studies/S03-regime-map/data/`: fixed design inputs and a structured
  hardware capability record.
- `studies/S03-regime-map/env/`: pinned Python environment and model/runtime
  provenance.
- `studies/S03-regime-map/src/`: experiment, analysis, and verification code.
- `studies/S03-regime-map/results/`: immutable canonical structured evidence.
- `studies/S03-regime-map/PROTOCOL.md`: the realized experimental protocol.
- `METHODS.md`: experimental design and statistical conventions.
- `RESULTS.md`: result inventory and interpretation guide.
- `CHECKSUMS.sha256`: SHA-256 inventory of every release file other than the
  checksum file itself.

Canonical `results/` are evidence, not a workspace. Regenerated output goes to
the gitignored `reproduced/` directory. Run live experiments only in a fresh,
disposable clone because the original experiment scripts write result files.

## Limits

- Hosted-model reruns require the researcher's own credentials and can incur
  charges; the offline default never invokes an API.
- Hosted model identifiers were not dated snapshots. The recorded identifiers
  and absence of provider fingerprints are preserved as an external-validity
  limitation.
- The recorded `llama.cpp` revision is a short commit prefix; the full commit
  and the binary hash were not captured. Build flags and runtime versions are
  recorded rather than reconstructed after the fact.

Code is MIT-licensed and the repository's original experimental data and
documentation are licensed under CC BY 4.0. See [LICENSE](LICENSE).
