# Reproducing the experiments

## 1. Offline integrity and analysis

The reference software environment is Python 3.12.13, managed with uv 0.12.1.
Create an isolated environment from the pinned package set:

```bash
uv venv --python 3.12.13 .venv
. .venv/bin/activate
uv pip sync studies/S03-regime-map/env/requirements.txt
./run_all.sh
```

`./run_all.sh` performs, in order:

1. a checksum, strict-JSON, gzip, and result-inventory audit;
2. offline regeneration of eight derived analysis tables into
   `reproduced/analysis/`; and
3. a recursive semantic comparison against the canonical tables with absolute
   and relative tolerance `1e-12` for floating-point values.

Useful individual commands are:

```bash
./run_all.sh verify
./run_all.sh analysis
./run_all.sh compare
./run_all.sh selfcheck
```

The canonical oracle-static table is verified as evidence but is not regenerated
by the normal analysis command because producing it reruns its calibration and
evaluation episodes.

## 2. Local model files

Download the three GGUF files listed in `studies/S03-regime-map/env/models.json`
and verify their byte sizes and SHA-256 digests. Point the scripts at them and at
the `llama-server` executable:

```bash
cp .env.example .env
# Edit only your local .env; it is ignored by Git.
```

The scripts read `S03_MODELS_DIR` and `S03_LLAMA_SERVER_BIN`. The model files are
third-party assets and are not redistributed by this repository.

## 3. Jetson prerequisites

The reference hardware/software configuration is in
`studies/S03-regime-map/env/system.json` and the structured capability
probe is in `data/jetson-probe/hardware-probe.json`. The important constraints
are Jetson AGX Orin, L4T R36.4.4, CUDA 12.6, `MODE_30W` mode 2, and a CUDA-enabled
`llama.cpp` build for compute capability 8.7.

Before energy or wall-clock measurements, ensure the GPU is otherwise idle and
no `llama-server` process is running. The hardware scripts deliberately fail
their precondition checks when those assumptions do not hold.

## 4. Live rerun order

Live commands write experiment output. Run them only in a fresh disposable clone
and archive that clone after the campaign. Never place API keys on a command line
or commit `.env`.

From `studies/S03-regime-map/`, the intended order is:

```bash
python src/workload.py
python src/plant.py
python src/characterize.py
python src/characterize.py --slots
python src/accuracy_probe.py
python src/fairness.py
python src/fairness.py --secondary
python src/sweep.py B
```

Start the local controller server on port 8200 with one slot before the serial
local sweep and RQ2 stages. The model and binary paths below are environment
variables rather than machine-specific paths:

```bash
"$S03_LLAMA_SERVER_BIN" \
  -m "$S03_MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -ngl 99 --host 127.0.0.1 --port 8200 -c 4096 --parallel 1
```

With the server healthy and hosted API credentials loaded from `.env`:

```bash
python src/sweep.py C
python src/sweep.py E
python src/sweep.py H
python src/rq2.py sample
python src/rq2.py probe-controls
python src/rq2.py hosted
python src/rq2.py local-serial
```

Stop only the server process you started, then restart it with `-c 8192
--parallel 4` and run:

```bash
python src/rq2.py local-batched
```

After stopping that server, run the hardware-owned stages, each of which manages
its own server process:

```bash
python src/rq3.py
python src/validate_replay.py
python src/oracle_static_check.py
```

Finally regenerate analysis to a separate directory:

```bash
python src/analyze.py tables \
  --input-results results \
  --output-dir ../../reproduced/analysis
```

Hosted calls incur charges and provider behavior can change. The original
provider model IDs, responses, rejection messages, and usage records are kept in
the canonical evidence; a future rerun is a replication under then-current
provider behavior, not a bitwise recreation of the hosted service.

## 5. Updating integrity metadata

If a legitimate release file changes, regenerate and then verify checksums:

```bash
python studies/S03-regime-map/src/update_checksums.py
./run_all.sh verify
```
