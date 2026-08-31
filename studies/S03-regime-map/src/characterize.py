"""Layer-A characterisation: real llama.cpp serving on the Jetson AGX Orin.

Measures, for each model variant and replica count r (PROTOCOL Sec. 2, R8):
  - service-time distribution uncontended (c=1) and with all replicas busy (c=r),
  - the realised capacity curve r -> throughput (never assumed linear — the S01
    plant assumed linear scaling from single-replica data and was never validated;
    that assumption is measured here on the actual shared-GPU deployment),
  - per-request serving energy, idle-subtracted (RQ3 component characterisation),
  - tokens in/out per request for J/token normalisation.

Deployment model: one "replica" = one llama-server process with a single slot; all
replicas share the one Orin GPU, so inter-replica contention is real and measured.
This is the single-node edge deployment the study manages.

Outputs: results/layer_a/<variant>_r<r>_c<c>.json (raw per-request records + power
trace) and results/layer_a/summary.json. Run: `python3 characterize.py [--smoke]`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from artifact_io import write_json
from power import PowerSampler, measure_idle, power_mode, read_rails_mw

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "results" / "layer_a"
MODELS = Path(os.environ.get("S03_MODELS_DIR", Path.home() / "models")).expanduser()
SERVER_BIN = Path(os.environ.get(
    "S03_LLAMA_SERVER_BIN",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
)).expanduser()
BASE_PORT = 8100

VARIANTS = {
    "full-3B":  {"gguf": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
    "lite-1B":  {"gguf": "Llama-3.2-1B-Instruct-Q8_0.gguf"},
    "tiny-1B4": {"gguf": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
}
R_LEVELS = [1, 2, 4, 6]
N_REQUESTS = 60
N_WARMUP = 5
MAX_TOKENS = 48
CTX = 2048

# Fixed request bank: a short-text service (classification/summary class of edge
# inference). 12 inputs cycled; per-request seed varies sampling, temperature 0.7.
REQUEST_BANK = [
    "The delivery arrived two days late and the packaging was damaged, but support resolved it quickly.",
    "Battery life on this device is exceptional; easily two full days of heavy use.",
    "The app crashes every time I open the settings page after the last update.",
    "Setup took under five minutes and the instructions were clear throughout.",
    "Audio quality is muddy at high volume and the bass distorts noticeably.",
    "Customer service kept me on hold for forty minutes and never solved the issue.",
    "The camera performs well in daylight but struggles badly in low light.",
    "Firmware update fixed the connectivity drops I was seeing on the older version.",
    "The subscription price doubled this year without any new features being added.",
    "Build quality feels premium and the hinge mechanism is smooth and solid.",
    "Shipping was fast but the item did not match the photos in the listing.",
    "After three months of daily use the strap broke at the buckle joint.",
]
PROMPT_TEMPLATE = (
    "You are an edge review-triage service. Classify the following customer review's "
    "sentiment (positive/negative/mixed) and name the product aspect discussed, in one "
    "short sentence.\nReview: {text}"
)


def launch_replicas(gguf: str, r: int, slots_mode: bool = False) -> list[subprocess.Popen]:
    """Two single-node deployment shapes:
    - process mode (default): r independent llama-server processes, 1 slot each
    - slots mode: ONE server with --parallel r (continuous batching across slots);
      context scaled so each slot keeps CTX tokens
    """
    procs = []
    n_proc = 1 if slots_mode else r
    for i in range(n_proc):
        port = BASE_PORT + i
        p = subprocess.Popen(
            [str(SERVER_BIN), "-m", str(MODELS / gguf), "-ngl", "99",
             "--port", str(port), "--host", "127.0.0.1",
             "-c", str(CTX * (r if slots_mode else 1)),
             "--parallel", str(r if slots_mode else 1)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    deadline = time.monotonic() + 180
    for i in range(n_proc):
        port = BASE_PORT + i
        while True:
            if time.monotonic() > deadline:
                kill_replicas(procs)
                raise RuntimeError(f"replica {i} (port {port}) failed health check")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                    if json.load(resp).get("status") == "ok":
                        break
            except Exception:
                time.sleep(1.0)
    return procs


def kill_replicas(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def one_request(port: int, idx: int, seed: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content":
                      PROMPT_TEMPLATE.format(text=REQUEST_BANK[idx % len(REQUEST_BANK)])}],
        "max_tokens": MAX_TOKENS, "temperature": 0.7, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    latency = time.monotonic() - t0
    usage = payload.get("usage", {})
    return {"latency_s": latency, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "seed": seed,
            "finish": payload["choices"][0].get("finish_reason")}


def run_cell(variant: str, r: int, c: int, n_requests: int, slots_mode: bool = False) -> dict:
    """Closed-loop: c workers issue requests round-robin over r replicas until
    n_requests completed (after warmup). Power sampled throughout."""
    records: list[dict] = []
    lock = threading.Lock()
    counter = {"i": 0}

    def worker(wid: int) -> None:
        while True:
            with lock:
                i = counter["i"]
                if i >= n_requests + N_WARMUP:
                    return
                counter["i"] = i + 1
            port = BASE_PORT if slots_mode else BASE_PORT + (i % r)
            try:
                rec = one_request(port, i, seed=10_000 + i)
                rec["warmup"] = i < N_WARMUP
                rec["worker"] = wid
                with lock:
                    records.append(rec)
            except Exception as e:
                with lock:
                    records.append({"error": str(e), "warmup": i < N_WARMUP})

    idle_before = measure_idle(8.0)
    with PowerSampler() as ps:
        t_start = time.monotonic()
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t_start
    idle_after = measure_idle(8.0)

    good = [x for x in records if "error" not in x and not x["warmup"]]
    lats = sorted(x["latency_s"] for x in good)
    errors = [x for x in records if "error" in x]

    def pct(p: float) -> float:
        return lats[min(len(lats) - 1, int(p * len(lats)))] if lats else float("nan")

    idle_gpu = (idle_before.mean_mw("VDD_GPU_SOC") + idle_after.mean_mw("VDD_GPU_SOC")) / 2
    idle_cpu = (idle_before.mean_mw("VDD_CPU_CV") + idle_after.mean_mw("VDD_CPU_CV")) / 2
    e_gpu = ps.trace.energy_j("VDD_GPU_SOC") - idle_gpu / 1000.0 * ps.trace.duration_s()
    e_cpu = ps.trace.energy_j("VDD_CPU_CV") - idle_cpu / 1000.0 * ps.trace.duration_s()
    tok_out = sum(x["completion_tokens"] or 0 for x in good)

    cell = {
        "variant": variant, "mode": "slots" if slots_mode else "process", "r": r, "c": c, "n_good": len(good), "n_errors": len(errors),
        "wall_s": wall, "throughput_rps": len(good) / wall if wall > 0 else 0,
        "latency_s": {"mean": statistics.fmean(lats) if lats else None,
                      "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
                      "min": lats[0] if lats else None, "max": lats[-1] if lats else None},
        "tokens": {"prompt_mean": statistics.fmean(x["prompt_tokens"] for x in good) if good else None,
                   "completion_mean": statistics.fmean(x["completion_tokens"] for x in good) if good else None},
        "energy": {"gpu_soc_j_active": e_gpu, "cpu_cv_j_active": e_cpu,
                   "gpu_soc_j_per_request": e_gpu / len(good) if good else None,
                   "gpu_soc_j_per_out_token": e_gpu / tok_out if tok_out else None,
                   "idle_gpu_soc_mw": idle_gpu, "idle_cpu_cv_mw": idle_cpu,
                   "window_s": ps.trace.duration_s(),
                   "resolution_note": "20 mA LSB ~= 0.4 W per sample on 20 V rails (D-003)"},
        "raw_requests": records,
        "power_trace": ps.trace.to_dict(),
        "idle_before": idle_before.to_dict(), "idle_after": idle_after.to_dict(),
    }
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", action="store_true",
                    help="one server with --parallel r (continuous batching) instead of r processes")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny cell (tiny-1B4, r=1, c=1, 8 requests) to validate the harness")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"power_mode": power_mode(), "idle_snapshot_mw": read_rails_mw(),
            "server_bin": SERVER_BIN.name,
            "n_requests": N_REQUESTS, "n_warmup": N_WARMUP, "max_tokens": MAX_TOKENS,
            "ctx": CTX, "r_levels": R_LEVELS, "temperature": 0.7, "mode": None}
    assert "MODE_30W" in meta["power_mode"], f"power mode changed: {meta['power_mode']} (pin per D-003)"

    plan = ([("tiny-1B4", 1, 1, 8)] if args.smoke else
            [(v, r, c, N_REQUESTS) for v in VARIANTS for r in R_LEVELS for c in sorted({1, r})])

    summary = {"meta": meta, "cells": []}
    current_servers: tuple[str, int] | None = None
    procs: list[subprocess.Popen] = []
    try:
        for variant, r, c, n in plan:
            if current_servers != (variant, r):
                if procs:
                    kill_replicas(procs)
                    time.sleep(5)  # settle before idle windows
                print(f"[{time.strftime('%H:%M:%S')}] launching {variant} x{r} ({'slots' if args.slots else 'process'})...", flush=True)
                procs = launch_replicas(VARIANTS[variant]["gguf"], r, slots_mode=args.slots)
                current_servers = (variant, r)
                time.sleep(3)
            print(f"[{time.strftime('%H:%M:%S')}] cell {variant} r={r} c={c} n={n}", flush=True)
            cell = run_cell(variant, r, c, n, slots_mode=args.slots)
            mode_tag = "slots_" if args.slots else ""
            path = OUT / f"{mode_tag}{variant}_r{r}_c{c}.json"
            write_json(path, cell)
            brief = {k: cell[k] for k in ("variant", "r", "c", "n_good", "n_errors",
                                          "throughput_rps", "latency_s")}
            brief["gpu_soc_j_per_request"] = cell["energy"]["gpu_soc_j_per_request"]
            summary["cells"].append(brief)
            print(f"  p50={cell['latency_s']['p50']:.3f}s p95={cell['latency_s']['p95']:.3f}s "
                  f"thpt={cell['throughput_rps']:.2f} rps "
                  f"E/req={cell['energy']['gpu_soc_j_per_request']:.2f} J", flush=True)
    finally:
        if procs:
            kill_replicas(procs)

    name = "summary_smoke.json" if args.smoke else ("summary_slots.json" if args.slots else "summary.json")
    write_json(OUT / name, summary)
    print("done:", OUT)


if __name__ == "__main__":
    sys.exit(main())
