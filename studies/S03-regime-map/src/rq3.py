"""RQ3 block G: deliberation-energy characterisation (PROTOCOL Secs. 1/6, D-003).

Measures what one LLM *deliberation* (a real controller decision call) costs in
joules on the edge device, to compose against the per-request *serving* energy
already measured in Layer A (results/layer_a/slots_*.json). Component
characterisation only: deliberation and serving are measured in separate time
windows, idle-subtracted; in-loop attribution is inadmissible on this device
(fused rails, D-003) and is not attempted.

Models measured: the selected local controller (full-3B — the block-C local arm)
and the smaller counterfactual controller (tiny-1B4). Prompts are the REAL
block-C decision prompts from results/rq2_states.json (30 replayed states),
issued with the sweep arm's exact call settings (seed 42, max_tokens 400, vendor
default sampling — NOT rq2's greedy settings, because the energy number must
describe the arm that ran). REPEATS repeats per state.

Preconditions (hard-checked): no llama-server running, GPU_SOC near the probed
idle floor, MODE_30W. Nothing else may run on the machine during measurement
(PROTOCOL Sec. 8c; energy windows need a quiet host).

Method: server launched fresh per model (--parallel 1, ctx 4096 = the :8200
controller config); 3 warmup calls; idle window 15 s; one continuous 10 Hz
power window over the whole call batch with per-call [start, end] timestamps;
idle window 15 s after. Per-call energy = trapezoidal integral over the call
span minus idle_mean * duration; batch-level mean = (window energy - idle_mean
* window) / n_calls (robust to inter-call gaps). Every number carries the 20 mA
LSB (~0.4 W) resolution bound and the power mode.

Output: results/rq3_deliberation.json (raw traces included) and a composed
deliberation-vs-serving table printed and stored. Usage:
  python3 rq3.py [--smoke]     # smoke: 1 model, 5 states, 1 repeat
  python3 rq3.py selfcheck     # offline math checks, no hardware assumptions
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from artifact_io import write_json
from power import PowerSampler, PowerTrace, measure_idle, power_mode, read_rails_mw

STUDY = Path(__file__).resolve().parents[1]
STATES = STUDY / "results" / "rq2_states.json"
OUT = STUDY / "results" / "rq3_deliberation.json"
MODELS_DIR = Path(os.environ.get("S03_MODELS_DIR", Path.home() / "models")).expanduser()
SERVER_BIN = Path(os.environ.get(
    "S03_LLAMA_SERVER_BIN",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
)).expanduser()
PORT = 8300                      # never the sweep's :8200 — this run owns the GPU

CONTROLLER_MODELS = {
    "full-3B":  "Llama-3.2-3B-Instruct-Q4_K_M.gguf",   # the block-C local arm
    "tiny-1B4": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",   # smaller counterfactual
}
REPEATS = 3
MAX_TOKENS = 400                 # = sweep local decision budget (llm_controllers)
SEED = 42                        # = sweep LLMController seed
GAP_S = 0.5                      # settle between calls inside the batch window
IDLE_S = 15.0
RAIL_PRIMARY = "VDD_GPU_SOC"
RAILS_REPORTED = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def span_energy_j(trace: PowerTrace, rail: str, t0: float, t1: float) -> float:
    """Trapezoidal integral of one rail restricted to [t0, t1] (trace-relative s)."""
    ts, ps = trace.t, trace.mw[rail]
    j = 0.0
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        if b <= t0 or a >= t1:
            continue
        lo, hi = max(a, t0), min(b, t1)
        # linear interp of power at lo/hi inside the sample interval
        if b == a:
            continue
        pa = ps[i - 1] + (ps[i] - ps[i - 1]) * (lo - a) / (b - a)
        pb = ps[i - 1] + (ps[i] - ps[i - 1]) * (hi - a) / (b - a)
        j += (pa + pb) / 2.0 * (hi - lo) / 1000.0
    return j


def launch_server(gguf: str) -> subprocess.Popen:
    p = subprocess.Popen(
        [str(SERVER_BIN), "-m", str(MODELS_DIR / gguf), "-ngl", "99",
         "--port", str(PORT), "--host", "127.0.0.1", "-c", "4096", "--parallel", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 180
    while True:
        if time.monotonic() > deadline:
            p.terminate()
            raise RuntimeError("server failed health check")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return p
        except Exception:
            time.sleep(1.0)


def decision_call(system: str, user: str) -> dict:
    body = json.dumps({"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": MAX_TOKENS, "seed": SEED}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    latency = time.monotonic() - t0
    u = payload.get("usage", {})
    return {"latency_s": latency,
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "text_head": payload["choices"][0]["message"]["content"][:120]}


def preconditions() -> dict:
    mode = power_mode()
    assert "MODE_30W" in mode, f"power mode changed: {mode} (pin per D-003)"
    servers = subprocess.run(["pgrep", "-af", "llama-server"],
                             capture_output=True, text=True).stdout.strip()
    assert not servers, f"llama-server already running — GPU not exclusively ours:\n{servers}"
    snap = read_rails_mw()
    assert snap["VDD_GPU_SOC"] < 3500, \
        f"GPU_SOC {snap['VDD_GPU_SOC']:.0f} mW: not at idle floor — something is using the GPU"
    return {"power_mode": mode, "idle_snapshot_mw": snap}


def measure_model(name: str, gguf: str, states: list[dict], system: str,
                  repeats: int) -> dict:
    print(f"[{time.strftime('%H:%M:%S')}] {name}: launching server", flush=True)
    proc = launch_server(gguf)
    try:
        for s in states[:3]:                     # warmup (not measured)
            decision_call(system, s["user"])
        time.sleep(2.0)
        idle_before = measure_idle(IDLE_S)
        calls = []
        with PowerSampler() as ps:
            t_ref = time.monotonic()
            for rep in range(repeats):
                for s in states:
                    time.sleep(GAP_S)
                    t0 = time.monotonic() - t_ref
                    rec = decision_call(system, s["user"])
                    t1 = time.monotonic() - t_ref
                    rec.update({"state_id": s["state_id"], "repeat": rep,
                                "t0": t0, "t1": t1})
                    calls.append(rec)
        idle_after = measure_idle(IDLE_S)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

    idle_mw = {r: (idle_before.mean_mw(r) + idle_after.mean_mw(r)) / 2
               for r in RAILS_REPORTED}
    # offset between the sampler's clock origin and t_ref is < thread-start time;
    # both were started back to back — bounded by one sample period, noted in output.
    for c in calls:
        for r in RAILS_REPORTED:
            gross = span_energy_j(ps.trace, r, c["t0"], c["t1"])
            c[f"j_{r}"] = gross - idle_mw[r] / 1000.0 * (c["t1"] - c["t0"])
    window = ps.trace.duration_s()
    batch = {}
    for r in RAILS_REPORTED:
        active = ps.trace.energy_j(r) - idle_mw[r] / 1000.0 * window
        batch[r] = {"window_s": window, "active_j_total": active,
                    "j_per_deliberation_batch": active / len(calls)}
    per_call = [c[f"j_{RAIL_PRIMARY}"] for c in calls]
    lat = [c["latency_s"] for c in calls]
    summary = {
        "model": name, "gguf": gguf, "n_calls": len(calls), "repeats": repeats,
        "j_per_deliberation": {
            "rail": RAIL_PRIMARY,
            "batch_estimate": batch[RAIL_PRIMARY]["j_per_deliberation_batch"],
            "per_call_mean": statistics.fmean(per_call),
            "per_call_sd": statistics.stdev(per_call) if len(per_call) > 1 else None,
            "per_call_p50": statistics.median(per_call),
            "per_call_min": min(per_call), "per_call_max": max(per_call),
        },
        "batch_by_rail": batch,
        "latency_s": {"mean": statistics.fmean(lat), "p50": statistics.median(lat),
                      "p95": sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))]},
        "tokens": {"prompt_mean": statistics.fmean(c["prompt_tokens"] or 0 for c in calls),
                   "completion_mean": statistics.fmean(c["completion_tokens"] or 0 for c in calls)},
        "idle_mw": idle_mw,
        "calls": calls,
        "idle_before": idle_before.to_dict(), "idle_after": idle_after.to_dict(),
        "power_trace": ps.trace.to_dict(),
        "clock_note": "call timestamps and sampler share time.monotonic; sampler "
                      "origin lags t_ref by < one 100 ms sample period",
        "resolution_note": "20 mA LSB ~= 0.4 W/sample on 20 V rails (D-003); over a "
                           "~2-4 s call that bounds ~0.8-1.6 J per-call quantisation",
    }
    print(f"  {name}: J/deliberation batch={summary['j_per_deliberation']['batch_estimate']:.1f} "
          f"per-call mean={summary['j_per_deliberation']['per_call_mean']:.1f} "
          f"(p50 {summary['j_per_deliberation']['per_call_p50']:.1f}) "
          f"latency p50={summary['latency_s']['p50']:.2f}s", flush=True)
    return summary


def compose_vs_serving(models: dict) -> list[dict]:
    """Deliberation J vs measured serving J/request (layer_a slots cells)."""
    rows = []
    for variant in ("tiny-1B4", "lite-1B", "full-3B"):
        for k in (1, 2, 4, 6):
            path = STUDY / "results" / "layer_a" / f"slots_{variant}_r{k}_c{k}.json"
            if not path.exists():
                continue
            cell = json.loads(path.read_text())
            serve_j = cell["energy"]["gpu_soc_j_per_request"]
            row = {"serving_variant": variant, "k": k, "serving_j_per_request": serve_j}
            for m, s in models.items():
                d = s["j_per_deliberation"]["batch_estimate"]
                row[f"deliberation_j_{m}"] = d
                row[f"ratio_{m}"] = d / serve_j if serve_j else None
            rows.append(row)
    return rows


def cmd_selfcheck() -> None:
    tr = PowerTrace(t=[0.0, 1.0, 2.0, 3.0, 4.0],
                    mw={r: [1000.0, 1000.0, 3000.0, 3000.0, 1000.0]
                        for r in ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0",
                                  "VDDQ_VDD2_1V8AO")})
    # full-window trapezoid: 1+2+3+2 = 8 J over 4 s
    assert abs(tr.energy_j("VDD_GPU_SOC") - 8.0) < 1e-9
    # sub-span [1,3]: 2 + 3 = 5 J
    assert abs(span_energy_j(tr, "VDD_GPU_SOC", 1.0, 3.0) - 5.0) < 1e-9
    # sub-span cutting samples: [1.5, 2.5] -> (2+3)/2*0.5 + (3+3)/2*0.5 = 2.75
    assert abs(span_energy_j(tr, "VDD_GPU_SOC", 1.5, 2.5) - 2.75) < 1e-9
    # spans outside the trace are zero
    assert span_energy_j(tr, "VDD_GPU_SOC", 10.0, 12.0) == 0.0
    # idle subtraction sanity: constant 1000 mW idle over [1,3] removes 2 J
    assert abs((span_energy_j(tr, "VDD_GPU_SOC", 1.0, 3.0) - 1.0 * 2.0) - 3.0) < 1e-9
    print("rq3 offline self-checks PASS (trapezoid, sub-span, idle subtraction)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "selfcheck"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.cmd == "selfcheck":
        cmd_selfcheck()
        return

    art = json.loads(STATES.read_text())
    system, states = art["system"], art["states"]
    pre = preconditions()
    if args.smoke:
        states, repeats, model_items = states[:5], 1, list(CONTROLLER_MODELS.items())[:1]
    else:
        repeats, model_items = REPEATS, list(CONTROLLER_MODELS.items())

    results = {"meta": {**pre, "n_states": len(states), "repeats": repeats,
                        "max_tokens": MAX_TOKENS, "seed": SEED,
                        "call_settings_note": "matches the block-C local arm "
                                              "(seed 42, max_tokens 400, default sampling)",
                        "states_source": str(STATES.name), "smoke": args.smoke},
               "models": {}, "composed": []}
    for name, gguf in model_items:
        results["models"][name] = measure_model(name, gguf, states, system, repeats)
    results["composed"] = compose_vs_serving(results["models"])
    out = OUT.with_name("rq3_deliberation_smoke.json") if args.smoke else OUT
    write_json(out, results)
    print(f"\ndeliberation vs serving (GPU_SOC, idle-subtracted):")
    for row in results["composed"]:
        extras = " ".join(f"{k.split('_', 1)[1]}={v:.1f}x" for k, v in row.items()
                          if k.startswith("ratio_") and v)
        print(f"  serve {row['serving_variant']}@k={row['k']}: "
              f"{row['serving_j_per_request']:.2f} J/req | {extras}")
    print("wrote", out.name)


if __name__ == "__main__":
    sys.exit(main())
