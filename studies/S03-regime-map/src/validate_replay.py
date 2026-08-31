"""Block F (R8, CORE): real-plant validation replay.

Replays >= 2 sweep cells IN REAL TIME against real llama.cpp serving on the
Jetson, and reports emulated-vs-real disagreement, whatever it is. R8 exists
because S01 promised this and silently never ran it.

Cells (PROTOCOL Sec. 9 / HANDOFF): (heuristic, I=400, c60) and
(llm_full local-3B, I=400, c60). Seed chosen MECHANICALLY per cell: the seed
whose emulated U_S03 is the cell median (ties -> lower seed) — no result-aware
selection. One 1800 s wall-clock episode per cell.

Real-plant realisation (D-018, fixed before any replay ran):
- Slots semantics as calibrated (D-010): ONE llama-server, --parallel r,
  ctx 2048*r — exactly the layer-A slots configuration whose distributions the
  emulated plant samples. Client-side FIFO queue gates in-flight to r.
- Requests are the layer-A triage bank (same prompts, temperature 0.7, seeded,
  max_tokens 48) so real service times are drawn from the same workload the
  calibration measured.
- ANY configuration change (variant or slot count) = drain in-flight, restart
  server, resume. The emulated plant charges load-time only for VARIANT
  switches and changes replicas for free; the real system cannot. This is a
  known, deliberate divergence F is designed to expose — the pause inventory is
  reported per replay, not hidden.
- The LLM cell's controller (local-3B) is served from a SECOND llama-server on
  :8300 sharing the GPU with the plant — real deliberation/serving contention,
  which the emulation does not model (PROTOCOL Sec. 11 threat 2). Its real
  latency delays action application, as in the plant.
- Scoring: the SAME code path as the sweep — real request records are fed to
  Plant._metrics (identical U_S03/U_v2/goodput arithmetic). Abandonment 30 s,
  warmup exclusion 300 s, identical.

Output: results/validation_F/<cell>.json (full request/decision records, pause
inventory, dispatch fidelity) + emulated-vs-real comparison table per cell.

Usage:
  python3 validate_replay.py            # both cells (~1 h + load times)
  python3 validate_replay.py --cell heuristic|llm --horizon 1800
  python3 validate_replay.py selfcheck  # offline: seed rule, obs windows, metrics glue
Preconditions: GPU exclusively ours (no llama-server up), MODE_30W, quiet host.
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

from artifact_io import dumps, write_json
from characterize import PROMPT_TEMPLATE, REQUEST_BANK
from controllers import HeuristicController
from fairness import build_cal, episode_cfg, load_design
from llm_controllers import LLMController, make_intent
from plant import Action, Observation, Plant, Request
from sweep import load_tuned, menu_text
from workload import make_trace

STUDY = Path(__file__).resolve().parents[1]
OUTDIR = STUDY / "results" / "validation_F"
SWEEP = STUDY / "results" / "sweep.jsonl"
MODELS_DIR = Path(os.environ.get("S03_MODELS_DIR", Path.home() / "models")).expanduser()
SERVER_BIN = Path(os.environ.get(
    "S03_LLAMA_SERVER_BIN",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
)).expanduser()
PLANT_PORT = 8100
CTL_PORT = 8300
CTX_PER_SLOT = 2048              # layer-A slots configuration
GGUF = {"full-3B": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "lite-1B": "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "tiny-1B4": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"}
TICK_S = 0.05

CELLS = {
    "heuristic": {"arm": "heuristic", "model": None, "I": 400.0, "cadence": 60},
    "llm":       {"arm": "llm_full", "model": "local-3B", "I": 400.0, "cadence": 60},
    # Added 2026-08-03 (D-025): mock-judge criticism — validation was absent at
    # the regime carrying the headlines. Same mechanical median-seed rule.
    "llm_I1000": {"arm": "llm_full", "model": "local-3B", "I": 1000.0, "cadence": 60},
}


def median_seed(arm: str, model, I: float, cadence: int) -> tuple[int, dict]:
    """The seed whose emulated U_S03 is the cell median (ties -> lower seed)."""
    rows = []
    for line in SWEEP.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("arm") == arm and r.get("I") == I and r.get("cadence_s") == cadence
                and (r.get("model") == model or (model is None and r.get("model") is None))):
            rows.append(r)
    if not rows:
        raise RuntimeError(f"no emulated rows for {arm}/{model}/I{I}/c{cadence}")
    rows.sort(key=lambda r: (r["metrics"]["U_S03"], -r["seed"]))
    mid = rows[(len(rows) - 1) // 2]
    return mid["seed"], mid


# ---------------------------------------------------------------- real plant

class RealPlant:
    """Wall-clock episode against a real llama-server (slots semantics)."""

    def __init__(self, design: dict, cfg, log: list):
        self.design = design
        self.cfg = cfg
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.variant: str | None = None
        self.replicas = 0
        self.lock = threading.Lock()
        self.in_flight = 0
        self.busy_integral = 0.0           # sum in_flight*dt (utilization)
        self.replica_seconds = 0.0         # scored-span integrals (plant parity)
        self.variant_acc_seconds = 0.0
        self._last_integrate = None
        self.restarting = False
        self.pauses: list[dict] = []

    # -- server management
    def _launch(self, variant: str, r: int) -> None:
        gguf = GGUF[variant]
        self.proc = subprocess.Popen(
            [str(SERVER_BIN), "-m", str(MODELS_DIR / gguf), "-ngl", "99",
             "--port", str(PLANT_PORT), "--host", "127.0.0.1",
             "-c", str(CTX_PER_SLOT * r), "--parallel", str(r)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 180
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("plant server failed health check")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PLANT_PORT}/health", timeout=2) as resp:
                    if json.load(resp).get("status") == "ok":
                        return
            except Exception:
                time.sleep(0.5)

    def _kill(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def integrate(self, now_ep: float) -> None:
        """Advance the cost integrals (call under lock)."""
        if self._last_integrate is None:
            self._last_integrate = now_ep
            return
        dt = now_ep - self._last_integrate
        if dt > 0 and self._last_integrate >= self.cfg.warmup_exclude_s:
            self.replica_seconds += self.replicas * dt
            acc = self.design["accuracy"][self.variant]
            self.variant_acc_seconds += acc * dt
            self.busy_integral += self.in_flight * dt
        self._last_integrate = now_ep

    def reconfigure(self, variant: str, r: int, now_ep: float, reason: str) -> None:
        """Drain-in-flight -> restart -> resume. Records the pause."""
        if variant == self.variant and r == self.replicas and self.proc:
            return
        t0 = time.monotonic()
        self.restarting = True
        while True:                          # drain (in-service finish, plant parity)
            with self.lock:
                if self.in_flight == 0:
                    break
            time.sleep(TICK_S)
        drain_s = time.monotonic() - t0
        self._kill()
        t1 = time.monotonic()
        self._launch(variant, r)
        load_s = time.monotonic() - t1
        with self.lock:
            old = (self.variant, self.replicas)
            self.variant, self.replicas = variant, r
            self.restarting = False
        self.pauses.append({"t_ep": now_ep, "from": old, "to": (variant, r),
                            "reason": reason, "drain_s": drain_s, "load_s": load_s})
        self.log.append(f"reconfig t={now_ep:.0f}s {old}->{(variant, r)} "
                        f"drain={drain_s:.1f}s load={load_s:.1f}s [{reason}]")

    def serve(self, req: Request, idx: int, t_ref: float, done_cb) -> None:
        """One real request in its own thread; latencies land on the Request."""
        body = json.dumps({
            "messages": [{"role": "user", "content":
                          PROMPT_TEMPLATE.format(text=REQUEST_BANK[idx % len(REQUEST_BANK)])}],
            "max_tokens": 48, "temperature": 0.7, "seed": 10_000 + idx}).encode()

        def run():
            try:
                r = urllib.request.Request(
                    f"http://127.0.0.1:{PLANT_PORT}/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(r, timeout=300):
                    pass
                req.t_done = time.monotonic() - t_ref
            except Exception as e:
                req.t_done = None
                req.error = str(e)            # counted in the report
            finally:
                with self.lock:
                    self.in_flight -= 1
                done_cb(req)
        with self.lock:
            self.in_flight += 1
        threading.Thread(target=run, daemon=True).start()


def run_real_episode(cell_name: str, cell: dict, seed: int, design: dict,
                     horizon_s: float) -> dict:
    cfg = episode_cfg(design, float(cell["cadence"]))
    cfg.horizon_s = horizon_s
    arrivals, trace_meta = make_trace(design["mean_rate_rps"], cell["I"],
                                      cfg.horizon_s, seed)
    arrivals = list(arrivals)

    # controller — identical construction to the sweep
    if cell["arm"] == "heuristic":
        controller = load_tuned(design, "heuristic", float(cell["cadence"]))()
        ctl_server = None
    else:
        intent = make_intent("full", slo_ms=design["slo_ms"],
                             abandon_s=design["abandon_s"],
                             max_replicas=design["max_replicas"],
                             variant_menu=menu_text(design),
                             switch_s=max(design["switch_load_s"].values()))
        controller = LLMController(backend="local", model="local-3B", intent=intent,
                                   seed=42, local_port=CTL_PORT, name="F-local")
        ctl_server = subprocess.Popen(
            [str(SERVER_BIN), "-m", str(MODELS_DIR / GGUF["full-3B"]), "-ngl", "99",
             "--port", str(CTL_PORT), "--host", "127.0.0.1",
             "-c", "4096", "--parallel", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 180
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("controller server failed health check")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{CTL_PORT}/health", timeout=2) as resp:
                    if json.load(resp).get("status") == "ok":
                        break
            except Exception:
                time.sleep(0.5)

    log: list[str] = []
    plant = RealPlant(design, cfg, log)
    print(f"[{time.strftime('%H:%M:%S')}] {cell_name}: warm start "
          f"{design['init_variant']} x{design['init_replicas']}", flush=True)
    plant._launch(design["init_variant"], design["init_replicas"])
    plant.variant = design["init_variant"]
    plant.replicas = design["init_replicas"]

    queue: list[tuple[Request, int]] = []
    all_requests: list[Request] = []
    completed: list[Request] = []
    window_completed: list[Request] = []
    window_arrivals = 0
    window_dropped = 0
    prev_rate = 0.0
    history: list = []
    decisions: list[Action] = []
    switches = 0
    dispatch_lag_max = 0.0
    deciding = threading.Event()
    pending_action: list = []

    done_lock = threading.Lock()

    def on_done(req: Request) -> None:
        with done_lock:
            if req.t_done is not None:
                completed.append(req)
                window_completed.append(req)

    def decide_async(obs: Observation) -> None:
        """Controller runs off-thread; action applied when it returns (real
        latency delays application, as the plant does with virtual time)."""
        def run():
            t0 = time.monotonic()
            act = controller.decide(obs, list(history))
            act.latency_s = act.latency_s or (time.monotonic() - t0)
            pending_action.append((obs, act))
            deciding.clear()
        deciding.set()
        threading.Thread(target=run, daemon=True).start()

    t_ref = time.monotonic()
    next_arrival_idx = 0
    next_wake = float(cell["cadence"])
    window_start = 0.0
    try:
        while True:
            now = time.monotonic() - t_ref
            if now >= cfg.horizon_s and not plant.in_flight:
                break
            if now >= cfg.horizon_s + 300:      # hard tail guard
                break
            with plant.lock:
                plant.integrate(min(now, cfg.horizon_s))

            # arrivals due
            while (next_arrival_idx < len(arrivals)
                   and arrivals[next_arrival_idx] <= now < cfg.horizon_s):
                sched = arrivals[next_arrival_idx]
                dispatch_lag_max = max(dispatch_lag_max, now - sched)
                req = Request(t_arrival=sched)
                all_requests.append(req)
                queue.append((req, next_arrival_idx))
                window_arrivals += 1
                next_arrival_idx += 1

            # abandonment (30 s waiting, same rule as the plant)
            still = []
            for req, idx in queue:
                if req.t_start is None and now - req.t_arrival >= cfg.abandon_s:
                    req.dropped = True
                    window_dropped += 1
                else:
                    still.append((req, idx))
            queue[:] = still

            # dispatch up to r in flight
            while queue and not plant.restarting:
                with plant.lock:
                    slot_free = plant.in_flight < plant.replicas
                if not slot_free:
                    break
                req, idx = queue.pop(0)
                req.t_start = now
                plant.serve(req, idx, t_ref, on_done)

            # apply a returned controller action
            if pending_action:
                obs0, act = pending_action.pop()
                decisions.append(act)
                history.append((obs0, act))
                new_r = max(1, min(cfg.max_replicas, act.replicas))
                new_v = act.variant if act.variant in GGUF else plant.variant
                if (new_v, new_r) != (plant.variant, plant.replicas):
                    if new_v != plant.variant:
                        switches += 1
                    plant.reconfigure(new_v, new_r, now,
                                      reason=f"decision@{obs0.t:.0f}s")

            # cadence wake
            if now >= next_wake and next_wake < cfg.horizon_s and not deciding.is_set():
                rate = window_arrivals / cfg.cadence_s
                lats = sorted((r.latency for r in window_completed), key=float)
                with plant.lock:
                    util_span = max(1e-9, plant.replicas * cfg.cadence_s)
                    obs = Observation(
                        t=next_wake, window_s=cfg.cadence_s, arrival_rate=rate,
                        prev_arrival_rate=prev_rate, queue_len=len(queue),
                        in_flight=plant.in_flight,
                        p50_ms=(lats[len(lats) // 2] * 1000 if lats else None),
                        p95_ms=(lats[min(len(lats) - 1, int(0.95 * len(lats)))] * 1000
                                if lats else None),
                        replicas=plant.replicas, variant=plant.variant,
                        utilization=min(1.0, plant.busy_integral / util_span)
                        if next_wake >= cfg.warmup_exclude_s else 0.0,
                        completed_in_window=len(window_completed),
                        dropped_in_window=window_dropped)
                    plant.busy_integral = 0.0
                decide_async(obs)
                prev_rate = rate
                window_completed, window_dropped, window_arrivals = [], 0, 0
                window_start = next_wake
                next_wake += cfg.cadence_s
            time.sleep(TICK_S)
    finally:
        plant._kill()
        if ctl_server is not None:
            ctl_server.terminate()
            try:
                ctl_server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ctl_server.kill()

    residual = [(r, i) for r, i in queue if not r.dropped]
    serve_errors = [r for r in all_requests if getattr(r, "error", None)]
    # identical scoring arithmetic: feed real records to the sweep's metric code
    scorer = Plant(build_cal(design), cfg, plant_seed=seed)
    metrics = scorer._metrics(all_requests, completed, [r for r, _ in residual],
                              decisions, switches,
                              plant.replica_seconds, plant.variant_acc_seconds)
    return {
        "cell": cell_name, "arm": cell["arm"], "model": cell["model"],
        "I": cell["I"], "cadence_s": cell["cadence"], "seed": seed,
        "horizon_s": cfg.horizon_s,
        "metrics": metrics,
        "trace_meta": {k: trace_meta[k] for k in
                       ("target_I", "mean_rate_realised", "n_arrivals")},
        "pauses": plant.pauses,
        "dispatch_lag_max_s": dispatch_lag_max,
        "serve_errors": len(serve_errors),
        "reconfig_log": log,
        "decisions": [{"t": o.t, "replicas": a.replicas, "variant": a.variant,
                       "latency_s": a.latency_s, "parse_failed": a.parse_failed}
                      for o, a in history],
        "controller_call_log": getattr(controller, "call_log", None),
        "requests": [{"t_arrival": r.t_arrival, "t_start": r.t_start,
                      "t_done": r.t_done, "dropped": r.dropped,
                      "error": getattr(r, "error", None)} for r in all_requests],
    }


def compare(real: dict, emu_row: dict) -> dict:
    em, rm = emu_row["metrics"], real["metrics"]
    keys = ["U_S03", "U_v2", "violation_rate_offered", "goodput",
            "offered", "completed", "dropped", "residual_queue"]
    out = {"cell": real["cell"], "seed": real["seed"]}
    for k in keys:
        out[k] = {"emulated": em[k], "real": rm[k],
                  "delta": (rm[k] - em[k]) if isinstance(em[k], (int, float)) else None}
    out["mean_replicas"] = {"emulated": em["components"]["mean_replicas"],
                            "real": rm["components"]["mean_replicas"]}
    out["controller"] = {"emulated": em["controller"], "real": rm["controller"]}
    out["real_only"] = {"pauses": real["pauses"],
                        "dispatch_lag_max_s": real["dispatch_lag_max_s"],
                        "serve_errors": real["serve_errors"]}
    return out


def cmd_selfcheck() -> None:
    design = load_design()
    # median-seed rule on synthetic rows
    rows = [{"arm": "x", "model": None, "I": 1.0, "cadence_s": 60, "seed": s,
             "metrics": {"U_S03": u}} for s, u in [(1, -0.3), (2, -0.1), (3, -0.2)]]
    import tempfile
    tmp = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    tmp.write_text("\n".join(dumps(r) for r in rows))
    global SWEEP
    old, SWEEP = SWEEP, tmp
    try:
        seed, row = median_seed("x", None, 1.0, 60)
        assert seed == 3 and row["metrics"]["U_S03"] == -0.2, (seed, row)
    finally:
        SWEEP = old
        tmp.unlink()
    # metrics glue: Plant._metrics over hand-built real-style records
    cfg = episode_cfg(design, 60.0)
    cfg.horizon_s = 900.0
    reqs = []
    for i in range(200):
        t = 300.0 + i * 2.5
        r = Request(t_arrival=t, t_start=t + 0.1, t_done=t + 0.6)
        reqs.append(r)
    dropped = Request(t_arrival=400.0, dropped=True)
    reqs.append(dropped)
    scorer = Plant(build_cal(design), cfg, plant_seed=1)
    m = scorer._metrics(reqs, [r for r in reqs if r.t_done], [], [], 0,
                        2.0 * 600.0, design["accuracy"]["tiny-1B4"] * 600.0)
    assert m["offered"] == 201 and m["dropped"] == 1
    assert m["violation_rate_offered"] == 1 / 201
    assert abs(m["components"]["mean_replicas"] - 2.0) < 1e-9
    print("validate_replay offline self-checks PASS (median seed, metrics glue)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "selfcheck"])
    ap.add_argument("--cell", choices=list(CELLS), default=None)
    ap.add_argument("--horizon", type=float, default=1800.0)
    args = ap.parse_args()
    if args.cmd == "selfcheck":
        cmd_selfcheck()
        return
    servers = subprocess.run(["pgrep", "-af", "llama-server"],
                             capture_output=True, text=True).stdout.strip()
    assert not servers, f"llama-server already running:\n{servers}"
    design = load_design()
    OUTDIR.mkdir(exist_ok=True)
    names = [args.cell] if args.cell else list(CELLS)
    for name in names:
        cell = CELLS[name]
        seed, emu_row = median_seed(cell["arm"], cell["model"], cell["I"],
                                    cell["cadence"])
        print(f"cell {name}: median seed {seed} "
              f"(emulated U={emu_row['metrics']['U_S03']:.3f})", flush=True)
        real = run_real_episode(name, cell, seed, design, args.horizon)
        cmp_ = compare(real, emu_row)
        write_json(OUTDIR / f"{name}_s{seed}.json",
                   {"real": real, "emulated_row": emu_row, "comparison": cmp_})
        print(json.dumps({k: cmp_[k] for k in
                          ("U_S03", "violation_rate_offered", "goodput")}, indent=1),
              flush=True)
    print("block F replays complete ->", OUTDIR)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    sys.exit(main())
