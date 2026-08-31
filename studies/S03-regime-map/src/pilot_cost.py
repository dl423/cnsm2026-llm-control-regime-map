"""Costed pilot (PROTOCOL Sec. 9 v1.1 budget-fit rule): measure real $/call,
latency, token counts, and parse reliability for each hosted controller model,
on realistic decision prompts, BEFORE the grid is sized.

50 calls per model over a scripted mix of observation states (calm / burst /
backlog / recovery), using the committed INTENT_FULL prompt. Records model id and
fingerprint per call (no dated snapshots exist for gpt-5.6). Spend flows through
the same ledger as the main runs.

Output: results/pilot_cost.json
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from artifact_io import write_json
from llm_controllers import LLMController, make_intent
from plant import Observation, VARIANT_ORDER

STUDY = Path(__file__).resolve().parents[1]
N_CALLS = 50

MENU = "\n      ".join([
    "tiny-1B4: accuracy 0.72, p50 service ~0.40 s",
    "lite-1B:  accuracy 0.69, p50 service ~0.41 s",
    "full-3B:  accuracy 0.92, p50 service ~1.17 s",
])

SCENARIOS = [
    # (arrival, prev, queue, in_flight, p50, p95, replicas, variant, util, comp, drop)
    (0.4, 0.5, 0, 1, 420.0, 980.0, 2, "lite-1B", 0.22, 24, 0),      # calm
    (4.8, 0.6, 22, 2, 1900.0, 2900.0, 2, "lite-1B", 0.98, 55, 3),   # burst onset
    (0.3, 4.5, 61, 2, 5200.0, 9400.0, 4, "lite-1B", 1.0, 40, 12),   # backlog after burst
    (1.8, 2.0, 3, 3, 900.0, 2100.0, 4, "full-3B", 0.75, 90, 0),     # steady mid
    (0.2, 0.2, 0, 0, 380.0, 850.0, 4, "full-3B", 0.08, 12, 0),      # over-provisioned
]


def obs_at(i: int) -> Observation:
    s = SCENARIOS[i % len(SCENARIOS)]
    return Observation(t=300.0 + 60.0 * i, window_s=60.0, arrival_rate=s[0],
                       prev_arrival_rate=s[1], queue_len=s[2], in_flight=s[3],
                       p50_ms=s[4], p95_ms=s[5], replicas=s[6], variant=s[7],
                       utilization=s[8], completed_in_window=s[9],
                       dropped_in_window=s[10])


def pilot(backend: str, model: str) -> dict:
    intent = make_intent("full", slo_ms=3000, abandon_s=30, max_replicas=6,
                         variant_menu=MENU, switch_s=8)
    ctl = LLMController(backend=backend, model=model, intent=intent,
                        seed=42, name=f"pilot-{model}")
    history: list = []
    t0 = time.time()
    for i in range(N_CALLS):
        obs = obs_at(i)
        act = ctl.decide(obs, history)
        history.append((obs, act))
        if len(history) > 6:
            history.pop(0)
    wall = time.time() - t0
    ok = [a for _, a in history]  # last few only; use call_log for full stats
    calls = [c for c in ctl.call_log if "raw" in c]
    errors = [c for c in ctl.call_log if "error" in c]
    lat = sorted(c["latency_s"] for c in calls)
    usd = [c["usd"] for c in calls]
    tin = [c["meta"]["usage"].get("prompt_tokens") or 0 for c in calls]
    tout = [c["meta"]["usage"].get("completion_tokens") or 0 for c in calls]
    parse_fails = sum(1 for c in calls if '"replicas"' not in c["raw"])
    model_ids = sorted({c["meta"].get("model_id") for c in calls})
    fingerprints = sorted({str(c["meta"].get("fingerprint")) for c in calls})
    return {
        "backend": backend, "model": model, "n_calls_attempted": N_CALLS,
        "n_api_success": len(calls), "n_api_errors": len(errors),
        "parse_fail_heuristic": parse_fails,
        "latency_s": {"mean": statistics.fmean(lat) if lat else None,
                      "p50": lat[len(lat)//2] if lat else None,
                      "p95": lat[min(len(lat)-1, int(0.95*len(lat)))] if lat else None},
        "tokens": {"in_mean": statistics.fmean(tin) if tin else None,
                   "out_mean": statistics.fmean(tout) if tout else None},
        "usd": {"per_call_mean": statistics.fmean(usd) if usd else None,
                "total": sum(usd)},
        "model_ids_seen": model_ids, "fingerprints_seen": fingerprints,
        "wall_s": wall,
        "errors": [c["error"] for c in errors][:5],
    }


def main() -> None:
    out = {"models": {}}
    for backend, model in (("openai", "gpt-5.6-luna"), ("anthropic", "claude-sonnet-5")):
        print(f"pilot: {model} ...", flush=True)
        res = pilot(backend, model)
        out["models"][model] = res
        print(f"  ${res['usd']['per_call_mean']:.6f}/call, "
              f"lat p50 {res['latency_s']['p50']:.2f}s, "
              f"tok {res['tokens']['in_mean']:.0f}/{res['tokens']['out_mean']:.0f}, "
              f"api_err {res['n_api_errors']}", flush=True)
    path = STUDY / "results" / "pilot_cost.json"
    write_json(path, out)
    print("wrote", path)


if __name__ == "__main__":
    import os
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    main()
