"""R9 follow-up to the D-020/D-021 adversarial audit: does luna's I=1000 win
die to a VOLATILITY-INFORMED static baseline?

The pre-registered envelope tunes each rule family per cadence on a mixed-I
calibration set (deployed controllers don't know I in advance — fairness.py).
The auditor's live alternative explanation: the high-I win reflects that
compromise, not control skill. Settling test, protocol-consistent:

  1. tune static PER-I: exhaustive (replicas, variant) grid on the held-out
     calibration seeds 1000-1009 at I=1000 only (same seeds the envelope's own
     tuning used — no evaluation seed touches selection);
  2. evaluate the winning config on evaluation seeds 1..20 at cadences 15/60 s;
  3. paired DeltaU = U_llm(seed) - U_oracle_static(seed), t-CI.

"Oracle" = the config selection knows the deployment's volatility regime; the
LLM does not (it only observes arrivals online). Reported as a labelled
robustness check, not a replacement envelope.

Output: results/oracle_static.jsonl (raw episodes) +
results/analysis/oracle_static_I1000.json (summary).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from artifact_io import dumps, write_json
from controllers import StaticController
from fairness import build_cal, episode_cfg, load_design
from plant import Plant, VARIANT_ORDER
from workload import make_trace

STUDY = Path(__file__).resolve().parents[1]
RAW = STUDY / "results" / "oracle_static.jsonl"
OUT = STUDY / "results" / "analysis" / "oracle_static_I1000.json"
CAL_SEEDS = list(range(1000, 1010))
EVAL_SEEDS = list(range(1, 21))
I = 1000.0
CADENCES = (15.0, 60.0)


def run_episode(design, cal, cfg, r, v, seed):
    arrivals, _ = make_trace(design["mean_rate_rps"], I, cfg.horizon_s, seed)
    plant = Plant(cal, cfg, plant_seed=seed)
    return plant.run(list(arrivals), StaticController(replicas=r, variant=v),
                     init_replicas=design["init_replicas"],
                     init_variant=design["init_variant"])


def t975(df):
    from analyze import t975 as t
    return t(df)


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    half = t975(n - 1) * math.sqrt(var / n)
    return {"mean": m, "n": n, "ci95": [m - half, m + half]}


def main():
    design = load_design()
    cal = build_cal(design)
    llm = {}
    for line in (STUDY / "results" / "sweep.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if (row.get("block") == "C" and row.get("model") == "gpt-5.6-luna"
                and row["I"] == I):
            llm[(row["cadence_s"], row["seed"])] = row["metrics"]["U_S03"]
    summary = {"design": "tune static per-I on cal seeds 1000-1009 at I=1000; "
                         "evaluate winner on eval seeds 1-20; paired vs luna",
               "cells": {}}
    raw = RAW.open("a")
    for cadence in CADENCES:
        cfg = episode_cfg(design, cadence)
        # 1) per-I tuning on calibration seeds
        best, best_u, trials = None, -1e9, []
        for r in range(1, design["max_replicas"] + 1):
            for v in VARIANT_ORDER:
                us = [run_episode(design, cal, cfg, r, v, s)["U_S03"]
                      for s in CAL_SEEDS]
                u = sum(us) / len(us)
                trials.append({"replicas": r, "variant": v, "cal_U": u})
                if u > best_u:
                    best, best_u = (r, v), u
        # 2) evaluate winner on eval seeds
        deltas, evals = [], {}
        for s in EVAL_SEEDS:
            m = run_episode(design, cal, cfg, best[0], best[1], s)
            evals[s] = m["U_S03"]
            raw.write(dumps({"cell": f"I1000|c{cadence:g}", "config": best,
                             "seed": s, "metrics": m}) + "\n")
            deltas.append(llm[(int(cadence), s)] - m["U_S03"])
        st = mean_ci(deltas)
        summary["cells"][f"I1000|c{cadence:g}"] = {
            "oracle_config": {"replicas": best[0], "variant": best[1],
                              "cal_U": best_u},
            "tuning_trials": trials,
            "oracle_eval_U": mean_ci(list(evals.values())),
            "llm_U": mean_ci([llm[(int(cadence), s)] for s in EVAL_SEEDS]),
            "delta_U_paired_llm_minus_oracle": st,
            "llm_beats_oracle_static": bool(st["ci95"][0] > 0),
            "oracle_beats_llm": bool(st["ci95"][1] < 0),
        }
        print(f"c{cadence:g}: oracle static = {best} (cal U {best_u:.4f}); "
              f"paired dU(llm-oracle) = {st['mean']:+.4f} "
              f"CI [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]", flush=True)
    raw.close()
    write_json(OUT, summary)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
