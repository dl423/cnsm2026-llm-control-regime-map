"""Baseline calibration: per-cadence Optuna tuning of every rule-based family on
held-out calibration traces, against the SAME U_S03 the LLM arm is scored on.

Fairness controls (S01 D-039 lessons, R5):
- Tuned PER CADENCE — temporal parameters are in seconds and re-tuned when the
  decision interval changes (the S01 v1 units trap).
- Range-matched search spaces across families (search breadth parity: a family
  must not win or lose because its space was wider).
- Calibration traces (seeds 1000+) are disjoint from evaluation traces (seeds
  1..N_SEEDS); tuning sees a mixed-volatility set because deployed controllers
  do not know I in advance.
- The static arm is calibrated too: exhaustive grid over (replicas, variant).

Output: results/tuning/<family>_c<cadence>.json (best params + all trials).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import optuna

from artifact_io import write_json
from controllers import HeuristicController, QueueAwareController, StaticController
from plant import Calibration, EpisodeConfig, Plant, VARIANT_ORDER
from workload import make_trace

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "results" / "tuning"

CAL_SEEDS = list(range(1000, 1010))       # 10 calibration traces per I
CAL_I = None                              # set from design.json I_levels at load
N_TRIALS = 60
MEAN_RATE = None                          # set from design.json at load time


def load_design() -> dict:
    """Plant calibration + episode constants fixed by D-010 (design.json is the
    committed bridge between Layer-A data and the sweep)."""
    return json.loads((STUDY / "data" / "design.json").read_text())


def build_cal(design: dict) -> Calibration:
    return Calibration.from_layer_a(
        mode=design["capacity_mode"],
        accuracy={v: design["accuracy"][v] for v in VARIANT_ORDER},
        switch_load_s={v: design["switch_load_s"][v] for v in VARIANT_ORDER})


def episode_cfg(design: dict, cadence_s: float) -> EpisodeConfig:
    return EpisodeConfig(
        horizon_s=design["horizon_s"], warmup_exclude_s=design["warmup_exclude_s"],
        slo_ms=design["slo_ms"], abandon_s=design["abandon_s"],
        max_replicas=design["max_replicas"], cadence_s=cadence_s)


def secondary_cfg(design: dict, cadence_s: float) -> EpisodeConfig:
    """Episode config expressing the block-E secondary objective THROUGHOUT:
    tightened SLO 2000 ms, weights (0.7, 0.1, 0.2) — the objective INTENT_SECONDARY
    announces at t=900 s. Used only to tune the rule arms' post-switch parameters
    (the 'equivalent parameter switch' of PROTOCOL Sec. 5): a rule controller
    correctly reconfigured for the new objective is one tuned against it."""
    cfg = episode_cfg(design, cadence_s)
    cfg.slo_ms = cfg.secondary_slo_ms
    cfg.w_slo, cfg.w_acc, cfg.w_cost = cfg.secondary_w
    return cfg


def mean_utility(controller_factory, cal: Calibration, cfg: EpisodeConfig,
                 design: dict) -> float:
    us = []
    for I in design["I_levels"]:
        for seed in CAL_SEEDS:
            arrivals, _ = make_trace(design["mean_rate_rps"], I, cfg.horizon_s, seed)
            ctl = controller_factory()
            plant = Plant(cal, cfg, plant_seed=seed)
            m = plant.run(list(arrivals), ctl,
                          init_replicas=design["init_replicas"],
                          init_variant=design["init_variant"])
            us.append(m["U_S03"])
    return sum(us) / len(us)


def tune_family(family: str, cadence_s: float, cal: Calibration,
                design: dict, cfg: EpisodeConfig | None = None,
                ctl_slo_ms: float | None = None) -> dict:
    cfg = cfg if cfg is not None else episode_cfg(design, cadence_s)
    ctl_slo = ctl_slo_ms if ctl_slo_ms is not None else design["slo_ms"]
    caps = design["capacity_rps"]

    if family == "static":
        best, best_u, trials = None, -1e9, []
        for r in range(1, design["max_replicas"] + 1):
            for v in VARIANT_ORDER:
                u = mean_utility(lambda r=r, v=v: StaticController(replicas=r, variant=v),
                                 cal, cfg, design)
                trials.append({"replicas": r, "variant": v, "U": u})
                if u > best_u:
                    best, best_u = {"replicas": r, "variant": v}, u
        return {"family": family, "cadence_s": cadence_s, "best_params": best,
                "best_U": best_u, "trials": trials}

    def objective(trial: optuna.Trial) -> float:
        if family == "heuristic":
            params = dict(
                target_util=trial.suggest_float("target_util", 0.3, 0.95),
                scale_down_headroom=trial.suggest_float("scale_down_headroom", 0.05, 0.7),
                cooldown_s=trial.suggest_float("cooldown_s", 0.0, 600.0),
                degrade_p95_ratio=trial.suggest_float("degrade_p95_ratio", 0.5, 2.0),
                upgrade_p95_ratio=trial.suggest_float("upgrade_p95_ratio", 0.1, 0.9),
                upgrade_persist_windows=trial.suggest_int("upgrade_persist_windows", 1, 5),
            )
            factory = lambda: HeuristicController(
                capacity_rps=caps, slo_ms=ctl_slo,
                max_replicas=design["max_replicas"], **params)
        elif family == "queue_aware":
            params = dict(
                drain_target_s=trial.suggest_float("drain_target_s", 5.0, 600.0),
                target_util=trial.suggest_float("target_util", 0.3, 0.95),
                hold_queue_len=trial.suggest_int("hold_queue_len", 0, 20),
                cooldown_s=trial.suggest_float("cooldown_s", 0.0, 600.0),
                upgrade_p95_ratio=trial.suggest_float("upgrade_p95_ratio", 0.1, 0.9),
            )
            factory = lambda: QueueAwareController(
                capacity_rps=caps, slo_ms=ctl_slo,
                max_replicas=design["max_replicas"], **params)
        else:
            raise ValueError(family)
        return mean_utility(factory, cal, cfg, design)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=17))
    study.optimize(objective, n_trials=N_TRIALS)
    return {"family": family, "cadence_s": cadence_s,
            "best_params": study.best_params, "best_U": study.best_value,
            "n_trials": N_TRIALS,
            "trials": [{"params": t.params, "U": t.value} for t in study.trials]}


def main(secondary: bool = False) -> None:
    design = load_design()
    cal = build_cal(design)
    OUT.mkdir(parents=True, exist_ok=True)
    if secondary:
        # Block-E rule-arm parity (PROTOCOL Sec. 5, R7; D-017): tune each family
        # at the intent-change cadence against the secondary objective.
        cadence = 60.0
        cfg = secondary_cfg(design, cadence)
        for family in ("static", "heuristic", "queue_aware"):
            t0 = time.time()
            res = tune_family(family, cadence, cal, design,
                              cfg=cfg, ctl_slo_ms=cfg.slo_ms)
            res["wall_s"] = time.time() - t0
            res["cal_seeds"] = CAL_SEEDS
            res["cal_I"] = design["I_levels"]
            res["objective"] = {"name": "secondary", "slo_ms": cfg.slo_ms,
                                "weights": [cfg.w_slo, cfg.w_acc, cfg.w_cost]}
            path = OUT / f"{family}_c{int(cadence)}_secondary.json"
            write_json(path, res)
            print(f"{family} @ {cadence}s SECONDARY: best U={res['best_U']:.4f} "
                  f"({res['wall_s']:.0f}s) -> {path.name}", flush=True)
        return
    for cadence in design["cadences_s"]:
        for family in ("static", "heuristic", "queue_aware"):
            t0 = time.time()
            res = tune_family(family, float(cadence), cal, design)
            res["wall_s"] = time.time() - t0
            res["cal_seeds"] = CAL_SEEDS
            res["cal_I"] = design["I_levels"]
            path = OUT / f"{family}_c{int(cadence)}.json"
            write_json(path, res)
            print(f"{family} @ {cadence}s: best U={res['best_U']:.4f} "
                  f"({res['wall_s']:.0f}s) -> {path.name}", flush=True)


if __name__ == "__main__":
    import sys
    main(secondary="--secondary" in sys.argv)
