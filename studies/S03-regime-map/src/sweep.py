"""Main sweep orchestrator (PROTOCOL Sec. 9, sized by v1.3b).

Blocks:
  B       rule arms (static/heuristic/queue_aware, tuned per cadence) —
          full grid, 30 seeds, CPU-only.
  C       LLM-periodic intent-full: gpt-5.6-luna full grid (20 seeds);
          claude-sonnet-5 at 60 s cadence x all core I (20 seeds);
          local-3B full grid (20 seeds, needs llama-server on :8200).
  E       LLM-advantage channels (R7): intent-change at t=900 s and
          self-triggered (invocation-matched vs 60 s periodic);
          luna + local, I in {400, 1000}, 20 seeds.
  H       intent-partial (R2 declared arm): luna + local, 60 s, all core I, 20 seeds.

Pairing: every arm sees the identical trace for a given (I, seed) — traces are
regenerated deterministically from (mean_rate, I, horizon, seed).

Resume: results/sweep.jsonl keyed by (block, arm, model, I, cadence, seed);
existing keys are skipped. Raw decision logs go to results/decisions/<key>.json.gz
(hosted/local arms only). Run: python3 sweep.py <block> [--dry]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from artifact_io import append_jsonl, write_json_gz
from controllers import HeuristicController, QueueAwareController, StaticController
from fairness import build_cal, episode_cfg, load_design
from llm_controllers import LLMController, make_intent
from plant import EpisodeConfig, Plant
from workload import make_trace

STUDY = Path(__file__).resolve().parents[1]
SWEEP = STUDY / "results" / "sweep.jsonl"
DECDIR = STUDY / "results" / "decisions"

RULE_SEEDS = list(range(1, 31))       # 30 (R4; CPU-free)
LLM_SEEDS = list(range(1, 21))        # 20 paired seeds per reported LLM cell
HOSTED_PARALLEL = 4

LOCAL_MODEL = "local-3B"              # PROTOCOL 4a rule; pilot_local.json
LOCAL_PORT = 8200


def menu_text(design: dict) -> str:
    rows = []
    for v in ("tiny-1B4", "lite-1B", "full-3B"):
        rows.append(f"{v}: accuracy {design['accuracy'][v]:.2f}, "
                    f"p50 service ~{ {'tiny-1B4': 0.40, 'lite-1B': 0.41, 'full-3B': 1.17}[v]:.2f} s")
    return "\n      ".join(rows)


def existing_keys() -> set:
    keys = set()
    if SWEEP.exists():
        for line in SWEEP.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add(r["key"])
            except Exception:
                continue
    return keys


def record(key: str, payload: dict) -> None:
    payload["key"] = key
    append_jsonl(SWEEP, payload)


def run_episode(design, cal, cfg: EpisodeConfig, controller, I: float, seed: int) -> dict:
    arrivals, trace_meta = make_trace(design["mean_rate_rps"], I, cfg.horizon_s, seed)
    plant = Plant(cal, cfg, plant_seed=seed)
    metrics = plant.run(list(arrivals), controller,
                        init_replicas=design["init_replicas"],
                        init_variant=design["init_variant"])
    metrics["trace"] = {k: trace_meta[k] for k in
                        ("target_I", "realised_I", "mean_rate_realised", "n_arrivals")}
    return metrics


def load_tuned(design, family: str, cadence: float):
    path = STUDY / "results" / "tuning" / f"{family}_c{int(cadence)}.json"
    best = json.loads(path.read_text())["best_params"]
    caps = design["capacity_rps"]
    if family == "static":
        return lambda: StaticController(**best)
    if family == "heuristic":
        return lambda: HeuristicController(capacity_rps=caps, slo_ms=design["slo_ms"],
                                           max_replicas=design["max_replicas"], **best)
    if family == "queue_aware":
        return lambda: QueueAwareController(capacity_rps=caps, slo_ms=design["slo_ms"],
                                            max_replicas=design["max_replicas"], **best)
    raise ValueError(family)


def block_B(design, cal, done: set, dry: bool) -> None:
    for cadence in design["cadences_s"]:
        cfg = episode_cfg(design, float(cadence))
        for family in ("static", "heuristic", "queue_aware"):
            factory = load_tuned(design, family, float(cadence))
            for I in design["I_levels"]:
                for seed in RULE_SEEDS:
                    key = f"B|{family}|-|I{I:g}|c{cadence}|s{seed}"
                    if key in done:
                        continue
                    if dry:
                        print("would run", key)
                        continue
                    m = run_episode(design, cal, cfg, factory(), I, seed)
                    record(key, {"block": "B", "arm": family, "model": None,
                                 "I": I, "cadence_s": cadence, "seed": seed,
                                 "metrics": m})
        print(f"B @ {cadence}s done", flush=True)


def llm_cells(design) -> list[dict]:
    """Hosted/local cells for block C per the v1.3b sizing."""
    cells = []
    for I in design["I_levels"]:
        for cadence in design["cadences_s"]:
            for seed in LLM_SEEDS:
                cells.append({"backend": "openai", "model": "gpt-5.6-luna",
                              "I": I, "cadence": cadence, "seed": seed})
                cells.append({"backend": "local", "model": LOCAL_MODEL,
                              "I": I, "cadence": cadence, "seed": seed})
                if cadence == 60:
                    cells.append({"backend": "anthropic", "model": "claude-sonnet-5",
                                  "I": I, "cadence": cadence, "seed": seed})
    return cells


def run_llm_cell(design, cal, cell: dict, intent_kind: str = "full",
                 block: str = "C", self_triggered: bool = False,
                 intent_switch: bool = False) -> tuple[str, dict]:
    cadence = cell["cadence"]
    cfg = episode_cfg(design, float(cadence))
    if intent_switch:
        cfg.intent_switch_at_s = 900.0
    intent = make_intent(intent_kind, slo_ms=design["slo_ms"],
                         abandon_s=design["abandon_s"],
                         max_replicas=design["max_replicas"],
                         variant_menu=menu_text(design),
                         switch_s=max(design["switch_load_s"].values()))
    ctl = LLMController(backend=cell["backend"], model=cell["model"], intent=intent,
                        self_triggered=self_triggered, seed=42,
                        local_port=LOCAL_PORT, name=f"{block}-{cell['model']}")
    if intent_switch:
        secondary = make_intent("secondary", slo_ms=cfg.secondary_slo_ms,
                                abandon_s=design["abandon_s"],
                                max_replicas=design["max_replicas"],
                                variant_menu=menu_text(design),
                                switch_s=max(design["switch_load_s"].values()))
        base_decide = ctl.decide

        def switching_decide(obs, history):
            if obs.t >= 900.0 and secondary not in ctl.intent:
                ctl.intent = intent + "\n\n" + secondary
            return base_decide(obs, history)
        ctl.decide = switching_decide
    m = run_episode(design, cal, cfg, ctl, cell["I"], cell["seed"])
    arm = ("llm_self" if self_triggered else
           "llm_switch" if intent_switch else
           f"llm_{intent_kind}")
    key = f"{block}|{arm}|{cell['model']}|I{cell['I']:g}|c{cadence}|s{cell['seed']}"
    write_json_gz(DECDIR / (key.replace("|", "_") + ".json.gz"), ctl.call_log)
    return key, {"block": block, "arm": arm, "model": cell["model"],
                 "I": cell["I"], "cadence_s": cadence, "seed": cell["seed"],
                 "metrics": m}


def block_C(design, cal, done: set, dry: bool) -> None:
    cells = [c for c in llm_cells(design)]
    todo = []
    for c in cells:
        key = f"C|llm_full|{c['model']}|I{c['I']:g}|c{c['cadence']}|s{c['seed']}"
        if key not in done:
            todo.append(c)
    print(f"C: {len(todo)}/{len(cells)} cells to run", flush=True)
    if dry:
        return
    local_cells = [c for c in todo if c["backend"] == "local"]
    hosted_cells = [c for c in todo if c["backend"] != "local"]
    with ThreadPoolExecutor(max_workers=HOSTED_PARALLEL) as ex:
        futs = {ex.submit(run_llm_cell, design, cal, c): c for c in hosted_cells}
        for fut in as_completed(futs):
            key, payload = fut.result()
            record(key, payload)
            print("done", key, f"U={payload['metrics']['U_S03']:.3f}", flush=True)
    # local runs serially (single GPU server, byte-stable serial serving)
    for c in local_cells:
        key, payload = run_llm_cell(design, cal, c)
        record(key, payload)
        print("done", key, f"U={payload['metrics']['U_S03']:.3f}", flush=True)


class SwitchingRuleController:
    """Rule-arm counterpart of the block-E intent change (PROTOCOL Sec. 5, R7;
    D-017): delegates to the primary-tuned controller before the switch instant
    and to the secondary-objective-tuned controller from t >= switch_at_s.
    The secondary instance starts with fresh internal state (cooldowns reset) —
    the rule-arm equivalent of being handed a new objective."""

    def __init__(self, primary, secondary, switch_at_s: float = 900.0):
        self.primary, self.secondary, self.switch_at_s = primary, secondary, switch_at_s
        self.name = f"{getattr(primary, 'name', 'rule')}_switch"

    def decide(self, obs, history):
        ctl = self.primary if obs.t < self.switch_at_s else self.secondary
        return ctl.decide(obs, history)


def load_tuned_secondary(design, family: str):
    """Secondary-objective tuning (fairness.py --secondary); None if not tuned yet."""
    path = STUDY / "results" / "tuning" / f"{family}_c60_secondary.json"
    if not path.exists():
        return None
    art = json.loads(path.read_text())
    best = art["best_params"]
    slo = art["objective"]["slo_ms"]
    caps = design["capacity_rps"]
    if family == "static":
        return lambda: StaticController(**best)
    if family == "heuristic":
        return lambda: HeuristicController(capacity_rps=caps, slo_ms=slo,
                                           max_replicas=design["max_replicas"], **best)
    if family == "queue_aware":
        return lambda: QueueAwareController(capacity_rps=caps, slo_ms=slo,
                                            max_replicas=design["max_replicas"], **best)
    raise ValueError(family)


def block_E(design, cal, done: set, dry: bool) -> None:
    # Rule arms under the same objective switch, on the same traces (R7 parity).
    for family in ("static", "heuristic", "queue_aware"):
        sec_factory = load_tuned_secondary(design, family)
        if sec_factory is None:
            print(f"E: SKIPPING {family}_switch — run `fairness.py --secondary` "
                  "first (tuning artifact missing)", flush=True)
            continue
        pri_factory = load_tuned(design, family, 60.0)
        for I in (400.0, 1000.0):
            for seed in RULE_SEEDS:
                key = f"E|{family}_switch|-|I{I:g}|c60|s{seed}"
                if key in done:
                    continue
                if dry:
                    print("would run", key)
                    continue
                cfg = episode_cfg(design, 60.0)
                cfg.intent_switch_at_s = 900.0
                ctl = SwitchingRuleController(pri_factory(), sec_factory())
                m = run_episode(design, cal, cfg, ctl, I, seed)
                record(key, {"block": "E", "arm": f"{family}_switch", "model": None,
                             "I": I, "cadence_s": 60, "seed": seed, "metrics": m})
        print(f"E rule arm {family}_switch done", flush=True)
    for model, backend in (("gpt-5.6-luna", "openai"), (LOCAL_MODEL, "local")):
        for I in (400.0, 1000.0):
            for seed in LLM_SEEDS:
                for mode in ("switch", "self"):
                    cell = {"backend": backend, "model": model, "I": I, "cadence": 60,
                            "seed": seed}
                    arm = "llm_switch" if mode == "switch" else "llm_self"
                    key = f"E|{arm}|{model}|I{I:g}|c60|s{seed}"
                    if key in done:
                        continue
                    if dry:
                        print("would run", key)
                        continue
                    key, payload = run_llm_cell(
                        design, cal, cell, block="E",
                        self_triggered=(mode == "self"),
                        intent_switch=(mode == "switch"))
                    record(key, payload)
                    print("done", key, flush=True)


def block_H(design, cal, done: set, dry: bool) -> None:
    for model, backend in (("gpt-5.6-luna", "openai"), (LOCAL_MODEL, "local")):
        for I in design["I_levels"]:
            for seed in LLM_SEEDS:
                cell = {"backend": backend, "model": model, "I": I, "cadence": 60,
                        "seed": seed}
                key = f"H|llm_partial|{model}|I{I:g}|c60|s{seed}"
                if key in done:
                    continue
                if dry:
                    print("would run", key)
                    continue
                key, payload = run_llm_cell(design, cal, cell, intent_kind="partial",
                                            block="H")
                record(key, payload)
                print("done", key, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["B", "C", "E", "H"])
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    design = load_design()
    cal = build_cal(design)
    done = existing_keys()
    {"B": block_B, "C": block_C, "E": block_E, "H": block_H}[args.block](
        design, cal, done, args.dry)


if __name__ == "__main__":
    import os
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    main()
