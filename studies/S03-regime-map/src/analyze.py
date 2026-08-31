"""S03 analysis: structured experiment evidence -> derived JSON tables.

The analysis is intentionally independent of manuscript generation. Per-cell
tables carry every cell, sample count, and confidence interval; both utility
metrics and the recorded failure modes are retained.

Statistical conventions (fixed here, before results were inspected):
- Cell aggregate: mean over seeds, 95% CI via Student t (n-1 df).
- Paired margin: for each LLM cell (arm, model, I, cadence), DeltaU vs the
  BASELINE ENVELOPE = the rule family with the best mean U_S03 in that same
  (I, cadence) cell (computed over the paired seed set); the margin is paired
  per seed, CI via t on the per-seed deltas. Envelope family named per cell.
- Realised volatility per I (D-015): disjoint 5-cycle windows pooled across the
  evaluation seeds' regenerated traces; window count reported.

Usage:
  python3 analyze.py tables --input-results ../results --output-dir /tmp/analysis
  python3 analyze.py selfcheck
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from artifact_io import write_json

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / "results"
OUT = RES / "analysis"

# Student t 97.5% quantiles; df>30 -> 1.96 (adequate at our n)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t975(df: int) -> float:
    return T975.get(df, 1.96) if df >= 1 else float("nan")


def mean_ci(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"mean": None, "n": 0, "ci95": None}
    m = sum(xs) / n
    if n == 1:
        return {"mean": m, "n": 1, "ci95": None}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    half = t975(n - 1) * math.sqrt(var / n)
    return {"mean": m, "n": n, "ci95": [m - half, m + half], "sd": math.sqrt(var)}


def load_sweep() -> list[dict]:
    rows = []
    for line in (RES / "sweep.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def arm_id(r: dict) -> str:
    return r["arm"] if r["model"] is None else f"{r['arm']}|{r['model']}"


# ------------------------------------------------------------------ RQ1 tables

RULE_FAMILIES = ("static", "heuristic", "queue_aware")


def cell_table(rows: list[dict]) -> dict:
    """Per (block, arm|model, I, cadence): distribution of U_S03/U_v2/components."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["block"], arm_id(r), f"I{r['I']:g}", f"c{r['cadence_s']}")].append(r)
    table = {}
    for key, rs in sorted(cells.items()):
        m = {
            "U_S03": mean_ci([x["metrics"]["U_S03"] for x in rs]),
            "U_v2": mean_ci([x["metrics"]["U_v2"] for x in rs]),
            "violation_rate_offered": mean_ci(
                [x["metrics"]["violation_rate_offered"] for x in rs]),
            "goodput": mean_ci([x["metrics"]["goodput"] for x in rs]),
            "acc_deficit": mean_ci(
                [x["metrics"]["components"]["acc_deficit"] for x in rs]),
            "mean_replicas": mean_ci(
                [x["metrics"]["components"]["mean_replicas"] for x in rs]),
            "seeds": sorted(x["seed"] for x in rs),
        }
        c = [x["metrics"]["controller"] for x in rs]
        m["controller"] = {
            "invocations": mean_ci([y["invocations"] for y in c]),
            "parse_failures_total": sum(y["parse_failures"] for y in c),
            "repaired_total": sum(y["repaired_parses"] for y in c),
            "wall_s": mean_ci([y["wall_s_total"] for y in c]),
            "tokens_in": mean_ci([y["tokens_in"] for y in c]),
            "tokens_out": mean_ci([y["tokens_out"] for y in c]),
            "usd_per_episode": mean_ci([y["usd"] for y in c]),
        }
        table["|".join(key)] = m
    return table


def envelope_and_margins(rows: list[dict]) -> dict:
    """Per LLM cell: paired DeltaU vs the best-mean rule family in that cell."""
    by_cell_family = defaultdict(dict)     # (I,c) -> family -> {seed: U}
    for r in rows:
        if r["block"] == "B" and r["arm"] in RULE_FAMILIES:
            by_cell_family[(f"I{r['I']:g}", f"c{r['cadence_s']}")].setdefault(
                r["arm"], {})[r["seed"]] = r["metrics"]["U_S03"]
    llm = defaultdict(dict)                # (arm|model, I, c) -> {seed: U}
    for r in rows:
        if r["block"] == "C" and r["arm"] == "llm_full":
            llm[(arm_id(r), f"I{r['I']:g}", f"c{r['cadence_s']}")][r["seed"]] = \
                r["metrics"]["U_S03"]
    out = {}
    for (am, I, c), seed_u in sorted(llm.items()):
        fams = by_cell_family.get((I, c), {})
        if not fams:
            continue
        paired_seeds = sorted(set(seed_u) & set.intersection(
            *[set(v) for v in fams.values()]))
        env_family = max(fams, key=lambda f: sum(fams[f][s] for s in paired_seeds)
                         / len(paired_seeds))
        deltas = [seed_u[s] - fams[env_family][s] for s in paired_seeds]
        stats = mean_ci(deltas)
        sig = (stats["ci95"] is not None
               and (stats["ci95"][0] > 0 or stats["ci95"][1] < 0))
        out[f"{am}|{I}|{c}"] = {
            "envelope_family": env_family,
            "delta_U_paired": stats, "significant_95": sig,
            "llm_U": mean_ci([seed_u[s] for s in paired_seeds]),
            "envelope_U": mean_ci([fams[env_family][s] for s in paired_seeds]),
            "paired_seeds": paired_seeds,
        }
    return out


def blocks_EH_tables(rows: list[dict]) -> dict:
    """E: llm_switch vs rule *_switch (paired); llm_self vs periodic C c60.
    H: llm_partial vs llm_full C c60 (paired) + over-provisioning check (H3)."""
    def seed_map(pred):
        m = defaultdict(dict)
        for r in rows:
            if pred(r):
                m[(arm_id(r), f"I{r['I']:g}")][r["seed"]] = r
        return m

    e_llm = seed_map(lambda r: r["block"] == "E" and r["arm"] == "llm_switch")
    e_rule = seed_map(lambda r: r["block"] == "E" and r["arm"].endswith("_switch")
                      and r["model"] is None)
    e_self = seed_map(lambda r: r["block"] == "E" and r["arm"] == "llm_self")
    c60 = seed_map(lambda r: r["block"] == "C" and r["cadence_s"] == 60)
    h = seed_map(lambda r: r["block"] == "H")

    out = {"intent_change": {}, "self_triggered": {}, "intent_partial": {}}
    # channel 1: llm_switch vs best switching rule family (paired per seed)
    for (am, I), sm in sorted(e_llm.items()):
        fams = {a: v for (a, i2), v in e_rule.items() if i2 == I}
        if not fams:
            continue
        paired = sorted(set(sm) & set.intersection(*[set(v) for v in fams.values()]))
        if not paired:
            continue
        env = max(fams, key=lambda f: sum(fams[f][s]["metrics"]["U_S03"]
                                          for s in paired) / len(paired))
        deltas = [sm[s]["metrics"]["U_S03"] - fams[env][s]["metrics"]["U_S03"]
                  for s in paired]
        out["intent_change"][f"{am}|{I}"] = {
            "envelope_family": env, "delta_U_paired": mean_ci(deltas),
            "llm_U": mean_ci([sm[s]["metrics"]["U_S03"] for s in paired]),
            "envelope_U": mean_ci([fams[env][s]["metrics"]["U_S03"] for s in paired]),
            "n_pairs": len(paired)}
    # channel 2: self-triggered vs its periodic-c60 twin, same model, same I
    for (am, I), sm in sorted(e_self.items()):
        model = am.split("|", 1)[1]
        twin = c60.get((f"llm_full|{model}", I), {})
        paired = sorted(set(sm) & set(twin))
        if not paired:
            continue
        out["self_triggered"][f"{am}|{I}"] = {
            "delta_U_paired": mean_ci([sm[s]["metrics"]["U_S03"]
                                       - twin[s]["metrics"]["U_S03"] for s in paired]),
            "self_U": mean_ci([sm[s]["metrics"]["U_S03"] for s in paired]),
            "periodic_U": mean_ci([twin[s]["metrics"]["U_S03"] for s in paired]),
            "self_invocations": mean_ci([sm[s]["metrics"]["controller"]["invocations"]
                                         for s in paired]),
            "periodic_invocations": mean_ci(
                [twin[s]["metrics"]["controller"]["invocations"] for s in paired]),
            "n_pairs": len(paired)}
    # H3: partial vs full on identical traces (same model, c60)
    for (am, I), sm in sorted(h.items()):
        model = am.split("|", 1)[1]
        full = c60.get((f"llm_full|{model}", I), {})
        paired = sorted(set(sm) & set(full))
        if not paired:
            continue
        out["intent_partial"][f"{am}|{I}"] = {
            "delta_U_paired": mean_ci([sm[s]["metrics"]["U_S03"]
                                       - full[s]["metrics"]["U_S03"] for s in paired]),
            "partial_mean_replicas": mean_ci(
                [sm[s]["metrics"]["components"]["mean_replicas"] for s in paired]),
            "full_mean_replicas": mean_ci(
                [full[s]["metrics"]["components"]["mean_replicas"] for s in paired]),
            "partial_U": mean_ci([sm[s]["metrics"]["U_S03"] for s in paired]),
            "full_U": mean_ci([full[s]["metrics"]["U_S03"] for s in paired]),
            "n_pairs": len(paired)}
    return out


# --------------------------------------------------------- realised volatility

def realised_I_table(rows: list[dict]) -> dict:
    """D-015: pooled disjoint 5-cycle windows across each I level's evaluation
    seeds (traces regenerated deterministically from committed seeds)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from workload import build_mmpp2, generate_arrivals
    import numpy as np
    design = json.loads((STUDY / "data" / "design.json").read_text())
    seeds_by_I = defaultdict(set)
    for r in rows:
        seeds_by_I[r["I"]].add(r["seed"])
    out = {}
    for I, seeds in sorted(seeds_by_I.items()):
        proc = build_mmpp2(design["mean_rate_rps"], I)
        win = 5.0 * proc.cycle_s
        horizon = design["horizon_s"]
        n_win_ep = int(horizon // win)
        counts = []
        for seed in sorted(seeds):
            rng = np.random.default_rng(seed)
            arrivals = generate_arrivals(proc, horizon, rng)
            edges = np.arange(n_win_ep + 1) * win
            c, _ = np.histogram(arrivals, bins=edges)
            counts.extend(c.tolist())
        counts = np.asarray(counts, dtype=float)
        entry = {
            "target_I": I, "analytic_I": proc.analytic_I,
            "window_s": win, "windows_per_episode": n_win_ep,
            "n_seeds": len(seeds), "n_windows_pooled": int(counts.size),
            "realised_I_pooled": float(counts.var(ddof=1) / counts.mean())
            if counts.size >= 2 and counts.mean() > 0 else None,
        }
        if counts.size >= 10:
            # honesty diagnostics for a heavy-tailed estimator on few windows:
            # bootstrap percentile band + the single most influential window
            rng = np.random.default_rng(20260803)
            boots = []
            for _ in range(2000):
                s = counts[rng.integers(0, counts.size, counts.size)]
                if s.mean() > 0 and s.var(ddof=1) > 0:
                    boots.append(s.var(ddof=1) / s.mean())
            boots = np.sort(np.asarray(boots))
            loo = [np.delete(counts, i) for i in range(counts.size)]
            loo_i = [x.var(ddof=1) / x.mean() for x in loo]
            entry["bootstrap_ci90"] = [float(boots[int(0.05 * len(boots))]),
                                       float(boots[int(0.95 * len(boots))])]
            entry["leave_one_out_range"] = [float(min(loo_i)), float(max(loo_i))]
            entry["max_window_count"] = float(counts.max())
        out[f"I{I:g}"] = entry
    return out


def overload_fraction_table(rows: list[dict]) -> dict:
    """Audit item (DESIGN-AUDIT Sec. 6): per I, fraction of scored 15 s windows
    whose offered rate exceeds the best measured capacity (4.70 rps) — bounds
    the controllable margin at high volatility."""
    sys.path.insert(0, str(Path(__file__).parent))
    from workload import build_mmpp2, generate_arrivals
    import numpy as np
    design = json.loads((STUDY / "data" / "design.json").read_text())
    cap = max(design["capacity_rps_k6"].values()) if "capacity_rps_k6" in design \
        else 4.70
    seeds_by_I = defaultdict(set)
    for r in rows:
        seeds_by_I[r["I"]].add(r["seed"])
    out = {"capacity_ceiling_rps": cap, "window_s": 15.0}
    for I, seeds in sorted(seeds_by_I.items()):
        proc = build_mmpp2(design["mean_rate_rps"], I)
        horizon, warm = design["horizon_s"], design["warmup_exclude_s"]
        fracs = []
        for seed in sorted(seeds):
            rng = np.random.default_rng(seed)
            arr = generate_arrivals(proc, horizon, rng)
            edges = np.arange(warm, horizon + 1e-9, 15.0)
            c, _ = np.histogram(arr, bins=edges)
            fracs.append(float((c / 15.0 > cap).mean()))
        out[f"I{I:g}"] = mean_ci(fracs)
    return out


# ------------------------------------------------------------------ RQ2 tables

def direction(replicas: int, current: int) -> str:
    return "up" if replicas > current else ("down" if replicas < current else "hold")


def rq2_table() -> dict:
    path = RES / "rq2.jsonl"
    if not path.exists():
        return {"note": "rq2.jsonl not present yet"}
    states = {s["state_id"]: s for s in
              json.loads((RES / "rq2_states.json").read_text())["states"]}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["config"], r["state_id"])].append(r)
    per_state = defaultdict(dict)
    for (config, sid), rs in by.items():
        cur = states[sid]["obs"]["replicas"]
        api_errs = sum(1 for r in rs if r.get("api_error"))
        rs = [r for r in rs if not r.get("api_error")]
        if not rs:
            continue
        raws = [r["raw"] for r in rs]
        parsed = [r["parsed"] for r in rs if r["parsed"]]
        pf = sum(1 for r in rs if not r["parsed"])
        byte_modal = max((raws.count(x) for x in set(raws)), default=0)
        acts = [(p["replicas"], p["variant"]) for p in parsed]
        act_modal = max((acts.count(a) for a in set(acts)), default=0)
        dirs = [direction(p["replicas"], cur) for p in parsed]
        probs = [dirs.count(d) / len(dirs) for d in ("up", "hold", "down")] \
            if dirs else []
        ent = -sum(p * math.log(p) for p in probs if p > 0) / math.log(3) \
            if dirs else None
        reps = [p["replicas"] for p in parsed]
        per_state[config][sid] = {
            "n": len(rs), "api_errors": api_errs, "parse_failures": pf,
            "byte_identity_rate": byte_modal / len(rs) if rs else None,
            "byte_all_identical": byte_modal == len(rs),
            "decision_identity_rate": act_modal / len(parsed) if parsed else None,
            "direction_entropy_norm": ent,
            "delta_replicas_spread": (max(reps) - min(reps)) if reps else None,
            "self_contradiction": ("up" in dirs and "down" in dirs),
            "stratum": states[sid]["stratum"],
        }
    summary = {}
    for config, sd in sorted(per_state.items()):
        vals = list(sd.values())
        def agg(key):
            xs = [v[key] for v in vals if v[key] is not None]
            return mean_ci([float(x) for x in xs])
        summary[config] = {
            "n_states": len(vals),
            "repeats_per_state": vals[0]["n"] if vals else 0,
            "byte_identity_rate": agg("byte_identity_rate"),
            "states_fully_byte_identical": sum(v["byte_all_identical"] for v in vals),
            "decision_identity_rate": agg("decision_identity_rate"),
            "direction_entropy_norm": agg("direction_entropy_norm"),
            "delta_replicas_spread": agg("delta_replicas_spread"),
            "self_contradiction_states": sum(v["self_contradiction"] for v in vals),
            "parse_failures_total": sum(v["parse_failures"] for v in vals),
            "by_stratum": {
                st: {"decision_identity_rate": mean_ci(
                        [v["decision_identity_rate"] for v in vals
                         if v["stratum"] == st and v["decision_identity_rate"] is not None]),
                     "self_contradiction_states": sum(
                         v["self_contradiction"] for v in vals if v["stratum"] == st)}
                for st in ("near", "clear-overload", "clear-calm")},
        }
    return {"per_config_summary": summary, "per_state": per_state}


# ------------------------------------------------------------------ failure modes

def failure_table(rows: list[dict]) -> dict:
    out = defaultdict(lambda: {"episodes": 0, "invocations": 0, "parse_failures": 0,
                               "repaired": 0, "decision_usd": 0.0})
    for r in rows:
        if r["model"] is None:
            continue
        c = r["metrics"]["controller"]
        k = f"{r['block']}|{r['model']}"
        out[k]["episodes"] += 1
        out[k]["invocations"] += c["invocations"]
        out[k]["parse_failures"] += c["parse_failures"]
        out[k]["repaired"] += c["repaired_parses"]
        out[k]["decision_usd"] += c["usd"]
    for k, v in out.items():
        v["parse_failure_rate"] = v["parse_failures"] / v["invocations"] \
            if v["invocations"] else None
    return dict(out)


def attempt_failure_table() -> dict:
    """Attempt-level failure modes from the decision sidecars (the episode
    metrics only see the decision level): empty completions (token budget
    consumed by reasoning), API errors by message, retry depth. Reported per
    model per block — failure modes are results (PROTOCOL Sec. 8/12)."""
    import gzip
    out = defaultdict(lambda: {"episodes": 0, "attempts": 0, "empty_raw": 0,
                               "api_errors": defaultdict(int),
                               "decisions": 0, "retried_decisions": 0,
                               "max_attempt_seen": 0})
    for path in sorted((RES / "decisions").glob("*.json.gz")):
        block = path.name.split("_")[0]
        try:
            with gzip.open(path, "rt") as f:
                entries = json.load(f)
        except Exception:
            continue
        mid = next((e["meta"].get("model_id") for e in entries
                    if e.get("meta") and e["meta"].get("model_id")), "unknown")
        k = f"{block}|{mid}"
        out[k]["episodes"] += 1
        seen_t = defaultdict(int)
        for e in entries:
            out[k]["attempts"] += 1
            seen_t[e["t"]] += 1
            out[k]["max_attempt_seen"] = max(out[k]["max_attempt_seen"],
                                             e.get("attempt", 0))
            if "error" in e:
                out[k]["api_errors"][e["error"][:60]] += 1
            elif not e.get("raw", "").strip():
                out[k]["empty_raw"] += 1
        out[k]["decisions"] += len(seen_t)
        out[k]["retried_decisions"] += sum(1 for n in seen_t.values() if n > 1)
    final = {}
    for k, v in out.items():
        v["api_errors"] = dict(v["api_errors"])
        v["empty_raw_rate_of_attempts"] = v["empty_raw"] / v["attempts"] \
            if v["attempts"] else None
        v["retry_rate_of_decisions"] = v["retried_decisions"] / v["decisions"] \
            if v["decisions"] else None
        final[k] = dict(v)
    return final


# ------------------------------------------------------------------ drivers

def cmd_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_sweep()
    write = lambda name, obj: write_json(OUT / name, obj)
    write("cells.json", cell_table(rows))
    write("margins.json", envelope_and_margins(rows))
    write("blocks_EH.json", blocks_EH_tables(rows))
    write("realised_I.json", realised_I_table(rows))
    write("overload_fraction.json", overload_fraction_table(rows))
    write("rq2_summary.json", rq2_table())
    write("failure_modes.json", failure_table(rows))
    write("failure_modes_attempts.json", attempt_failure_table())
    counts = defaultdict(int)
    for r in rows:
        counts[r["block"]] += 1
    print("rows per block:", dict(sorted(counts.items())))
    print("tables written ->", OUT)


def cmd_selfcheck() -> None:
    s = mean_ci([1.0, 2.0, 3.0])
    assert abs(s["mean"] - 2.0) < 1e-12 and s["n"] == 3
    assert abs((s["ci95"][1] - 2.0) - 4.303 * 1.0 / math.sqrt(3)) < 1e-9
    assert direction(3, 2) == "up" and direction(1, 2) == "down" \
        and direction(2, 2) == "hold"
    # paired-margin arithmetic on a synthetic mini-sweep
    mini = []
    for fam, us in [("static", [-0.3, -0.3]), ("heuristic", [-0.2, -0.4]),
                    ("queue_aware", [-0.5, -0.5])]:
        for seed, u in enumerate(us, 1):
            mini.append({"block": "B", "arm": fam, "model": None, "I": 1.0,
                         "cadence_s": 60, "seed": seed,
                         "metrics": {"U_S03": u}})
    for seed, u in enumerate([-0.1, -0.2], 1):
        mini.append({"block": "C", "arm": "llm_full", "model": "m", "I": 1.0,
                     "cadence_s": 60, "seed": seed, "metrics": {"U_S03": u}})
    m = envelope_and_margins(mini)
    cell = m["llm_full|m|I1|c60"]
    # heuristic and static tie at mean -0.30; max() keeps first-seen (static)
    assert cell["envelope_family"] in ("static", "heuristic")
    assert abs(cell["delta_U_paired"]["mean"] - 0.15) < 1e-12 or \
        abs(cell["delta_U_paired"]["mean"] - (( -0.1 - -0.2) + (-0.2 - -0.4)) / 2) < 1e-12
    print("analyze offline self-checks PASS (CI, direction, paired margins)")


def main() -> None:
    global RES, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="tables",
                        choices=["tables", "selfcheck"])
    parser.add_argument("--input-results", type=Path, default=RES,
                        help="authoritative structured result directory")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="destination for regenerated analysis tables")
    args = parser.parse_args()
    RES = args.input_results.resolve()
    OUT = args.output_dir.resolve()
    if args.cmd == "tables":
        cmd_tables()
    elif args.cmd == "selfcheck":
        cmd_selfcheck()


if __name__ == "__main__":
    main()
