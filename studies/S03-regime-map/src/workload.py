"""Bursty arrival-process generator with a controllable index of dispersion.

ADAPTED FOR S03 from S01 (`studies/S01-llm-control-cadence/src/workload.py`), per
S03 DECISIONS.md D-005: the one S01 module carried over (it survived the D-045 audit).
Adaptation terms: re-verified by its own self-check in this study before first use
(`python3 workload.py` asserts and writes `results/selfcheck_workload.json`).

Implements a two-state Markov-Modulated Poisson Process (MMPP(2), a MAP special case)
fitted to a target index of dispersion I, following the burstiness characterisation of

    N. Mi, G. Casale, L. Cherkasova, E. Smirni, "Injecting realistic burstiness to a
    traditional client-server benchmark", ICAC 2009 / JISA 1(2):117-134, 2010.

Their definition (JISA 2010, Sec. 2, p.120) is over inter-arrival intervals:
    I = SCV * (1 + 2 * sum_k rho_k)
We adopt I as the single volatility knob, as they do; the MMPP(2) moment-matching is ours.

MEASUREMENT NOTE (inherited S01 lesson, D-026 there). Verification uses the index of
dispersion for *counts*, IDC(T) = Var(N(T))/E[N(T)] over disjoint windows: the
interval-based estimator truncates autocorrelation sums and silently reports I ~= 1 for
exactly the strongly-bursty processes under study.

S03 PROTOCOL.md Sec. 4. Anchors from the source: I=1 Poisson, ~400 mild, ~4000 severe.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from artifact_io import write_json

STUDY = Path(__file__).resolve().parents[1]

# Process shape held fixed across all volatility levels so that I is the only thing
# that varies. Burst timescale sits inside the cadence range under study (15-300 s).
DEFAULT_DUTY = 0.06          # fraction of time in the high-rate state
DEFAULT_BURST_S = 20.0       # mean sojourn in the high-rate state


@dataclass(frozen=True)
class MMPP2:
    """Two-state MMPP. State i emits Poisson arrivals at rate lam_i; sojourn in state i
    is exponential with rate q_i (mean sojourn 1/q_i)."""

    lam_hi: float
    lam_lo: float
    q_hi: float
    q_lo: float

    @property
    def stationary(self) -> tuple[float, float]:
        p_hi = self.q_lo / (self.q_hi + self.q_lo)
        return p_hi, 1.0 - p_hi

    @property
    def mean_rate(self) -> float:
        p_hi, p_lo = self.stationary
        return p_hi * self.lam_hi + p_lo * self.lam_lo

    @property
    def cycle_s(self) -> float:
        return 1.0 / self.q_hi + 1.0 / self.q_lo

    @property
    def analytic_I(self) -> float:
        p_hi, p_lo = self.stationary
        d = self.lam_hi - self.lam_lo
        if d == 0:
            return 1.0
        return 1.0 + 2.0 * p_hi * p_lo * d * d / (self.mean_rate * (self.q_hi + self.q_lo))


def build_mmpp2(mean_rate: float, target_I: float, *, duty: float = DEFAULT_DUTY,
                burst_s: float = DEFAULT_BURST_S) -> MMPP2:
    """Closed-form moment match: fix duty cycle and burst length, solve for rate contrast.

        I = 1 + 2 * p_hi * p_lo * (lam_hi - lam_lo)^2 / (mean * (q_hi + q_lo))
        d = sqrt( (I - 1) * mean * (q_hi + q_lo) / (2 * p_hi * p_lo) )

    Solving for the contrast rather than the switching timescale keeps burst duration on
    a fixed, control-relevant timescale across volatility levels, so cadence and
    volatility stay independent variables rather than being confounded.
    """
    if target_I <= 1.0 + 1e-9:
        return MMPP2(lam_hi=mean_rate, lam_lo=mean_rate, q_hi=1.0 / burst_s,
                     q_lo=1.0 / burst_s)

    p_hi, p_lo = duty, 1.0 - duty
    q_hi = 1.0 / burst_s
    q_lo = q_hi * p_hi / p_lo
    d = math.sqrt((target_I - 1.0) * mean_rate * (q_hi + q_lo) / (2.0 * p_hi * p_lo))
    lam_lo = mean_rate - p_hi * d
    if lam_lo < 0.0:
        raise ValueError(
            f"target_I={target_I} unreachable at mean_rate={mean_rate}, duty={duty}, "
            f"burst_s={burst_s}: would require a negative low-state rate. "
            "Lower the duty cycle or lengthen bursts."
        )
    return MMPP2(lam_hi=lam_lo + d, lam_lo=lam_lo, q_hi=q_hi, q_lo=q_lo)


def generate_arrivals(proc: MMPP2, horizon_s: float, rng: np.random.Generator) -> np.ndarray:
    """Sorted arrival timestamps in [0, horizon_s)."""
    times: list[np.ndarray] = []
    t = 0.0
    p_hi, _ = proc.stationary
    in_hi = rng.random() < p_hi
    while t < horizon_s:
        lam = proc.lam_hi if in_hi else proc.lam_lo
        q = proc.q_hi if in_hi else proc.q_lo
        seg_end = min(t + rng.exponential(1.0 / q), horizon_s)
        if lam > 0:
            n = rng.poisson(lam * (seg_end - t))
            if n:
                times.append(rng.uniform(t, seg_end, size=n))
        t = seg_end
        in_hi = not in_hi
    if not times:
        return np.empty(0, dtype=float)
    out = np.sort(np.concatenate(times))
    return out[out < horizon_s]


def index_of_dispersion_counts(arrivals: np.ndarray, horizon_s: float,
                               window_s: float) -> float:
    """IDC(T) = Var(N(T)) / E[N(T)] over disjoint windows of length T."""
    n_win = int(horizon_s // window_s)
    if n_win < 30:
        return float("nan")
    edges = np.arange(n_win + 1) * window_s
    counts, _ = np.histogram(arrivals, bins=edges)
    m = counts.mean()
    if m <= 0:
        return float("nan")
    return float(counts.var(ddof=1) / m)


def measure_index_of_dispersion(arrivals: np.ndarray, horizon_s: float,
                                cycle_s: float) -> float:
    """Empirical I as IDC at T = 5 modulation cycles (>= 30 windows required)."""
    return index_of_dispersion_counts(arrivals, horizon_s, window_s=5.0 * cycle_s)


def make_trace(mean_rate: float, target_I: float, horizon_s: float, seed: int
               ) -> tuple[np.ndarray, dict]:
    """One arrival trace plus a provenance record carrying the realised I, so every
    reported cell states what was actually generated, not merely what was requested."""
    rng = np.random.default_rng(seed)
    proc = build_mmpp2(mean_rate, target_I)
    arrivals = generate_arrivals(proc, horizon_s, rng)
    meta = {
        "seed": seed,
        "target_I": target_I,
        "analytic_I": proc.analytic_I,
        "realised_I": measure_index_of_dispersion(arrivals, horizon_s, proc.cycle_s),
        "mean_rate_nominal": mean_rate,
        "mean_rate_realised": float(arrivals.size / horizon_s) if horizon_s > 0 else 0.0,
        "n_arrivals": int(arrivals.size),
        "horizon_s": horizon_s,
        "cycle_s": proc.cycle_s,
        "mmpp2": asdict(proc),
    }
    return arrivals, meta


def self_check(write_artifact: bool = True) -> dict:
    """PROTOCOL Sec. 4 gate: the fitted process must hit the target I within 1%
    (closed-form moment match => analytic_I), and the realised IDC over a long horizon
    must agree with the analytic value within a stated statistical tolerance (25%,
    averaged over 5 seeds at 200,000 s — the count-variance estimator's own noise floor
    at I=4000 with this horizon). Raises AssertionError on failure; commits the record.
    """
    HZ = 200_000.0
    MEAN = 25.0
    rows = []
    for target in (1.0, 40.0, 400.0, 4000.0):
        proc = build_mmpp2(MEAN, target)
        fit_err = abs(proc.analytic_I - target) / target
        assert fit_err < 0.01, f"I={target}: analytic fit off by {fit_err:.3%} (>1%)"
        realised, rates = [], []
        for seed in range(5):
            _, m = make_trace(MEAN, target, HZ, seed)
            realised.append(m["realised_I"])
            rates.append(m["mean_rate_realised"])
        r_mean = float(np.mean(realised))
        stat_err = abs(r_mean - proc.analytic_I) / proc.analytic_I
        assert stat_err < 0.25, (
            f"I={target}: realised IDC {r_mean:.1f} vs analytic {proc.analytic_I:.1f} "
            f"({stat_err:.1%} > 25%)")
        rate_err = abs(float(np.mean(rates)) - MEAN) / MEAN
        assert rate_err < 0.02, f"I={target}: mean rate off by {rate_err:.2%}"
        rows.append({
            "target_I": target, "analytic_I": proc.analytic_I, "fit_rel_err": fit_err,
            "realised_I_mean5": r_mean, "realised_rel_err": stat_err,
            "mean_rate_realised": float(np.mean(rates)),
        })
    record = {"module": "workload.py", "adapted_from": "S01 src/workload.py",
              "horizon_s": HZ, "seeds": 5, "criteria": {
                  "analytic_fit": "<1% (PROTOCOL Sec.4)",
                  "realised_idc": "<25% of analytic (statistical)",
                  "mean_rate": "<2%"},
              "rows": rows, "pass": True}
    if write_artifact:
        out = STUDY / "results" / "selfcheck_workload.json"
        out.parent.mkdir(exist_ok=True)
        write_json(out, record)
        print(f"wrote {out}")
    return record


if __name__ == "__main__":
    rec = self_check()
    print(f"{'target I':>9} {'analytic':>10} {'realised':>10} {'fit err':>9} {'stat err':>9}")
    for r in rec["rows"]:
        print(f"{r['target_I']:>9.0f} {r['analytic_I']:>10.1f} {r['realised_I_mean5']:>10.1f} "
              f"{r['fit_rel_err']:>9.3%} {r['realised_rel_err']:>9.2%}")
    print("SELF-CHECK PASS")
