"""Layer-B plant: discrete-event, virtual-time edge inference tier, calibrated
from committed Layer-A measurements (PROTOCOL Sec. 2).

Semantics (D-010, from measured Layer-A data — R8): a "replica" is one parallel
serving SLOT of a continuous-batching llama-server on the single edge node.
Measured fact (layer_a campaigns, 2026-08-02): independent processes on the shared
GPU add ZERO aggregate throughput, while slots scale sublinearly (e.g. full-3B
0.74 -> 1.60 rps from 1 -> 6 slots) with rising per-request latency. The plant
therefore samples each dispatched request's service time from the measured
distribution AT THE CONCURRENCY LEVEL in effect at dispatch (nearest measured
level of {1,2,4,6}); block F's real replay validates this state-dependent
approximation and any disagreement is reported.

R1 is enforced here structurally: episode metrics are computed over OFFERED work.
Dropped requests and end-of-episode residual queue are violations, not invisible.
Both U_S03 (primary) and U_v2 (comparability) are always computed (PROTOCOL Sec. 6).

The controller is never simulated: `Controller.decide()` implementations perform
real model invocations; their measured wall-clock latency is injected into the
virtual clock (action applies at wake + latency).
"""
from __future__ import annotations

import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

STUDY = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- calibration

VARIANT_ORDER = ["tiny-1B4", "lite-1B", "full-3B"]  # quality ascending


@dataclass(frozen=True)
class VariantCal:
    name: str
    service_levels: dict                   # {concurrency k: tuple(measured latencies s)}
    switch_load_s: float                   # measured server start+load time
    energy_j_per_request: dict             # {k: idle-subtracted GPU_SOC J/request}
    accuracy: float                        # measured task accuracy (accuracy probe)

    def nearest_level(self, k: int) -> int:
        levels = sorted(self.service_levels)
        below = [l for l in levels if l <= k]
        return below[-1] if below else levels[0]


@dataclass(frozen=True)
class Calibration:
    variants: dict[str, VariantCal]
    mode: str                              # "nodes" | "slots" — D-010
    source: str                            # provenance note

    @classmethod
    def from_layer_a(cls, mode: str, accuracy: dict[str, float],
                     switch_load_s: dict[str, float]) -> "Calibration":
        assert mode == "slots", "D-010: slots is the calibrated mode"
        variants = {}
        for v in VARIANT_ORDER:
            levels, energy = {}, {}
            for k in (1, 2, 4, 6):
                path = STUDY / "results" / "layer_a" / f"slots_{v}_r{k}_c{k}.json"
                cell = json.loads(path.read_text())
                lats = tuple(x["latency_s"] for x in cell["raw_requests"]
                             if "error" not in x and not x["warmup"])
                if len(lats) < 30:
                    raise ValueError(f"{path}: only {len(lats)} good samples")
                levels[k] = lats
                energy[k] = cell["energy"]["gpu_soc_j_per_request"]
            variants[v] = VariantCal(
                name=v, service_levels=levels, switch_load_s=switch_load_s[v],
                energy_j_per_request=energy, accuracy=accuracy[v])
        return cls(variants=variants, mode=mode,
                   source="layer_a slots_{variant}_r{k}_c{k} cells, k in {1,2,4,6}")


# ---------------------------------------------------------------- data model

@dataclass
class Request:
    t_arrival: float
    t_start: float | None = None
    t_done: float | None = None
    dropped: bool = False

    @property
    def latency(self) -> float | None:
        return None if self.t_done is None else self.t_done - self.t_arrival


@dataclass
class Observation:
    """What every controller sees — observation parity is mandatory (PROTOCOL Sec. 5)."""
    t: float
    window_s: float
    arrival_rate: float          # req/s in the last window
    prev_arrival_rate: float
    queue_len: int
    in_flight: int
    p50_ms: float | None         # completed-in-window latencies
    p95_ms: float | None
    replicas: int
    variant: str
    utilization: float           # busy-replica fraction over the window
    completed_in_window: int
    dropped_in_window: int


@dataclass
class Action:
    replicas: int
    variant: str
    next_wake_s: float | None = None      # self-triggered arm only
    raw: str = ""                          # controller's verbatim output (evidence)
    latency_s: float = 0.0                 # measured deliberation wall-clock
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    parse_failed: bool = False
    repaired: bool = False


class Controller(Protocol):
    name: str

    def decide(self, obs: Observation, history: list[tuple[Observation, Action]]) -> Action: ...


# ---------------------------------------------------------------- the plant

@dataclass
class EpisodeConfig:
    horizon_s: float = 1800.0
    warmup_exclude_s: float = 300.0        # R6 accounting window
    slo_ms: float = 0.0                    # set from calibration (Sec. 7 rule)
    abandon_s: float = 0.0                 # queue abandonment = 10x SLO (drop rule)
    max_replicas: int = 6
    cadence_s: float = 60.0
    w_slo: float = 0.5
    w_acc: float = 0.2
    w_cost: float = 0.3
    # Block E (R7): at intent_switch_at_s the objective genuinely changes —
    # requests arriving after the switch are scored against the secondary SLO,
    # and the weight vector is time-averaged across the scored span.
    intent_switch_at_s: float | None = None
    secondary_slo_ms: float = 2000.0
    secondary_w: tuple = (0.7, 0.1, 0.2)


class Plant:
    """Virtual-time discrete-event simulator. Replicas = independent single-slot
    servers; service times resampled from the calibrated empirical distribution
    (per variant). Variant switches make the tier unavailable for the measured
    load time of the incoming variant (all replicas reload)."""

    def __init__(self, cal: Calibration, cfg: EpisodeConfig, plant_seed: int):
        self.cal = cal
        self.cfg = cfg
        self.rng = random.Random(plant_seed)

    def _sample_service(self, variant: str, k: int) -> float:
        vc = self.cal.variants[variant]
        return self.rng.choice(vc.service_levels[vc.nearest_level(max(1, k))])

    def run(self, arrivals: list[float], controller: Controller,
            init_replicas: int, init_variant: str) -> dict:
        cfg = self.cfg
        EV_ARRIVAL, EV_DONE, EV_WAKE, EV_ABANDON = 0, 1, 2, 3
        events: list[tuple[float, int, int, object]] = []
        seq = 0

        def push(t: float, kind: int, payload: object = None) -> None:
            nonlocal seq
            heapq.heappush(events, (t, kind, seq, payload))
            seq += 1

        for t in arrivals:
            push(t, EV_ARRIVAL)
        push(cfg.cadence_s, EV_WAKE)

        replicas = init_replicas
        variant = init_variant
        switch_until = 0.0                 # tier unavailable during variant load
        queue: list[Request] = []
        in_service: dict[int, Request] = {}  # replica idx -> request
        free: list[int] = list(range(replicas))
        all_requests: list[Request] = []
        completed: list[Request] = []
        window_completed: list[Request] = []
        window_dropped = 0
        window_arrivals = 0
        prev_rate = 0.0
        busy_time = 0.0
        last_t = 0.0
        history: list[tuple[Observation, Action]] = []
        decisions: list[Action] = []
        replica_seconds = 0.0
        variant_acc_seconds = 0.0
        switches = 0
        intent_switched = False

        def try_dispatch(now: float) -> None:
            while queue and free and now >= switch_until:
                idx = free.pop()
                req = queue.pop(0)
                req.t_start = now
                svc = self._sample_service(variant, len(in_service) + 1)
                req.t_done = now + svc
                in_service[idx] = req
                push(req.t_done, EV_DONE, idx)

        while events:
            t, kind, _, payload = heapq.heappop(events)
            if t > cfg.horizon_s and kind != EV_DONE:
                continue
            # integrate cost terms
            dt = t - last_t
            if dt > 0 and last_t >= cfg.warmup_exclude_s:
                replica_seconds += replicas * dt
                variant_acc_seconds += self.cal.variants[variant].accuracy * dt
                busy_time += len(in_service) * dt
            last_t = t

            if kind == EV_ARRIVAL:
                req = Request(t_arrival=t)
                all_requests.append(req)
                window_arrivals += 1
                queue.append(req)
                push(t + cfg.abandon_s, EV_ABANDON, req)
                try_dispatch(t)

            elif kind == EV_DONE:
                idx = payload
                req = in_service.pop(idx, None)
                if req is not None:
                    completed.append(req)
                    window_completed.append(req)
                    if idx < replicas:
                        free.append(idx)
                    try_dispatch(t)

            elif kind == EV_ABANDON:
                req = payload
                if req.t_start is None and not req.dropped:
                    req.dropped = True
                    if req in queue:
                        queue.remove(req)
                    window_dropped += 1

            elif kind == EV_WAKE:
                if t >= cfg.horizon_s:
                    continue
                rate = window_arrivals / cfg.cadence_s
                lats = sorted((r.latency for r in window_completed), key=float)
                obs = Observation(
                    t=t, window_s=cfg.cadence_s, arrival_rate=rate,
                    prev_arrival_rate=prev_rate, queue_len=len(queue),
                    in_flight=len(in_service),
                    p50_ms=(lats[len(lats) // 2] * 1000 if lats else None),
                    p95_ms=(lats[min(len(lats) - 1, int(0.95 * len(lats)))] * 1000
                            if lats else None),
                    replicas=replicas, variant=variant,
                    utilization=min(1.0, busy_time / max(1e-9, replicas * cfg.cadence_s))
                    if t >= cfg.warmup_exclude_s else 0.0,
                    completed_in_window=len(window_completed),
                    dropped_in_window=window_dropped)
                act = controller.decide(obs, history)
                decisions.append(act)
                history.append((obs, act))
                apply_t = t + act.latency_s      # controller cost is real time
                new_r = max(1, min(cfg.max_replicas, act.replicas))
                new_v = act.variant if act.variant in self.cal.variants else variant
                if new_v != variant:
                    switches += 1
                    switch_until = apply_t + self.cal.variants[new_v].switch_load_s
                    # in-service requests finish on the old variant; new dispatches wait
                variant_next = new_v
                # apply at apply_t via a zero-length wake? simplest: apply now at apply_t
                # replicas change: grow adds free slots; shrink retires idle slots first
                def apply_action(now: float, new_r=new_r, new_v=variant_next) -> None:
                    nonlocal replicas, variant, free
                    if new_r > replicas:
                        free.extend(range(replicas, new_r))
                    elif new_r < replicas:
                        free = [i for i in free if i < new_r]
                    replicas = new_r
                    variant = new_v
                    try_dispatch(now)
                if act.latency_s <= 0:
                    apply_action(t)
                else:
                    push(apply_t, EV_WAKE + 100, apply_action)
                prev_rate = rate
                window_completed, window_dropped, window_arrivals = [], 0, 0
                busy_time = 0.0
                next_wake = (act.next_wake_s if act.next_wake_s
                             else cfg.cadence_s)
                push(t + max(1.0, next_wake), EV_WAKE)

            elif kind == EV_WAKE + 100:
                payload(t)

        return self._metrics(all_requests, completed, queue, decisions, switches,
                             replica_seconds, variant_acc_seconds)

    # -------------------------------------------------------------- metrics

    def _metrics(self, all_requests, completed, residual_queue, decisions,
                 switches, replica_seconds, variant_acc_seconds) -> dict:
        cfg = self.cfg
        scored_span = cfg.horizon_s - cfg.warmup_exclude_s
        scored = [r for r in all_requests if r.t_arrival >= cfg.warmup_exclude_s]
        offered = len(scored)
        comp = [r for r in scored if r.t_done is not None and r.t_done <= cfg.horizon_s]
        dropped = [r for r in scored if r.dropped]
        residual = [r for r in scored if not r.dropped and
                    (r.t_done is None or r.t_done > cfg.horizon_s)]
        def slo_for(req) -> float:
            if cfg.intent_switch_at_s is not None and \
                    req.t_arrival >= cfg.intent_switch_at_s:
                return cfg.secondary_slo_ms
            return cfg.slo_ms
        late = [r for r in comp if r.latency * 1000 > slo_for(r)]
        ok = [r for r in comp if r.latency * 1000 <= slo_for(r)]

        # R1 primary: violations over OFFERED work
        v_offered = (len(late) + len(dropped) + len(residual)) / offered if offered else 0.0
        # v2-style comparability: violations over completed only
        v_completed = len(late) / len(comp) if comp else 0.0

        mean_replicas = replica_seconds / scored_span
        mean_acc = variant_acc_seconds / scored_span
        acc_deficit = self.cal.variants[VARIANT_ORDER[-1]].accuracy - mean_acc
        cost_norm = mean_replicas / cfg.max_replicas

        if cfg.intent_switch_at_s is not None:
            frac2 = max(0.0, min(1.0, (cfg.horizon_s - cfg.intent_switch_at_s) /
                                 scored_span))
            w = tuple((1 - frac2) * a + frac2 * b for a, b in
                      zip((cfg.w_slo, cfg.w_acc, cfg.w_cost), cfg.secondary_w))
        else:
            w = (cfg.w_slo, cfg.w_acc, cfg.w_cost)
        u_s03 = -(w[0] * v_offered) - (w[1] * acc_deficit) - (w[2] * cost_norm)
        u_v2 = -(w[0] * v_completed) - (w[1] * acc_deficit) - (w[2] * cost_norm)

        lat_ms = sorted(r.latency * 1000 for r in comp)

        def pct(p):
            return lat_ms[min(len(lat_ms) - 1, int(p * len(lat_ms)))] if lat_ms else None

        parse_fails = sum(1 for d in decisions if d.parse_failed)
        return {
            "offered": offered, "completed": len(comp), "dropped": len(dropped),
            "residual_queue": len(residual), "late": len(late),
            "goodput": len(ok) / offered if offered else 0.0,
            "violation_rate_offered": v_offered,
            "violation_rate_completed": v_completed,
            "U_S03": u_s03, "U_v2": u_v2,
            "components": {"acc_deficit": acc_deficit, "mean_replicas": mean_replicas,
                           "cost_norm": cost_norm},
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
            "controller": {
                "invocations": len(decisions),
                "parse_failures": parse_fails,
                "repaired_parses": sum(1 for d in decisions if getattr(d, "repaired", False)),
                "wall_s_total": sum(d.latency_s for d in decisions),
                "tokens_in": sum(d.tokens_in for d in decisions),
                "tokens_out": sum(d.tokens_out for d in decisions),
                "usd": sum(d.usd for d in decisions),
                "variant_switches": switches,
            },
        }


if __name__ == "__main__":
    # Self-check with a synthetic calibration (no Layer-A dependency):
    # deterministic service, Poisson-ish arrivals; conservation law must hold:
    # offered == completed + dropped + residual, and a static controller under
    # light load must have ~zero violations.
    class Static:
        name = "static"

        def decide(self, obs, history):
            return Action(replicas=2, variant="full-3B")

    cal = Calibration(
        variants={v: VariantCal(v, {1: (0.5,)}, 5.0, {1: 1.0}, a) for v, a in
                  [("tiny-1B4", 0.7), ("lite-1B", 0.8), ("full-3B", 0.9)]},
        mode="slots", source="synthetic-selfcheck")
    cfg = EpisodeConfig(horizon_s=1200.0, warmup_exclude_s=300.0, slo_ms=2000.0,
                        abandon_s=20.0, cadence_s=60.0)
    rng = random.Random(7)
    arrivals = []
    t = 0.0
    while t < 1200.0:
        t += rng.expovariate(2.0)  # 2 req/s vs capacity 2/0.5 = 4 req/s
        arrivals.append(t)
    m = Plant(cal, cfg, plant_seed=1).run(arrivals, Static(), 2, "full-3B")
    assert m["offered"] == m["completed"] + m["dropped"] + m["residual_queue"], m
    assert m["violation_rate_offered"] < 0.05, m
    assert m["controller"]["invocations"] >= 15, m
    # Overload case: 1 replica of a 2 s service under 2 req/s must drop/violate heavily
    cal2 = Calibration(
        variants={v: VariantCal(v, {1: (2.0,)}, 5.0, {1: 1.0}, a) for v, a in
                  [("tiny-1B4", 0.7), ("lite-1B", 0.8), ("full-3B", 0.9)]},
        mode="slots", source="synthetic-selfcheck")
    m2 = Plant(cal2, cfg, plant_seed=1).run(arrivals, type("S", (), {
        "name": "static1", "decide": lambda self, o, h: Action(replicas=1, variant="full-3B")})(), 1, "full-3B")
    assert m2["offered"] == m2["completed"] + m2["dropped"] + m2["residual_queue"], m2
    assert m2["violation_rate_offered"] > 0.5, m2
    assert m2["U_S03"] < m["U_S03"], "overload must score worse"
    print("plant conservation + light/overload self-checks PASS")
    print(json.dumps({k: m[k] for k in ("offered", "completed", "dropped",
                                        "violation_rate_offered", "U_S03")}, indent=1))
