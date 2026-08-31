"""Rule-based controllers (PROTOCOL Sec. 5). Three structural families (R5):

1. StaticController — fixed configuration; cost ceiling reference.
2. HeuristicController — production-style multi-knob rule: HPA-shaped replica
   scaling on arrival-rate utilisation + latency-headroom variant laddering.
   EVERY temporal parameter is in SECONDS and cooldowns compare virtual-clock
   timestamps — the S01 v1 units trap (steps vs seconds) is structurally
   impossible here because the controller never sees a step counter.
3. QueueAwareController — structurally different family (R5): scales on backlog
   drain time, not window rate; exists to kill a fake LLM win. A deep queue
   demands capacity even when the instantaneous arrival rate looks calm, and it
   holds capacity until the backlog is actually drained.

All controllers implement the same interface and see the same Observation —
observation parity is mandatory. Tunable parameters are dataclass fields; the
Optuna tuner (fairness.py) optimises them per cadence on held-out calibration
traces against the same U_S03 the LLM arm is scored on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from plant import Action, Observation, VARIANT_ORDER


@dataclass
class StaticController:
    """Fixed provisioning at the peak-adequate configuration."""
    replicas: int = 6
    variant: str = "full-3B"
    name: str = "static"

    def decide(self, obs: Observation, history) -> Action:
        return Action(replicas=self.replicas, variant=self.variant)


@dataclass
class HeuristicController:
    """HPA-shaped: desired = ceil(replicas * load / target_util), clamped, with
    hysteresis on scale-down and latency-headroom variant selection.

    Load proxy is arrival_rate / (replicas * variant_capacity_rps): the classical
    queue-blind formulation (deliberately — this family's blindness to backlog is
    a *measured property*, per R5's rationale, not an accident).
    """
    target_util: float = 0.7
    scale_down_headroom: float = 0.4       # only shrink when load proxy below this
    cooldown_s: float = 120.0              # SECONDS between scale-downs (unit-audited)
    degrade_p95_ratio: float = 1.0         # degrade variant when p95 > ratio * SLO
    upgrade_p95_ratio: float = 0.5         # upgrade when p95 < ratio * SLO persistently
    upgrade_persist_windows: int = 2
    slo_ms: float = 2000.0
    capacity_rps: dict = field(default_factory=dict)   # per-variant measured c=1 rps
    max_replicas: int = 6
    name: str = "heuristic"
    _last_scale_down_t: float = field(default=-1e9, repr=False)
    _headroom_streak: int = field(default=0, repr=False)

    def decide(self, obs: Observation, history) -> Action:
        cap = self.capacity_rps.get(obs.variant, 1.0)
        load = obs.arrival_rate / max(1e-9, obs.replicas * cap)
        desired = obs.replicas
        if load > self.target_util:
            desired = min(self.max_replicas,
                          max(obs.replicas + 1,
                              int(-(-obs.replicas * load // self.target_util))))
        elif load < self.scale_down_headroom and \
                obs.t - self._last_scale_down_t >= self.cooldown_s:
            desired = max(1, obs.replicas - 1)
            self._last_scale_down_t = obs.t
        variant = obs.variant
        vi = VARIANT_ORDER.index(variant)
        if obs.p95_ms is not None and obs.p95_ms > self.degrade_p95_ratio * self.slo_ms:
            if vi > 0:
                variant = VARIANT_ORDER[vi - 1]
            self._headroom_streak = 0
        elif obs.p95_ms is not None and obs.p95_ms < self.upgrade_p95_ratio * self.slo_ms:
            self._headroom_streak += 1
            if self._headroom_streak >= self.upgrade_persist_windows and \
                    vi < len(VARIANT_ORDER) - 1:
                variant = VARIANT_ORDER[vi + 1]
                self._headroom_streak = 0
        else:
            self._headroom_streak = 0
        return Action(replicas=desired, variant=variant)


@dataclass
class QueueAwareController:
    """Backlog-drain scaling (structurally different family, R5).

    Capacity demand = arrivals to absorb + backlog to drain within drain_target_s:
        needed_rps = arrival_rate + queue_len / drain_target_s
        desired    = ceil(needed_rps / (capacity_rps * target_util))
    Holds capacity while a backlog exists (drain logic); degrades the variant when
    the backlog exceeds what the drain target can absorb at max replicas; upgrades
    only when queue is empty and latency has headroom.
    """
    drain_target_s: float = 60.0           # SECONDS to clear current backlog
    target_util: float = 0.8
    hold_queue_len: int = 2                # never scale down above this backlog
    cooldown_s: float = 120.0              # SECONDS between scale-downs
    upgrade_p95_ratio: float = 0.5
    slo_ms: float = 2000.0
    capacity_rps: dict = field(default_factory=dict)
    max_replicas: int = 6
    name: str = "queue_aware"
    _last_scale_down_t: float = field(default=-1e9, repr=False)

    def decide(self, obs: Observation, history) -> Action:
        cap = self.capacity_rps.get(obs.variant, 1.0)
        needed_rps = obs.arrival_rate + obs.queue_len / self.drain_target_s
        desired = max(1, min(self.max_replicas,
                             int(-(-needed_rps // (cap * self.target_util)))))
        if desired < obs.replicas:
            if obs.queue_len > self.hold_queue_len or \
                    obs.t - self._last_scale_down_t < self.cooldown_s:
                desired = obs.replicas
            else:
                desired = obs.replicas - 1     # gradual shrink
                self._last_scale_down_t = obs.t
        variant = obs.variant
        vi = VARIANT_ORDER.index(variant)
        # backlog beyond max-capacity drain -> shed quality for speed
        max_drain_rps = self.max_replicas * self.capacity_rps.get(VARIANT_ORDER[0], cap)
        if desired >= self.max_replicas and \
                needed_rps > self.max_replicas * cap and vi > 0:
            variant = VARIANT_ORDER[vi - 1]
        elif obs.queue_len == 0 and obs.p95_ms is not None and \
                obs.p95_ms < self.upgrade_p95_ratio * self.slo_ms and \
                vi < len(VARIANT_ORDER) - 1:
            variant = VARIANT_ORDER[vi + 1]
        return Action(replicas=desired, variant=variant)


if __name__ == "__main__":
    # Self-check: the two heuristic families must diverge on the scenario that
    # motivated R5 — calm arrival rate over a deep backlog. The queue-blind HPA
    # scales down; the queue-aware controller must not.
    caps = {"tiny-1B4": 2.0, "lite-1B": 1.7, "full-3B": 0.74}
    calm_deep_backlog = Observation(
        t=1000.0, window_s=60.0, arrival_rate=0.2, prev_arrival_rate=3.0,
        queue_len=80, in_flight=4, p50_ms=5000.0, p95_ms=9000.0,
        replicas=4, variant="full-3B", utilization=1.0,
        completed_in_window=10, dropped_in_window=5)
    h = HeuristicController(capacity_rps=caps, slo_ms=2000.0, cooldown_s=0.0)
    q = QueueAwareController(capacity_rps=caps, slo_ms=2000.0)
    ah = h.decide(calm_deep_backlog, [])
    aq = q.decide(calm_deep_backlog, [])
    assert ah.replicas < 4, f"HPA should scale down over backlog (got {ah.replicas})"
    assert aq.replicas >= 4, f"queue-aware must hold/grow over backlog (got {aq.replicas})"
    # Burst: both must scale up
    burst = Observation(t=500.0, window_s=60.0, arrival_rate=4.0,
                        prev_arrival_rate=0.5, queue_len=10, in_flight=2,
                        p50_ms=800.0, p95_ms=1800.0, replicas=2, variant="full-3B",
                        utilization=0.95, completed_in_window=40, dropped_in_window=0)
    h2 = HeuristicController(capacity_rps=caps, slo_ms=2000.0)
    q2 = QueueAwareController(capacity_rps=caps, slo_ms=2000.0)
    assert h2.decide(burst, []).replicas > 2
    assert q2.decide(burst, []).replicas > 2
    # Static is static
    s = StaticController()
    assert s.decide(burst, []).replicas == 6
    print("controllers structural-divergence self-checks PASS")
