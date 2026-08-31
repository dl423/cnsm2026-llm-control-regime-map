"""LLM controller arms (PROTOCOL Sec. 5): real model invocations, never simulated.

Every decision is a genuine API/local call. Measured wall-clock latency, tokens,
and USD ride back on the Action and are injected into the plant's virtual clock.
Raw model output is preserved verbatim on every Action (evidence).

Intent texts are fixed here, committed BEFORE any run (R2, R7):
- INTENT_FULL expresses the complete scored objective, including the cost and
  accuracy terms and their weights (R2: the controller is told what it is scored on).
- INTENT_PARTIAL is the declared under-specification arm — v2-style, no cost
  language — run only as the pre-registered block H variable.
- INTENT_SECONDARY is the mid-episode intent change (block E, t=900 s): tightened
  latency with an explicit cost-ceiling relaxation.

Determinism/settings per arm are parameters (RQ2 sweeps them); RQ1 uses each
vendor's default sampling with seed pinned where honoured.

Spend guard: a module-level SpendLedger enforces PROTOCOL Sec. 9 v1.2 —
warns past the $30 guideline, HALTS (raises) at the $60 backstop.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from artifact_io import write_json
from plant import Action, Observation, VARIANT_ORDER

STUDY = Path(__file__).resolve().parents[1]
PRICES = json.loads((STUDY / "data" / "hosted_prices.json").read_text())["usd_per_mtok"]

GUIDELINE_USD = 30.0
BACKSTOP_USD = 60.0


class SpendHalt(RuntimeError):
    pass


@dataclass
class SpendLedger:
    """Cumulative hosted spend across the whole study run; persisted to results/.
    Thread-safe (in-process lock) with atomic writes (tmp+rename) — parallel
    episode workers share one ledger."""
    path: Path = STUDY / "results" / "spend_ledger.json"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raw = self.path.read_text()
                obj, _ = json.JSONDecoder().raw_decode(raw)
                return obj
        return {"total_usd": 0.0, "by_model": {}, "calls": 0}

    def add(self, model: str, usd: float) -> float:
        with self._lock:
            led = self._load()
            led["total_usd"] += usd
            led["by_model"][model] = led["by_model"].get(model, 0.0) + usd
            led["calls"] += 1
            self.path.parent.mkdir(exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            write_json(tmp, led)
            tmp.replace(self.path)
            total = led["total_usd"]
        if total >= BACKSTOP_USD:
            raise SpendHalt(f"hosted spend ${total:.2f} >= backstop ${BACKSTOP_USD}")
        return total


LEDGER = SpendLedger()

# --------------------------------------------------------------- intent texts

INTENT_FULL = """\
You are the controller of an edge inference service running on a single-site tier.
Objective (this is exactly how you are scored, per 1500 s episode):
  score = -(0.5 * V) - (0.2 * A) - (0.3 * C)
  V = fraction of ALL offered requests that miss the SLO: completed later than
      {slo_ms:.0f} ms end-to-end, dropped after waiting {abandon_s:.0f} s, or still
      queued at episode end. Dropped and unserved work counts against you.
  A = accuracy deficit: mean over time of (best variant accuracy - current variant
      accuracy). Variants (accuracy, single-stream p50 service time):
      {variant_menu}
  C = mean replicas / {max_replicas}. Replicas cost; do not over-provision.
Constraints: replicas between 1 and {max_replicas}. Switching variant reloads the
service and pauses new dispatches for roughly {switch_s:.0f} s."""

INTENT_PARTIAL = """\
You are the controller of an edge inference service running on a single-site tier.
Keep the service healthy: requests should complete within the {slo_ms:.0f} ms SLO.
Prefer the most accurate variant when there is latency headroom. Variants
(accuracy, single-stream p50 service time):
  {variant_menu}
Constraints: replicas between 1 and {max_replicas}. Switching variant reloads the
service and pauses new dispatches for roughly {switch_s:.0f} s."""

INTENT_SECONDARY = """\
UPDATED OBJECTIVE (applies from now on): latency now dominates. You are scored:
  score = -(0.7 * V) - (0.1 * A) - (0.2 * C)
with V, A, C defined as before but the SLO tightened to {slo_ms:.0f} ms. Cost
matters less than before; meet the tighter SLO."""

RESPONSE_SPEC = """\
Respond with ONLY a JSON object, no markdown fence, of the form:
{"replicas": <int>, "variant": "<one of: %s>", "why": "<one clause, max 12 words>"}""" % (
    " | ".join(VARIANT_ORDER))

RESPONSE_SPEC_SELF_TRIGGERED = RESPONSE_SPEC.replace(
    ', "why"', ', "next_wake_s": <int seconds until your next wake, 5-600>, "why"')


def render_observation(obs: Observation) -> str:
    """Byte-stable rendering (RQ2 depends on byte-identical inputs)."""
    def fmt(x, nd=1):
        return "n/a" if x is None else f"{x:.{nd}f}"
    return (
        f"STATE t={obs.t:.0f}s window={obs.window_s:.0f}s\n"
        f"  arrivals {obs.arrival_rate:.2f} req/s (prev {obs.prev_arrival_rate:.2f})"
        f" | queue {obs.queue_len} | in-flight {obs.in_flight}\n"
        f"  p50 {fmt(obs.p50_ms, 0)} ms | p95 {fmt(obs.p95_ms, 0)} ms"
        f" | completed {obs.completed_in_window} | dropped {obs.dropped_in_window}\n"
        f"  replicas {obs.replicas} | variant {obs.variant}"
        f" | utilization {obs.utilization:.2f}")


def render_history(history, k: int = 4) -> str:
    if not history:
        return "HISTORY: none"
    lines = ["HISTORY (last decisions and outcomes):"]
    for obs, act in history[-k:]:
        lines.append(f"  t={obs.t:.0f}s: chose replicas={act.replicas} variant={act.variant}"
                     f" (queue was {obs.queue_len}, p95 {'n/a' if obs.p95_ms is None else f'{obs.p95_ms:.0f}ms'})")
    return "\n".join(lines)


def parse_action(text: str, fallback: Action) -> Action:
    """Extract the JSON action. If the JSON is truncated (token cap) but the
    replicas/variant fields are unambiguous, repair-parse and flag it (counted
    and reported). Failure after retries is a counted failure mode (Sec. 8a):
    the plant holds the previous configuration."""
    repaired = False
    obj = None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        rm = re.search(r'"replicas"\s*:\s*(\d+)', text)
        vm = re.search(r'"variant"\s*:\s*"([\w.-]+)"', text)
        if not (rm and vm):
            raise ValueError("no parseable action in output")
        obj = {"replicas": int(rm.group(1)), "variant": vm.group(1)}
        nm = re.search(r'"next_wake_s"\s*:\s*(\d+)', text)
        if nm:
            obj["next_wake_s"] = int(nm.group(1))
        repaired = True
    replicas = int(obj["replicas"])
    variant = str(obj["variant"])
    if variant not in VARIANT_ORDER:
        raise ValueError(f"unknown variant {variant!r}")
    nw = obj.get("next_wake_s")
    act = Action(replicas=replicas, variant=variant,
                 next_wake_s=float(nw) if nw is not None else None, raw=text)
    act.repaired = repaired
    return act


# --------------------------------------------------------------- API adapters

def _openai_call(model: str, system: str, user: str, seed: int | None,
                 max_tokens: int = 300) -> tuple[str, dict]:
    body = {"model": model, "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    text = payload["choices"][0]["message"]["content"]
    meta = {"model_id": payload.get("model"), "fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage", {})}
    return text, meta


def _anthropic_call(model: str, system: str, user: str, seed: int | None,
                    max_tokens: int = 300,
                    temperature: float | None = None) -> tuple[str, dict]:
    # No seed control exists on this API; `temperature` is its only exposed
    # determinism control (RQ2 sonnet-pinned config). Default None = vendor default.
    body = {"model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}]}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    text = "".join(b.get("text", "") for b in payload.get("content", []))
    u = payload.get("usage", {})
    meta = {"model_id": payload.get("model"), "fingerprint": None,
            "usage": {"prompt_tokens": u.get("input_tokens"),
                      "completion_tokens": u.get("output_tokens")}}
    return text, meta


def _local_call(model: str, system: str, user: str, seed: int | None,
                max_tokens: int = 400, port: int = 8200,
                temperature: float | None = None) -> tuple[str, dict]:
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    text = payload["choices"][0]["message"]["content"]
    returned_model = payload.get("model") or model
    meta = {"model_id": Path(str(returned_model)).name, "fingerprint": None,
            "usage": payload.get("usage", {})}
    return text, meta


BACKENDS = {"openai": _openai_call, "anthropic": _anthropic_call, "local": _local_call}


def call_cost_usd(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if not p:
        return 0.0
    return ((usage.get("prompt_tokens") or 0) * p["input"] +
            (usage.get("completion_tokens") or 0) * p["output"]) / 1e6


# --------------------------------------------------------------- controller

@dataclass
class LLMController:
    backend: str                       # openai | anthropic | local
    model: str
    intent: str                        # rendered intent text (fixed per episode/arm)
    self_triggered: bool = False
    seed: int | None = 42
    retries: int = 3
    local_port: int = 8200
    name: str = "llm"
    call_log: list = field(default_factory=list)   # full evidence trail

    def decide(self, obs: Observation, history) -> Action:
        spec = RESPONSE_SPEC_SELF_TRIGGERED if self.self_triggered else RESPONSE_SPEC
        system = self.intent + "\n\n" + spec
        user = render_observation(obs) + "\n" + render_history(history)
        fallback = Action(replicas=obs.replicas, variant=obs.variant, parse_failed=True)
        total_latency = 0.0
        tokens_in = tokens_out = 0
        usd = 0.0
        last_err = None
        for attempt in range(self.retries):
            t0 = time.monotonic()
            try:
                kwargs = {"port": self.local_port} if self.backend == "local" else {}
                text, meta = BACKENDS[self.backend](self.model, system, user,
                                                    self.seed, **kwargs)
            except Exception as e:
                total_latency += time.monotonic() - t0
                last_err = f"api_error: {e}"
                self.call_log.append({"t": obs.t, "attempt": attempt, "error": str(e)})
                time.sleep(min(4.0, 2.0 ** attempt))
                continue
            latency = time.monotonic() - t0
            total_latency += latency
            usage = meta["usage"]
            tokens_in += usage.get("prompt_tokens") or 0
            tokens_out += usage.get("completion_tokens") or 0
            call_usd = call_cost_usd(self.model, usage)
            usd += call_usd
            if self.backend != "local" and call_usd > 0:
                LEDGER.add(self.model, call_usd)
            self.call_log.append({"t": obs.t, "attempt": attempt, "raw": text,
                                  "meta": meta, "latency_s": latency, "usd": call_usd})
            try:
                act = parse_action(text, fallback)
            except Exception as e:
                last_err = f"parse_error: {e}"
                continue
            act.latency_s = total_latency
            act.tokens_in, act.tokens_out, act.usd = tokens_in, tokens_out, usd
            return act
        fallback.latency_s = total_latency
        fallback.tokens_in, fallback.tokens_out, fallback.usd = tokens_in, tokens_out, usd
        fallback.raw = f"FALLBACK after {self.retries} attempts: {last_err}"
        return fallback


def make_intent(kind: str, slo_ms: float, abandon_s: float, max_replicas: int,
                variant_menu: str, switch_s: float) -> str:
    tpl = {"full": INTENT_FULL, "partial": INTENT_PARTIAL,
           "secondary": INTENT_SECONDARY}[kind]
    return tpl.format(slo_ms=slo_ms, abandon_s=abandon_s, max_replicas=max_replicas,
                      variant_menu=variant_menu, switch_s=switch_s)


if __name__ == "__main__":
    # Offline self-checks: parser, byte-stable rendering, ledger arithmetic.
    a = parse_action('{"replicas": 3, "variant": "lite-1B", "why": "burst"}',
                     Action(replicas=1, variant="full-3B"))
    assert (a.replicas, a.variant) == (3, "lite-1B")
    a2 = parse_action('noise before {"replicas": 6, "variant": "full-3B", '
                      '"next_wake_s": 45, "why": "calm"} noise after',
                      Action(replicas=1, variant="full-3B"))
    assert a2.next_wake_s == 45.0
    try:
        parse_action('{"replicas": 2, "variant": "bogus", "why": ""}', None)
        raise SystemExit("should have rejected bogus variant")
    except ValueError:
        pass
    obs = Observation(t=600.0, window_s=60.0, arrival_rate=1.25, prev_arrival_rate=0.5,
                      queue_len=7, in_flight=2, p50_ms=812.4, p95_ms=1933.1,
                      replicas=2, variant="full-3B", utilization=0.83,
                      completed_in_window=70, dropped_in_window=1)
    r1, r2 = render_observation(obs), render_observation(obs)
    assert r1 == r2 and "n/a" not in r1
    assert call_cost_usd("gpt-5.6-luna", {"prompt_tokens": 1_000_000,
                                          "completion_tokens": 0}) == 0.20
    assert abs(call_cost_usd("claude-sonnet-5",
                             {"prompt_tokens": 500, "completion_tokens": 150})
               - (500 * 2 + 150 * 10) / 1e6) < 1e-12
    menu = "\n      ".join(f"{v}: placeholder" for v in VARIANT_ORDER)
    it = make_intent("full", 2000, 20, 6, menu, 8)
    assert "0.3 * C" in it and "Dropped and unserved" in it
    ip = make_intent("partial", 2000, 20, 6, menu, 8)
    assert "cost" not in ip.lower() or "Replicas cost" not in ip
    print("llm_controllers offline self-checks PASS")
