# Realized experimental protocol

This protocol records the experiment that was actually run: the methodological
information needed to interpret or repeat the evidence.

The underlying protocol was registered on 2026-08-02, before any
result-producing run, and afterwards amended only in place, dated, with stated
reasons — six amendments, 2026-08-02 to 2026-08-03. Every amendment preceded
the experiment block it affected except the last: the wake-range enforcement
gap (§13) was found by a post-run audit of the self-triggered block and is
disclosed rather than repaired. §13 lists every realized deviation.

## 1. Research questions

1. Across workload volatility and control cadence, when does LLM-driven control
   exceed the utility of calibrated rule controllers after accounting for the
   controller's latency, token, monetary, accuracy, and resource costs?
2. Given byte-identical inputs, how reproducible are decisions and raw outputs
   under local serial, local continuous-batching, and hosted configurations?
3. How much local Jetson energy does one controller deliberation consume relative
   to one managed serving request?

## 2. System under management

One Jetson AGX Orin serves a short-text inference workload. A controller selects
one of three measured Llama 3.2 variants and 1–6 serving slots. Real hardware
measurements calibrate a discrete-event plant; three final cells are replayed in
wall-clock time against the real service.

## 3. Falsifiable expectations

The registered expectations were a structured rather than uniformly positive
LLM regime surface, an interior cadence optimum under high volatility, measurable
over-provisioning from an incomplete intent, reduced repeatability moving from
local serial to batched and hosted service, and materially greater energy for a
local deliberation than for one serving request. Null and opposing results remain
in the complete surface.

## 4. Independent variables and models

- Core workload index of dispersion: 1, 400, 1,000.
- Controller cadence: 15, 60, 300 seconds.
- Rule families: static, multi-knob heuristic, queue-aware heuristic.
- LLM conditions: periodic full intent, self-triggered, intent change, and
  partial intent.
- Hosted models: `gpt-5.6-luna` and `claude-sonnet-5`.
- Local controller: Llama 3.2 3B Instruct Q4_K_M.
- RQ2 serving configurations: local serial, local four-slot continuous batching,
  OpenAI default, OpenAI seeded, and Anthropic default.

Model file hashes and runtime provenance are in `env/`.

## 5. Controller parity

All arms receive the same observation fields. Rule controllers are tuned per
cadence on held-out traces. The full-intent LLM prompt states the complete scored
objective and allowed actions. The partial-intent condition omits the resource
cost term by design. A secondary intent applies at episode time 900 seconds in
the intent-change block, with an equivalent parameter change for rule arms.

## 6. Outcomes

Primary utility uses all offered work:

```text
U_S03 = -0.5 V_offered - 0.2 accuracy_deficit - 0.3 mean_slots/6
```

Late completions, abandoned requests, and the residual queue contribute to
`V_offered`. The completed-only `U_v2`, goodput, utility components, decision
latency, tokens, hosted cost, failure modes, and invocation count are always
retained. RQ2 records byte identity, action identity, direction entropy,
replica spread, and contradictions. RQ3 reports idle-subtracted component-level
energy; hosted energy is outside the client measurement boundary.

## 7. Repetition and statistics

CPU-only cells use 30 evaluation seeds. Reported LLM cells use 20 paired seeds.
Rule tuning uses seeds 1000–1009. Means and two-sided 95% Student-t confidence
intervals use the seed as the unit of analysis. LLM margins are paired by seed
against the best-mean rule family in the same volatility/cadence cell.

## 8. Failures and exclusions

Unparseable outputs after retries and persistent API errors are recorded as
failure modes rather than silently dropped. Host contention invalidates a
hardware measurement. No episode is excluded because of its outcome.

## 9. Realized run matrix

- Layer A: 42 final raw hardware cells plus accuracy measurement.
- B: 810 rule episodes.
- C: 420 periodic full-intent LLM episodes.
- E: 340 intent-change/self-triggered/rule-parity episodes.
- H: 120 partial-intent episodes.
- RQ2: 4,200 repeated calls over 30 sampled states.
- Validation F: three real-time Jetson replays.
- Oracle-static: 40 evaluation episodes at `I=1000`.

The hosted grid was sized from a 50-call-per-model cost pilot. A persistent
spend guard used a 60 USD halt threshold. Its cumulative ledger
(`results/spend_ledger.json`) records every hosted controller call, including
pilot, probe, and retried calls; per-episode usage-derived cost fields remain
in the experiment records themselves.

## 10. Traceability

The fixed design is in `data/design.json`. Raw and derived artifact inventories
are described in the repository `RESULTS.md`. `src/verify_artifact.py` enforces
row counts, unique keys, request conservation, data syntax, compressed archive
integrity, and checksums. It verifies arithmetic and inventory;
it does not by itself establish the validity of the experimental design.

## 11. Validity limits

- Most regime-map episodes use a calibrated discrete-event plant; three cells
  supply real-time hardware disagreement measurements.
- Energy attribution is component-level because the available rails are fused.
- A local control call and managed serving would contend if concurrent; the
  emulation does not model that interaction.
- Hosted service internals, snapshots, and energy are not client-observable.
- Results are specific to the recorded 30 W mode, model variants, prompt, and
  software environment.

## 12. Reporting rule

The complete cell surface, both utility definitions, cell sample sizes,
confidence intervals, and recorded failure modes are retained. Negative and null
cells are not removed.

## 13. Realized deviations and constraints

- The planned very-high volatility level was unreachable at the measured mean
  load without a negative low-state rate; 1,000 became the highest core level.
- The more expensive hosted vendor was run at 60-second cadence for cross-vendor
  coverage rather than across the full surface.
- The Anthropic model rejected non-default determinism controls, so no pinned
  Anthropic RQ2 rung could be run; the rejection is preserved as evidence.
- Self-triggered wake values were prompted to a bounded range but not clamped by
  the parser. Realized wake times and invocation counts are preserved as run.
