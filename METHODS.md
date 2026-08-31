# Experimental methods

## Managed system

The managed service performs short-text review triage on one NVIDIA Jetson AGX
Orin Developer Kit in `MODE_30W` (mode 2). Inference is served by `llama.cpp`
with CUDA. The controller jointly selects:

1. a serving-slot count from 1 through 6; and
2. one of three measured Llama 3.2 GGUF variants:
   `tiny-1B4` (1B Q4_K_M), `lite-1B` (1B Q8_0), or `full-3B` (3B Q4_K_M).

Both independent one-slot processes and one continuously batched server with
multiple slots were characterized. The calibrated plant uses the measured
slot-mode capacity curve; it never assumes linear scaling.

## Episode and workload

Each episode lasts 1,800 virtual seconds. The first 300 seconds are uniformly
excluded from scoring. Mean offered load is 2 requests/s, the end-to-end SLO is
3,000 ms, and queued requests abandon after 30 seconds. MMPP(2) arrival traces
use core index-of-dispersion levels 1, 400, and 1,000. Controller cadences are
15, 60, and 300 seconds. Seeds are paired across controller arms.

All fixed constants and calibrated capacity values are in
`studies/S03-regime-map/data/design.json`.

## Controller arms

The comparison includes a static controller, a calibrated multi-knob heuristic,
a queue-aware heuristic, periodic LLM controllers, self-triggered LLM control,
an intent-change condition, and an intentionally underspecified intent arm.
Hosted controllers used `gpt-5.6-luna` and `claude-sonnet-5`; the local
controller used the 3B Q4_K_M model on the Jetson. Prompt text and response
parsing are in `src/llm_controllers.py`.

Rule controllers were tuned per cadence on held-out seeds 1000–1009. Evaluation
uses 30 seeds for CPU-only arms and 20 paired seeds for each reported LLM cell.

## Utility and reporting

The primary utility is

```text
U_S03 = -0.5 V_offered - 0.2 accuracy_deficit - 0.3 mean_slots/6
```

`V_offered` counts late completions, abandoned requests, and the residual queue
over all offered requests. A completed-only comparison utility (`U_v2`),
goodput, utility components, decision latency, token use, hosted cost, and
failure modes are also retained.

Cell means use the seed as the unit of analysis and report two-sided 95%
Student-t confidence intervals. LLM margins are paired by seed against the rule
family with the highest mean utility in the same volatility/cadence cell. The
regime surface is retained in full rather than selecting cells after observing
results.

## Experiment blocks

- **Layer A:** real Jetson serving latency, throughput, energy, and accuracy.
- **B:** static and rule-controller sweep.
- **C:** periodic full-intent LLM sweep.
- **E:** mid-episode intent change and self-triggered control.
- **H:** partial-intent condition.
- **RQ2:** repeated byte-identical states under local serial, local batched,
  OpenAI default/seeded, and Anthropic default configurations.
- **RQ3:** idle-subtracted local deliberation energy, composed with separately
  measured serving energy.
- **Validation F:** three wall-clock replays against the real Jetson service.
- **Oracle-static robustness check:** a volatility-informed static baseline at
  `I=1000`.

## Measurement boundary

Jetson rail data came from INA3221 sysfs channels sampled at 10 Hz. The device
exposes fused `VDD_GPU_SOC` and `VDD_CPU_CV` rails, so deliberation and serving
were characterized in separate windows and composed afterward. Hosted-model
energy is not observable from the client and is not estimated here.

The release converts unavailable per-episode volatility estimates from the
non-standard JSON token `NaN` to strict JSON `null`. Pooled volatility estimates
are present in `results/analysis/realised_I.json`.
