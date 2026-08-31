"""INA3221 power-rail sampler for the Jetson AGX Orin (S03 RQ3 + Layer-A energy).

Reads the sysfs hwmon nodes enumerated by the committed go/no-go probe
(`data/jetson-probe/hardware-probe.json`). All nodes are world-readable;
no sudo. Sampling method per PROTOCOL Sec. 6: direct reads at 10 Hz nominal
(probe-verified ceiling ~750 Hz, jitter <1 ms at 10 Hz), trapezoidal integration.

Attribution limits (D-003, stated wherever numbers are reported):
- VDD_GPU_SOC fuses GPU+SOC; VDD_CPU_CV fuses CPU+CV. Rail != process.
- Current LSB is 20 mA => ~0.4 W quantisation per sample on the ~20 V rails.
- All energy numbers are idle-subtracted with the idle window recorded adjacent
  in time, and carry the power mode (pinned MODE_30W) in their provenance.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field

RAILS = {
    "VDD_GPU_SOC":     ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 1),
    "VDD_CPU_CV":      ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 2),
    "VIN_SYS_5V0":     ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 3),
    "VDDQ_VDD2_1V8AO": ("/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2", 2),
}


def read_rails_mw() -> dict[str, float]:
    """One instantaneous sample of every rail, in mW."""
    out = {}
    for name, (h, ch) in RAILS.items():
        with open(f"{h}/in{ch}_input") as f:
            mv = int(f.read())
        with open(f"{h}/curr{ch}_input") as f:
            ma = int(f.read())
        out[name] = mv * ma / 1000.0
    return out


def power_mode() -> str:
    try:
        return subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                              timeout=10).stdout.strip().replace("\n", " ")
    except Exception as e:  # recorded, never fatal
        return f"unavailable: {e}"


@dataclass
class PowerTrace:
    """Result of a sampling window: timestamps (monotonic s) and per-rail mW."""
    t: list[float] = field(default_factory=list)
    mw: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in RAILS})

    def energy_j(self, rail: str) -> float:
        """Trapezoidal integral of one rail over the window, in joules."""
        ts, ps = self.t, self.mw[rail]
        if len(ts) < 2:
            return float("nan")
        j = 0.0
        for i in range(1, len(ts)):
            j += (ps[i - 1] + ps[i]) / 2.0 * (ts[i] - ts[i - 1]) / 1000.0
        return j

    def mean_mw(self, rail: str) -> float:
        ts = self.t
        if len(ts) < 2:
            return float("nan")
        return self.energy_j(rail) * 1000.0 / (ts[-1] - ts[0])

    def duration_s(self) -> float:
        return self.t[-1] - self.t[0] if len(self.t) >= 2 else 0.0

    def to_dict(self) -> dict:
        return {"t": self.t, "mw": self.mw, "n": len(self.t),
                "duration_s": self.duration_s()}


class PowerSampler:
    """Background 10 Hz sampler. Usage:
        with PowerSampler() as ps: ...work...
        trace = ps.trace
    """

    def __init__(self, hz: float = 10.0):
        self.period = 1.0 / hz
        self.trace = PowerTrace()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        t0 = time.monotonic()
        i = 0
        while not self._stop.is_set():
            now = time.monotonic()
            sample = read_rails_mw()
            self.trace.t.append(now - t0)
            for k, v in sample.items():
                self.trace.mw[k].append(v)
            i += 1
            next_t = t0 + i * self.period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)

    def __enter__(self) -> "PowerSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def measure_idle(seconds: float = 10.0) -> PowerTrace:
    """Idle-floor window; caller is responsible for the system actually being idle."""
    with PowerSampler() as ps:
        time.sleep(seconds)
    return ps.trace


if __name__ == "__main__":
    # Self-check: sample 5 s idle, print means; assert plausible ranges from the
    # committed probe (GPU_SOC idle ~1.4-2.5 W at MODE_30W; nonzero VIN).
    print("power mode:", power_mode())
    tr = measure_idle(5.0)
    for rail in RAILS:
        print(f"{rail}: mean {tr.mean_mw(rail):7.0f} mW  energy {tr.energy_j(rail):6.2f} J "
              f"over {tr.duration_s():.1f} s ({len(tr.t)} samples)")
    assert 500 < tr.mean_mw("VDD_GPU_SOC") < 4000, "GPU_SOC idle out of expected range"
    assert tr.mean_mw("VIN_SYS_5V0") > 1000, "VIN_SYS implausibly low"
    assert len(tr.t) >= 45, f"sampler too slow: {len(tr.t)} samples in 5 s"
    print("SELF-CHECK PASS")
