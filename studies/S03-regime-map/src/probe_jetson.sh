#!/usr/bin/env bash
# Jetson capability probe for the S03 experiments.
# Enumerates the scientifically relevant platform and INA3221 state.
# Read-only; requires no sudo (privileged-command failures remain visible).
#
# Usage: probe_jetson.sh <output-file>
set -u
OUT="${1:?usage: probe_jetson.sh <output-file>}"
exec > "$OUT" 2>&1

section() { echo; echo "==================== $1 ===================="; }

echo "S03 Jetson go/no-go probe"
echo "observation date: $(date -u +%Y-%m-%d)"

section "platform"
uname -srm
sed -nE 's/^# (R[0-9]+) \(release\), REVISION: ([0-9.]+).*/L4T: \1.\2/p' \
  /etc/nv_tegra_release 2>&1
echo "--- JetPack/L4T packages:"
dpkg -l 2>/dev/null | grep -Ei 'nvidia-jetpack|nvidia-l4t-core' | awk '{print $2, $3}'
echo "--- module:"
cat /sys/firmware/devicetree/base/model 2>&1; echo
echo "--- CPUs online: $(nproc); governors:"
sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>&1
echo "--- memory:"
free -m | head -2

section "power mode"
nvpmodel -q 2>&1
echo "--- jetson_clocks --show (expected to need sudo; failure recorded as evidence):"
jetson_clocks --show 2>&1 | head -20

section "INA3221 enumeration"
for dev in /sys/bus/i2c/drivers/ina3221/*-00*/; do
  echo "--- device: $dev"
  for hw in "$dev"hwmon/hwmon*/; do
    echo "hwmon: $hw  name=$(cat "${hw}name" 2>&1)"
    for ch in 1 2 3 7; do
      lbl="${hw}in${ch}_label"
      [ -f "$lbl" ] || continue
      v="${hw}in${ch}_input"; c="${hw}curr${ch}_input"
      echo "  ch${ch}: label='$(cat "$lbl" 2>&1)'  V=$(cat "$v" 2>&1) mV  I=$(cat "$c" 2>&1) mA"
      echo "         perms: $(stat -c '%A %U:%G' "$v" 2>&1) / $(stat -c '%A %U:%G' "$c" 2>&1)"
      for lim in crit max; do
        f="${hw}curr${ch}_${lim}"
        [ -f "$f" ] && echo "         curr_${lim}=$(cat "$f" 2>&1) mA"
      done
    done
  done
done

section "read latency (1000 paired V+I reads of VDD_GPU_SOC, wall time)"
H=/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1
python3 - "$H" <<'EOF'
import sys, time
h = sys.argv[1]
t0 = time.perf_counter()
n = 1000
for _ in range(n):
    open(h + "/in1_input").read()
    open(h + "/curr1_input").read()
dt = time.perf_counter() - t0
print(f"{n} paired reads in {dt:.3f} s -> {dt/n*1e3:.3f} ms/pair -> max rate ~{n/dt:.0f} Hz")
EOF

section "sampling capture: all rails, 10 Hz nominal, 30 s, idle system"
python3 - <<'EOF'
import time, json
rails = {
    "VDD_GPU_SOC":   ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 1),
    "VDD_CPU_CV":    ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 2),
    "VIN_SYS_5V0":   ("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1", 3),
    "VDDQ_VDD2_1V8AO": ("/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon2", 2),
}
samples = []
t_start = time.monotonic()
for i in range(300):
    t = time.monotonic() - t_start
    row = {"t_s": round(t, 4)}
    for name, (h, ch) in rails.items():
        mv = int(open(f"{h}/in{ch}_input").read())
        ma = int(open(f"{h}/curr{ch}_input").read())
        row[name] = {"mV": mv, "mA": ma, "mW": round(mv * ma / 1000)}
    samples.append(row)
    time.sleep(max(0, 0.1 * (i + 1) - (time.monotonic() - t_start)))
# summary
import statistics
for name in rails:
    p = [s[name]["mW"] for s in samples]
    print(f"{name}: n={len(p)} mean={statistics.mean(p):.0f} mW  min={min(p)}  max={max(p)}  stdev={statistics.pstdev(p):.0f}")
dts = [samples[i+1]["t_s"] - samples[i]["t_s"] for i in range(len(samples)-1)]
print(f"inter-sample interval: mean={statistics.mean(dts)*1e3:.1f} ms  max={max(dts)*1e3:.1f} ms  (nominal 100 ms)")
print("FIRST_5_SAMPLES_JSON:", json.dumps(samples[:5]))
print("LAST_5_SAMPLES_JSON:", json.dumps(samples[-5:]))
EOF

section "GPU state"
for d in /sys/devices/platform/bus@0/*.gpu/devfreq/*/; do
  [ -d "$d" ] || continue
  echo "devfreq: $d"
  for f in cur_freq min_freq max_freq available_frequencies governor; do
    echo "  $f = $(cat "$d$f" 2>&1)"
  done
done
echo "--- gpu load: $(cat /sys/devices/platform/bus@0/*.gpu/load 2>&1)"

section "thermal zones"
for tz in /sys/class/thermal/thermal_zone*/; do
  echo "$(cat "${tz}type" 2>&1): $(cat "${tz}temp" 2>&1) m°C"
done

section "tegrastats one-shot (5 s; failure recorded as evidence)"
timeout 6 tegrastats --interval 1000 2>&1 | head -5 | \
  sed -E 's/^[0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} //'

section "CUDA"
ls -d /usr/local/cuda-* 2>&1
/usr/local/cuda/bin/nvcc --version 2>&1 | tail -2

echo
echo "probe complete"
