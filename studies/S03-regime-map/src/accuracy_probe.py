"""Per-variant task accuracy, measured on-device (feeds VariantCal.accuracy).

The plant's accuracy axis is measured, never asserted: each variant classifies a
fixed labeled bank of 36 reviews (12 positive / 12 negative / 12 mixed, including
sarcasm and multi-aspect items so capability differences surface) using the SAME
service prompt as Layer-A. Grading is on the sentiment label only, parsed
case-insensitively from the output; unparseable output counts as wrong (a serving
variant that cannot follow the output contract is inaccurate in deployment).

temperature 0 (canonical decoding — this measures capability, not sampling luck).
Output: results/accuracy_probe.json with per-item records (evidence).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from artifact_io import write_json
from characterize import (VARIANTS, SERVER_BIN, MODELS, BASE_PORT, CTX,
                          launch_replicas, kill_replicas, PROMPT_TEMPLATE)

STUDY = Path(__file__).resolve().parents[1]

# (text, gold sentiment). Mixed = genuinely both-sided. Includes sarcasm and
# subtle polarity items; labels are the bank author's fixed gold standard,
# committed before any variant was scored against it.
LABELED_BANK: list[tuple[str, str]] = [
    # --- positive (12)
    ("Battery life on this device is exceptional; easily two full days of heavy use.", "positive"),
    ("Setup took under five minutes and the instructions were clear throughout.", "positive"),
    ("Firmware update fixed the connectivity drops I was seeing on the older version.", "positive"),
    ("Build quality feels premium and the hinge mechanism is smooth and solid.", "positive"),
    ("Honestly expected little at this price, but the sound is rich and detailed.", "positive"),
    ("Replacement arrived within a day and works flawlessly.", "positive"),
    ("The keyboard is quiet, responsive, and comfortable for long typing sessions.", "positive"),
    ("Photos come out crisp even when my hands shake a bit.", "positive"),
    ("It survived a drop down the stairs without a scratch.", "positive"),
    ("Support walked me through calibration and now it tracks perfectly.", "positive"),
    ("Three years of daily use and it still holds a full charge.", "positive"),
    ("The new strap design finally stays put during runs.", "positive"),
    # --- negative (12)
    ("The app crashes every time I open the settings page after the last update.", "negative"),
    ("Audio quality is muddy at high volume and the bass distorts noticeably.", "negative"),
    ("Customer service kept me on hold for forty minutes and never solved the issue.", "negative"),
    ("The subscription price doubled this year without any new features being added.", "negative"),
    ("After three months of daily use the strap broke at the buckle joint.", "negative"),
    ("Great, another update that deletes my saved presets. Just what I wanted.", "negative"),
    ("Oh sure, 'water resistant' — it died the first time it drizzled.", "negative"),
    ("The 'quick start' guide is forty pages and still skips the pairing step.", "negative"),
    ("Returned it twice; both units had the same dead pixel cluster.", "negative"),
    ("It heats up so much I can't hold it after ten minutes of navigation.", "negative"),
    ("The advertised 'silent mode' is louder than my old unit at full speed.", "negative"),
    ("Packaging was pretty, which is more than I can say for what was inside.", "negative"),
    # --- mixed (12)
    ("The delivery arrived two days late and the packaging was damaged, but support resolved it quickly.", "mixed"),
    ("The camera performs well in daylight but struggles badly in low light.", "mixed"),
    ("Shipping was fast but the item did not match the photos in the listing.", "mixed"),
    ("Love the screen; hate that the battery barely lasts a morning.", "mixed"),
    ("Solid hardware let down by clunky, ad-riddled software.", "mixed"),
    ("It's lighter than my old one, though the fan noise is much worse.", "mixed"),
    ("The stitching is beautiful, but the zipper jammed within a week.", "mixed"),
    ("Setup was painless; the companion app, however, logs me out daily.", "mixed"),
    ("Fantastic value for the price, even if the plastic feels cheap.", "mixed"),
    ("Range is excellent outdoors, yet it drops connection through one wall.", "mixed"),
    ("The seat is comfortable on short trips but unbearable after an hour.", "mixed"),
    ("Bright, vivid display — shame about the reflections in daylight.", "mixed"),
]

LABELS = ("positive", "negative", "mixed")


def classify_output(text: str) -> str | None:
    """First label word appearing in the output; None if none or ambiguous-empty."""
    t = text.lower()
    hits = [(t.find(lbl), lbl) for lbl in LABELS if lbl in t]
    if not hits:
        return None
    return min(hits)[1]


def probe_variant(variant: str, port: int = BASE_PORT) -> dict:
    records = []
    for i, (text, gold) in enumerate(LABELED_BANK):
        body = json.dumps({
            "messages": [{"role": "user",
                          "content": PROMPT_TEMPLATE.format(text=text)}],
            "max_tokens": 48, "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)["choices"][0]["message"]["content"]
        pred = classify_output(out)
        records.append({"i": i, "gold": gold, "pred": pred,
                        "correct": pred == gold, "raw": out})
    acc = sum(r["correct"] for r in records) / len(records)
    per_class = {lbl: (sum(r["correct"] for r in records if r["gold"] == lbl) /
                       sum(1 for r in records if r["gold"] == lbl))
                 for lbl in LABELS}
    return {"variant": variant, "accuracy": acc, "per_class": per_class,
            "n": len(records), "records": records}


def main() -> None:
    out = {"bank_size": len(LABELED_BANK), "temperature": 0.0,
           "variants": {}}
    for variant in VARIANTS:
        print(f"probing {variant}...", flush=True)
        procs = launch_replicas(VARIANTS[variant]["gguf"], 1)
        try:
            res = probe_variant(variant)
        finally:
            kill_replicas(procs)
            time.sleep(3)
        out["variants"][variant] = res
        print(f"  accuracy {res['accuracy']:.3f}  per-class {res['per_class']}")
    path = STUDY / "results" / "accuracy_probe.json"
    write_json(path, out)
    print("wrote", path)


if __name__ == "__main__":
    # Offline self-check of the grader itself
    assert classify_output("Sentiment: Mixed. Aspect: battery.") == "mixed"
    assert classify_output("This is a POSITIVE review about shipping.") == "positive"
    assert classify_output("The review is negative; it is not positive.") == "negative"
    assert classify_output("no label here") is None
    balance = [sum(1 for _, g in LABELED_BANK if g == l) for l in LABELS]
    assert balance == [12, 12, 12], balance
    print("grader self-checks PASS")
    main()
