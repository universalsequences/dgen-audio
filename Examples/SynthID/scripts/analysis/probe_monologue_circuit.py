#!/usr/bin/env python3
"""Circuit-topology probes for the monologue-bass fit (NumPy, CPU metric).

The 12-scalar single-stage voice ceilings at ~0.2087 (50.13%). These probes
test *structural* upgrades toward the real signal chain — nonlinear stages
between layers, saturation inside the filter — before any DGen work:

  V1  drive placed PRE-filter (osc -> softsign -> VCF -> VCA -> trim)
  V2  parametric asymmetric polynomial saturator pre-filter
      y = g*x + b; y + a2*y^2 + a3*y^3 + a5*y^5   (inert at a*=0)
  V3  V2 plus a second polynomial stage post-filter (replaces softsign)
  V4  saturation inside the biquad feedback: y_state -> y/(1+|k*y|),
      trainable k (inert at k=0) — the "scream" path.

Every new parameter is an ordinary bounded scalar (rule-legal). Each
variant gets a coordinate re-fit from both the compensated winner and a
measurement-informed sane start.
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import compare  # noqa: E402
import render_reference as rr  # noqa: E402
import refine_rung3  # noqa: E402

ROOT = Path("/Users/alecresende/code/swift/dgen/output/monologue_bass")
BASELINE = 0.418476


def biquad_sat(x, cutoff, q, k):
    """time_varying_lowpass_biquad with softsign saturation on the feedback
    states; k=0 reduces exactly to the linear filter."""
    y = np.zeros_like(x, dtype=np.float32)
    x1 = x2 = y1 = y2 = np.float32(0.0)
    angular_scale = np.float32(0.00014247585730565955)
    one, two, half = np.float32(1.0), np.float32(2.0), np.float32(0.5)
    q = np.abs(np.float32(q))
    k = np.float32(k)
    for i in range(len(x)):
        w0 = np.abs(np.float32(cutoff[i])) * angular_scale
        cw, sw = np.cos(w0, dtype=np.float32), np.sin(w0, dtype=np.float32)
        alpha = sw * half / q
        a0 = one + alpha
        b0 = ((one - cw) * half) / a0
        b1 = (one - cw) / a0
        a1 = (-two * cw) / a0
        a2c = (one - alpha) / a0
        s1 = y1 / (one + np.abs(k * y1))
        s2 = y2 / (one + np.abs(k * y2))
        out = np.float32(b0 * np.float32(x[i]) + b1 * x1 + b0 * x2
                         - a1 * s1 - a2c * s2)
        y[i] = out
        x2, x1 = x1, np.float32(x[i])
        y2, y1 = y1, out
    return y


def poly_sat(x, g, bias, a2, a3, a5):
    y = np.float32(g) * x + np.float32(bias)
    y = (y + np.float32(a2) * y * y + np.float32(a3) * y**3
         + np.float32(a5) * y**5)
    return (y - (np.float32(bias) + np.float32(a2) * bias * bias
                 + np.float32(a3) * bias**3 + np.float32(a5) * bias**5)
            ).astype(np.float32)  # remove DC of the bias operating point


def envelope(p, t):
    attack = np.float32(1.0) - np.exp(-t / np.float32(p["attackTime"]))
    dec = (np.float32(p["sustain"]) + (np.float32(1.0) - np.float32(p["sustain"]))
           * np.exp(-t / np.float32(p["decayTime"])))
    rel = np.float32(1.0) / (np.float32(1.0) + np.exp(
        (t - np.float32(p.get("subNoteOff", 0.6))) / np.float32(p["releaseTime"])))
    return (attack * dec * rel).astype(np.float32)


def render_variant(p, frames, rate, variant):
    t = rr.dgen_time_ramp(frames, np.float32(rate))
    osc = rr.render_subtractive_bass(p, t, np.float32(rate), True,
                                     oscillator_only=True)
    cutoff = (np.float32(p["fBase"])
              + np.float32(p["fAmt"]) * np.exp(-t / np.float32(p["fDecay"])))
    env = envelope(p, t)

    if variant == "V1":  # osc -> softsign(drive) -> filter -> VCA
        driven = osc * np.float32(p["drive"])
        shaped = driven / (np.float32(1.0) + np.abs(driven))
        filt = rr.time_varying_lowpass_biquad(shaped, cutoff, np.float32(p["res"]))
        return (filt * env * np.float32(p["outGain"])).astype(np.float32)

    if variant in ("V2", "V3"):
        pre = poly_sat(osc, p["satGain"], p["satBias"], p["satA2"],
                       p["satA3"], p["satA5"])
        filt = rr.time_varying_lowpass_biquad(pre, cutoff, np.float32(p["res"]))
        if variant == "V2":  # existing softsign output stage
            driven = filt * env * np.float32(p["drive"])
            shaped = driven / (np.float32(1.0) + np.abs(driven))
            return (shaped * np.float32(p["outGain"])).astype(np.float32)
        post = poly_sat(filt * np.float32(p["drive"]), 1.0, p["postBias"],
                        p["postA2"], p["postA3"], 0.0)
        return (post * env * np.float32(p["outGain"])).astype(np.float32)

    if variant == "V4":  # saturating filter feedback
        filt = biquad_sat(osc, cutoff, np.float32(p["res"]), p["filtSat"])
        driven = filt * env * np.float32(p["drive"])
        shaped = driven / (np.float32(1.0) + np.abs(driven))
        return (shaped * np.float32(p["outGain"])).astype(np.float32)

    raise ValueError(variant)


class VariantObjective(refine_rung3.Objective):
    def __init__(self, *args, variant="V1", **kw):
        self.variant = variant
        super().__init__(*args, **kw)

    def evaluate(self, params):
        rendered = render_variant(params, self.frames, self.sample_rate,
                                  self.variant)
        peak = float(np.max(np.abs(rendered)))
        if not np.isfinite(rendered).all():
            return float("inf")
        if peak > 0.9:
            rendered = rendered * np.float32(0.9 / peak)
        return self.distance_filtered(compare.capture_highpass(
            rendered, self.sample_rate, self.highpass_hz))


EXTRA_BOUNDS = {
    "satGain": (0.25, 8.0, "log"),
    "satBias": (-0.4, 0.4, "linear"),
    "satA2": (-1.0, 1.0, "linear"),
    "satA3": (-1.0, 1.0, "linear"),
    "satA5": (-0.5, 0.5, "linear"),
    "postBias": (-0.4, 0.4, "linear"),
    "postA2": (-1.0, 1.0, "linear"),
    "postA3": (-1.0, 1.0, "linear"),
    "filtSat": (0.0, 4.0, "linear"),
}

VARIANT_PARAMS = {
    "V1": [],
    "V2": ["satGain", "satBias", "satA2", "satA3", "satA5"],
    "V3": ["satGain", "satBias", "satA2", "satA3", "satA5",
           "postBias", "postA2", "postA3"],
    "V4": ["filtSat"],
}

BASE_ORDER = ["fBase", "fAmt", "fDecay", "res", "shape", "pw",
              "drive", "outGain", "decayTime", "releaseTime"]


def main():
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["V1", "V2", "V3", "V4"]
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    target, rate = compare.read_wav(str(ROOT / "prepared/target.wav"))
    initial, _ = compare.read_wav(str(ROOT / "prepared/initial.wav"))
    winner = json.load(open(ROOT / "real/refined_params.json"))
    sane = dict(winner)
    sane.update({"shape": 0.6, "pw": 0.55, "fBase": 300.0, "fAmt": 600.0,
                 "fDecay": 0.15, "res": 1.5, "attackTime": 0.005,
                 "decayTime": 0.2, "sustain": 0.0, "releaseTime": 0.02,
                 "drive": 2.5, "outGain": 0.45})
    defaults = {"satGain": 1.0, "satBias": 0.0, "satA2": 0.0, "satA3": 0.0,
                "satA5": 0.0, "postBias": 0.0, "postA2": 0.0, "postA3": 0.0,
                "filtSat": 0.0}

    refine_rung3.BOUNDS = dict(refine_rung3.BOUNDS_SUBTRACTIVE_BASS)
    refine_rung3.BOUNDS.update(EXTRA_BOUNDS)

    results = {}
    for variant in variants:
        for label, start in (("winner", winner), ("sane", sane)):
            p0 = dict(start)
            p0.update(defaults)
            obj = VariantObjective(target, initial, rate, 30.0,
                                   profile="subtractive-bass", variant=variant)
            order = VARIANT_PARAMS[variant] + BASE_ORDER
            p, d = refine_rung3.coordinate_refine(
                obj, p0, passes=passes, steps=13,
                order_override=order, contraction_rate=0.6)
            print(f"### {variant} from {label}: mrstft {d:.6f} "
                  f"(improv {1 - d / BASELINE:.2%})")
            extras = {k: round(p[k], 4) for k in VARIANT_PARAMS[variant]}
            core = {k: round(p[k], 4) for k in
                    ("fBase", "fAmt", "fDecay", "res", "drive", "shape", "pw")}
            print(f"    extras={extras}")
            print(f"    core={core}", flush=True)
            results[f"{variant}/{label}"] = {"mrstft": d, "params": p}

    best = min(results.items(), key=lambda kv: kv[1]["mrstft"])
    print(f"\nBEST: {best[0]} at {best[1]['mrstft']:.6f} "
          f"({1 - best[1]['mrstft'] / BASELINE:.2%})")
    out = ROOT / "real/circuit_probe_results.json"
    json.dump({k: {"mrstft": v["mrstft"], "params": v["params"]}
               for k, v in results.items()}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
