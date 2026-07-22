#!/usr/bin/env python3
"""Residual grid + NumPy capacity probes for the monologue-bass fit.

Playbook Phase 5: locate the remaining MR-STFT distance in time x frequency
cells, then probe candidate capacity additions against the independent CPU
metric BEFORE implementing them in DGen. Probes here:

  - detune pair: osc rendered as the mean of two polyblep oscillators at
    f0 +/- delta (rule-legal symmetric detune; Phase-1 measured two-VCO
    beating with delta ~ 0.3-0.8 Hz)
  - each probe gets a light coordinate re-fit of the coupled params so it
    is not handicapped by compensations baked into the single-osc winner.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import compare  # noqa: E402
import render_reference as rr  # noqa: E402
import refine_rung3  # noqa: E402


def residual_grid(target, learned, rate):
    """Where does log-spectral distance live? 2048-window grid."""
    win, hop = 2048, 512
    hann = np.hanning(win).astype(np.float32)
    scale = max(float(hann.sum()) / 2.0, 1e-12)

    def feats(x):
        frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
        return np.log(np.abs(np.fft.rfft(frames * hann, axis=1)) / scale + 1e-3)

    ft, fl = feats(target), feats(learned)
    diff = np.abs(ft - fl)  # [frame, bin]
    freqs = np.fft.rfftfreq(win, 1 / rate)
    tsteps = np.arange(diff.shape[0]) * hop / rate
    fbands = [(0, 80), (80, 200), (200, 500), (500, 1200), (1200, 3000),
              (3000, 8000), (8000, 22050)]
    tbands = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.62)]
    total = diff.sum()
    print("\nresidual grid (% of total 2048-window residual):")
    print("time\\freq   " + "".join(f"{lo}-{hi}".rjust(11) for lo, hi in fbands))
    for tlo, thi in tbands:
        tm = (tsteps >= tlo) & (tsteps < thi)
        row = []
        for flo, fhi in fbands:
            fm = (freqs >= flo) & (freqs < fhi)
            row.append(diff[np.ix_(tm, fm)].sum() / total * 100)
        print(f"{tlo:.2f}-{thi:.2f}s " + "".join(f"{v:10.1f}%" for v in row))


def render_detuned(params, frames, rate, delta_hz):
    """Two symmetric polyblep oscillator pairs at subF0 +/- delta."""
    p = dict(params)
    t = rr.dgen_time_ramp(frames, np.float32(rate))
    out = None
    for sign in (+1.0, -1.0):
        q = dict(p)
        q["subF0"] = float(p.get("subF0", 110.0)) + sign * delta_hz
        osc = rr.render_subtractive_bass(
            q, t, np.float32(rate), True, oscillator_only=True)
        out = osc if out is None else out + osc
    osc = (out * np.float32(0.5)).astype(np.float32)
    # rest of the voice, transplanted from render_subtractive_bass
    cutoff = (np.float32(p["fBase"])
              + np.float32(p["fAmt"]) * np.exp(-t / np.float32(p["fDecay"])))
    filtered = rr.time_varying_lowpass_biquad(osc, cutoff, np.float32(p["res"]))
    attack = np.float32(1.0) - np.exp(-t / np.float32(p["attackTime"]))
    sustain = np.float32(p["sustain"])
    decay = sustain + (np.float32(1.0) - sustain) * np.exp(
        -t / np.float32(p["decayTime"]))
    release = np.float32(1.0) / (
        np.float32(1.0) + np.exp((t - np.float32(p.get("subNoteOff", 0.6)))
                                 / np.float32(p["releaseTime"])))
    driven = filtered * attack * decay * release * np.float32(p["drive"])
    shaped = driven / (np.float32(1.0) + np.abs(driven))
    return (shaped * np.float32(p["outGain"])).astype(np.float32)


class DetunedObjective(refine_rung3.Objective):
    def __init__(self, *args, delta_hz=0.0, **kw):
        self.delta_hz = delta_hz
        super().__init__(*args, **kw)

    def evaluate(self, params):
        rendered = render_detuned(
            params, self.frames, self.sample_rate, self.delta_hz)
        peak = float(np.max(np.abs(rendered)))
        if peak > 0.9:
            rendered = rendered * np.float32(0.9 / peak)
        filtered = compare.capture_highpass(
            rendered, self.sample_rate, self.highpass_hz)
        return self.distance_filtered(filtered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/Users/alecresende/code/swift/dgen/output/monologue_bass"))
    ap.add_argument("--deltas", type=str, default="0,0.2,0.35,0.5,0.7,1.0")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    target, rate = compare.read_wav(str(args.root / "prepared/target.wav"))
    initial, _ = compare.read_wav(str(args.root / "prepared/initial.wav"))
    learned, _ = compare.read_wav(str(args.root / "real/learned.wav"))
    params = json.load(open(args.root / "real/refined_params.json"))

    residual_grid(
        compare.capture_highpass(target, rate, 30.0),
        compare.capture_highpass(learned, rate, 30.0), rate)

    refine_rung3.BOUNDS = refine_rung3.BOUNDS_SUBTRACTIVE_BASS
    refit_order = ["fBase", "fAmt", "fDecay", "res", "shape", "pw",
                   "drive", "outGain", "decayTime", "releaseTime"]
    print("\ndetune probes (light re-fit of coupled params per delta):")
    for delta in [float(x) for x in args.deltas.split(",")]:
        obj = DetunedObjective(target, initial, rate, 30.0,
                               profile="subtractive-bass", delta_hz=delta)
        refit, dist = refine_rung3.coordinate_refine(
            obj, dict(params), passes=args.passes, steps=13,
            order_override=refit_order, contraction_rate=0.55)
        print(f"  delta={delta:.2f} Hz -> mrstft {dist:.6f} "
              f"(improvement {1 - dist / 0.418476:.2%}) "
              f"res={refit['res']:.2f} drive={refit['drive']:.2f} "
              f"fAmt={refit['fAmt']:.0f}")


if __name__ == "__main__":
    main()
