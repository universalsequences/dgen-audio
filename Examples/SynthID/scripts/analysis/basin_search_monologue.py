#!/usr/bin/env python3
"""Candidate-vectorized basin search for the monologue-bass circuit voice.

BasinSearch v2's algorithm (stratified uniform round + two Gaussian
resampling rounds with shrinking sigma, stratified elites over fBase x
shape), run in NumPy with the whole candidate batch vectorized per sample:
the SVF loop iterates over 26,624 samples doing [B]-wide vector math, so a
256-candidate batch renders in seconds. Rankings feed the GPU scalar Adam
polish; final scoring always goes through the canonical float32 reference.

Usage:
  basin_search_monologue.py --target prepared.wav --base-params init.json \
      --out out_dir [--count 8192] [--batch 256] [--seed 6]
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

# name -> (min, max, mode); mirrors KickParamSpecs.monologueBass
SEARCH_SPACE = {
    "shape": (0.0, 1.0, "linear"),
    "pw": (0.03, 0.97, "linear"),
    "fBase": (30.0, 8000.0, "log"),
    "fAmt": (0.0, 12000.0, "log1p"),
    "fDecay": (0.005, 2.0, "log"),
    "res": (0.5, 6.0, "log"),
    "attackTime": (0.001, 0.5, "log"),
    "decayTime": (0.01, 2.0, "log"),
    "sustain": (0.0, 1.0, "linear"),
    "releaseTime": (0.01, 1.0, "log"),
    "drive": (0.25, 8.0, "log"),
    "outGain": (0.05, 2.0, "log"),
    "vco2Level": (0.0, 1.5, "linear"),
    "vco2Detune": (0.05, 2.0, "log"),
    "satGain": (0.25, 8.0, "log"),
    "satBias": (-0.4, 0.4, "linear"),
    "satA2": (-1.0, 1.0, "linear"),
    "satA3": (-1.0, 1.0, "linear"),
    "satA5": (-0.5, 0.5, "linear"),
    "filtSat": (0.0, 4.0, "linear"),
}
NAMES = list(SEARCH_SPACE)


def to_z(name, v):
    lo, hi, mode = SEARCH_SPACE[name]
    if mode == "log":
        return np.log(v)
    if mode == "log1p":
        return np.log1p(v)
    return v


def from_z(name, z):
    lo, hi, mode = SEARCH_SPACE[name]
    if mode == "log":
        return np.exp(z)
    if mode == "log1p":
        return np.expm1(z)
    return z


Z_LO = np.array([to_z(n, SEARCH_SPACE[n][0] if SEARCH_SPACE[n][2] != "log"
                      else SEARCH_SPACE[n][0]) for n in NAMES])
Z_HI = np.array([to_z(n, SEARCH_SPACE[n][1]) for n in NAMES])
SPAN = Z_HI - Z_LO


def render_batch(z, f0, note_off, frames, rate):
    """Render B candidates, vectorized over the batch dimension.

    z: [B, P] transformed params. Returns [B, frames] float32.
    """
    B = z.shape[0]
    cols = {n: from_z(n, z[:, i]).astype(np.float32)[:, None]
            for i, n in enumerate(NAMES)}  # each [B,1]
    t = rr.dgen_time_ramp(frames, np.float32(rate))[None, :]  # [1,F]
    one = np.float32(1.0)

    phase1 = rr.dgen_phasor(frames, float(f0), float(rate))[None, :]
    phase2 = np.mod(phase1 - t * cols["vco2Detune"], one).astype(np.float32)
    dt = np.float32(np.clip(f0 / rate, 0.000001, 0.5))

    def polyblep(p):
        lx = p / dt
        left = 2.0 * lx - lx * lx - 1.0
        rx = (p - 1.0) / dt
        right = rx * rx + 2.0 * rx + 1.0
        return (p < dt) * left + (p > (1.0 - dt)) * right

    def osc(p):
        saw = p * 2.0 - 1.0 - polyblep(p)
        w = np.clip(cols["pw"], 0.01, 0.99)
        falling = np.mod(p - w, 1.0)
        raw = (p < w) * 2.0 - 1.0
        pulse = raw + polyblep(p) - polyblep(falling)
        return (1.0 - cols["shape"]) * saw + cols["shape"] * pulse

    mixed = (osc(phase1) + cols["vco2Level"] * osc(phase2)) / (1.0 + cols["vco2Level"])

    y = cols["satGain"] * mixed + cols["satBias"]
    b = cols["satBias"]
    pre = (y + cols["satA2"] * y**2 + cols["satA3"] * y**3 + cols["satA5"] * y**5
           - (b + cols["satA2"] * b**2 + cols["satA3"] * b**3 + cols["satA5"] * b**5))
    pre = pre.astype(np.float32)

    cutoff = np.clip(cols["fBase"] + cols["fAmt"] * np.exp(-t / cols["fDecay"]),
                     20.0, rate * 0.49)
    g_all = np.tan(np.float32(np.pi) * cutoff / np.float32(rate)).astype(np.float32)
    k_damp = (1.0 / cols["res"]).astype(np.float32)[:, 0]
    k_sat = cols["filtSat"].astype(np.float32)[:, 0]

    out = np.empty((B, frames), dtype=np.float32)
    ic1 = np.zeros(B, dtype=np.float32)
    ic2 = np.zeros(B, dtype=np.float32)
    for i in range(frames):
        g = g_all[:, i]
        a1 = 1.0 / (1.0 + g * (g + k_damp))
        a2 = g * a1
        s1 = ic1 / (1.0 + np.abs(k_sat * ic1))
        s2 = ic2 / (1.0 + np.abs(k_sat * ic2))
        v3 = pre[:, i] - s2
        v1 = a1 * s1 + a2 * v3
        v2 = s2 + g * v1
        ic1 = 2.0 * v1 - s1
        ic2 = 2.0 * v2 - s2
        out[:, i] = v2

    attack = 1.0 - np.exp(-t / cols["attackTime"])
    decay = cols["sustain"] + (1.0 - cols["sustain"]) * np.exp(-t / cols["decayTime"])
    release = 1.0 / (1.0 + np.exp((t - np.float32(note_off)) / cols["releaseTime"]))
    driven = out * attack * decay * release * cols["drive"]
    shaped = driven / (1.0 + np.abs(driven))
    return (shaped * cols["outGain"]).astype(np.float32)


class Scorer:
    """compare.py mrstft against precomputed target features."""

    def __init__(self, target, rate):
        self.rate = rate
        self.windows = compare.WINDOWS
        self.feat = {}
        for w in self.windows:
            self.feat[w] = self.features(target, w)

    @staticmethod
    def features(x, w):
        hop = w // 4
        hann = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, w)[::hop]
        scale = max(float(hann.sum()) / 2.0, 1e-12)
        return np.log(np.abs(np.fft.rfft(frames * hann, axis=1)) / scale
                      + compare.LOG_EPSILON)

    def score(self, x):
        return sum(
            float(np.mean(np.abs(self.features(x, w) - self.feat[w])))
            for w in self.windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--base-params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    target, rate = compare.read_wav(args.target)
    base = json.load(open(args.base_params))
    f0 = base.get("subF0", 110.0)
    note_off = base.get("subNoteOff", 0.6)
    frames = len(target)
    scorer = Scorer(target, rate)
    rng = np.random.default_rng(args.seed)

    def score_all(zs, label):
        scores = np.empty(len(zs))
        for lo in range(0, len(zs), args.batch):
            hi = min(lo + args.batch, len(zs))
            audio = render_batch(zs[lo:hi], f0, note_off, frames, rate)
            for j in range(hi - lo):
                scores[lo + j] = scorer.score(audio[j])
            print(f"  {label} {hi}/{len(zs)} best={scores[:hi].min():.5f}",
                  flush=True)
        return scores

    inset = SPAN * 0.001
    z0 = rng.uniform(Z_LO + inset, Z_HI - inset, size=(args.count, len(NAMES)))
    s0 = score_all(z0, "round0")

    archive_z, archive_s = z0, s0
    n_children = args.count * 15 // 16
    for rnd in (1, 2):
        sigma = 0.08 / (2 ** (rnd - 1))
        order = np.argsort(archive_s)
        parents = archive_z[order[: max(64, len(order) // 32)]]
        picks = rng.integers(0, len(parents), size=n_children)
        kids = parents[picks] + rng.normal(0, 1, (n_children, len(NAMES))) * sigma * SPAN
        kids = np.clip(kids, Z_LO + inset, Z_HI - inset)
        ks = score_all(kids, f"round{rnd}")
        archive_z = np.vstack([archive_z, kids])
        archive_s = np.concatenate([archive_s, ks])

    # Stratified elites over fBase (log bands) x shape halves + 2 global.
    fbase_idx = NAMES.index("fBase")
    shape_idx = NAMES.index("shape")
    bands = [(60, 120), (120, 240), (240, 480), (480, 960), (960, 2000)]
    elites = []
    for lo, hi in bands:
        for shape_half in (0, 1):
            mask = ((np.exp(archive_z[:, fbase_idx]) >= lo)
                    & (np.exp(archive_z[:, fbase_idx]) < hi)
                    & ((archive_z[:, shape_idx] >= 0.5) == bool(shape_half)))
            if mask.any():
                idx = np.flatnonzero(mask)[np.argmin(archive_s[mask])]
                elites.append(idx)
    for idx in np.argsort(archive_s):
        if idx not in elites:
            elites.append(int(idx))
        if len(elites) >= 12:
            break

    out = Path(args.out)
    (out / "elites").mkdir(parents=True, exist_ok=True)
    report = []
    for rank, idx in enumerate(elites[:12]):
        vals = dict(base)
        for i, n in enumerate(NAMES):
            vals[n] = float(from_z(n, archive_z[idx, i]))
        vals["subF0"] = f0
        vals["subNoteOff"] = note_off
        path = out / "elites" / f"elite-{rank:02d}.json"
        path.write_text(json.dumps(vals, indent=1, sort_keys=True))
        report.append({"rank": rank, "score": float(archive_s[idx]),
                       "fBase": vals["fBase"], "shape": vals["shape"],
                       "filtSat": vals["filtSat"], "satGain": vals["satGain"]})
        print(f"elite-{rank:02d}: score={archive_s[idx]:.5f} "
              f"fBase={vals['fBase']:.1f} shape={vals['shape']:.2f} "
              f"res={vals['res']:.2f} filtSat={vals['filtSat']:.2f} "
              f"vco2Level={vals['vco2Level']:.2f}")
    (out / "basin_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
