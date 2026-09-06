#!/usr/bin/env python3
"""SynthID Roland R-8 'Kick03' voice: NumPy reference renderer + CPU fit against
the independent MR-STFT metric (compare.py) plus a long-window low-band term,
for Assets/r8-kick03.wav (44.1 kHz, 182 ms, peak 0 dBFS).

Why a modal voice: measurement (scratch 2026-09-03, see HANDOFF_R8_KICK03.md)
shows the hit is not a swept sine with a harmonic ladder. Heterodyning along
any single pitch track finds the same *inharmonic* partial set in every window
(ratios ~1 : 1.8 : 2.7 : 3.7), and all partials drift down together by the same
factor late in the sound: a fixed inharmonic membrane mode set under a shared
multiplicative tension glide. Above the low bank sits a ringing 'knock' cluster
(~300-400 Hz, ~620 Hz, 2.4-3.0 kHz, T60 20-150 ms) and a 5 ms broadband click.

Topology (all documented scalars, no target-derived tables):
  glide   g(t) = 1 + gA1 e^{gR1 t} + gA2 e^{gR2 t}   (shared tension glide)
  low     sum_k la_k e^{ld_k t} sin(2 pi frac(lf_k (t + lg_k G(t))))   k = 1..5
          G(t) = gA1/(-gR1)(1-e^{gR1 t}) + gA2/(-gR2)(1-e^{gR2 t}),  lg_k scales the glide per mode
  mid     sum_j ma_j e^{md_j t} sin(2 pi frac(mf_j t))                j = 1..8  (fixed-pitch ring modes)
  attack  (1 - e^{-t/attackTime}) on both banks
  noise   noiseAmp * LP(noise, noiseCutoff) * e^{noiseDecay t}       (beater click)
  hiss    hissAmp * HP(noise, hissCutoff) * e^{hissDecay t}          (recording texture, fitted alone)
  out     tanh(drive * mix) / drive * outGain

Fit protocol mirrors fit_virus_kick.py: baseline = spec midpoints; restarts
scored (random draws keep the measured mode frequencies and glide); top --keep
get coordinate descent with contraction; the winner gets a final tight pass.
The gate stays compare.py; training adds a 4096 window and the low-band term.
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare
import render_reference as reference

NL = 5   # low membrane modes (shared glide)
NM = 8   # mid/high ring modes (fixed pitch)

LOW_SEED = [  # (freq Hz at t->inf, amp, decay 1/s, glide scale)
    (43.0, 0.5, -25.0, 1.6),
    (80.0, 0.3, -45.0, 1.0),
    (122.0, 0.9, -40.0, 1.0),
    (175.0, 0.25, -60.0, 1.0),
    (235.0, 0.15, -90.0, 1.0),
]
MID_SEED = [  # (freq Hz, amp, decay 1/s)
    (315.0, 0.03, -40.0), (400.0, 0.03, -40.0), (620.0, 0.04, -150.0), (1100.0, 0.01, -60.0),
    (1720.0, 0.01, -100.0), (2450.0, 0.04, -300.0), (2900.0, 0.03, -100.0), (3450.0, 0.02, -200.0),
]

BOUNDS = {
    "gA1": (0.05, 3.0, "log"),   # measured fast glide ratio 1.4-2; round 1 at 3.65 swept every mode in from >700 Hz as a fake click
    "gR1": (-1200.0, -60.0, "logneg"),
    "gA2": (0.02, 1.5, "log"),
    "gR2": (-80.0, -8.0, "logneg"),  # measured slow glide ~-20/s; round 1 at -6.5 was a constant pitch offset, not a glide
    "attackTime": (0.0001, 0.006, "log"),
    "noiseCutoff": (500.0, 14000.0, "log"),
    "noiseAmp": (0.0, 0.8, "linear"),
    "noiseDecay": (-4000.0, -40.0, "logneg"),  # the click is a 5-8 ms burst (>4 kHz RMS -15 dB at 0-5 ms, -27 at 5-15); round 1 at -112 made it a mid-band pad
    # rattle: noise gated by the negative half of the membrane signal (the fuzz on every
    # trough in the target waveform, the diffuse 0.5-3 kHz cloud over the first 50 ms)
    "rattleAmp": (0.0, 1.0, "linear"),
    "rattleHp": (300.0, 4000.0, "log"),
    "rattleDecay": (-300.0, -5.0, "logneg"),
    "hissCutoff": (2000.0, 12000.0, "log"),
    "hissAmp": (0.0, 0.02, "linear"),
    "hissDecay": (-80.0, -2.0, "logneg"),
    "drive": (0.02, 0.15, "log"),   # linear regime only: v1-v4 abused the tanh to hard-clip the body at ±0.37 (the target swings ±0.75 unclipped) because a clipped pulse train mimics the target's dense low-band smear   # the waveform shows no flat-top clipping; round 1 pinned drive at 3.9 and hard-clipped the body at 0.29
    "outGain": (0.05, 5.0, "log"),
}
for k, (f, a, d, g) in enumerate(LOW_SEED, 1):
    BOUNDS[f"lf{k}"] = (f / 1.45, f * 1.45, "log")
    BOUNDS[f"la{k}"] = (1e-3, 1.0, "log")
    BOUNDS[f"ld{k}"] = (-400.0, -3.0, "logneg")
    BOUNDS[f"lg{k}"] = (0.0, 3.0, "linear")
    BOUNDS[f"lp{k}"] = (0.0, 1.0, "linear")   # initial phase, cycles: matters once the saturator shapes the summed waveform
for j, (f, a, d) in enumerate(MID_SEED, 1):
    BOUNDS[f"mf{j}"] = (f / 1.5, f * 1.5, "log")
    BOUNDS[f"ma{j}"] = (1e-4, 0.3, "log")
    BOUNDS[f"md{j}"] = (-800.0, -5.0, "logneg")
ORDER = list(BOUNDS.keys())

MEASURED = {
    "gA1": 0.4, "gR1": -250.0, "gA2": 0.6, "gR2": -20.0, "attackTime": 0.0005,
    "noiseCutoff": 6000.0, "noiseAmp": 0.5, "noiseDecay": -900.0,
    "rattleAmp": 0.0, "rattleHp": 1000.0, "rattleDecay": -40.0,
    "hissCutoff": 5000.0, "hissAmp": 0.0005, "hissDecay": -20.0,
    "drive": 0.1, "outGain": 1.0,
}
for k, (f, a, d, g) in enumerate(LOW_SEED, 1):
    MEASURED.update({f"lf{k}": f, f"la{k}": a, f"ld{k}": d, f"lg{k}": g, f"lp{k}": 0.0})
for j, (f, a, d) in enumerate(MID_SEED, 1):
    MEASURED.update({f"mf{j}": f, f"ma{j}": a, f"md{j}": d})
FREQ_KEYS = ["gA1", "gR1", "gA2", "gR2"] + [f"lf{k}" for k in range(1, NL + 1)] + [f"mf{j}" for j in range(1, NM + 1)]

TRAIN_WINDOWS = (256, 512, 1024, 2048, 4096)
HARMONIC_WEIGHT = 0.5   # weight of the low-band long-window term
HARMONIC_FLOOR = 1e-3


def transformed(name, v):
    m = BOUNDS[name][2]
    return math.log(v) if m == "log" else math.log(-v) if m == "logneg" else v


def natural(name, z):
    m = BOUNDS[name][2]
    return math.exp(z) if m == "log" else -math.exp(z) if m == "logneg" else z


def midpoints():
    out = {}
    for name, (lo, hi, _) in BOUNDS.items():
        zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
        out[name] = natural(name, 0.5 * (zl + zh))
    return out


def biquad_coeffs(mode, fc, q, sr):
    w0 = 2.0 * math.pi * fc / sr
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2.0 * q)
    if mode == "bp":
        b0, b1, b2 = alpha, 0.0, -alpha
    elif mode == "hp":
        b0, b1, b2 = (1 + c) / 2, -(1 + c), (1 + c) / 2
    else:
        b0, b1, b2 = (1 - c) / 2, 1 - c, (1 - c) / 2
    a0 = 1 + alpha
    return b0 / a0, b1 / a0, b2 / a0, -2 * c / a0, (1 - alpha) / a0


def biquad(x, mode, fc, q, sr):
    b0, b1, b2, a1, a2 = biquad_coeffs(mode, fc, q, sr)
    y = np.empty(len(x), dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x.tolist()):
        out = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = out
        x2, x1 = x1, v
        y2, y1 = y1, out
    return y


@functools.lru_cache(maxsize=512)
def noise_hp(frames, sr, fc):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    return biquad(n, "hp", fc, 0.707, sr)


@functools.lru_cache(maxsize=512)
def noise_lp(frames, sr, fc):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    return biquad(n, "lp", fc, 0.707, sr)


def glide_integral(p, t):
    return (p["gA1"] / (-p["gR1"]) * (1.0 - np.exp(p["gR1"] * t))
            + p["gA2"] / (-p["gR2"]) * (1.0 - np.exp(p["gR2"] * t)))


def render(p, frames, sr):
    t = np.arange(frames, dtype=np.float64) / sr
    G = glide_integral(p, t)
    attack = 1.0 - np.exp(-t / p["attackTime"])
    mix = np.zeros(frames)
    for k in range(1, NL + 1):
        phi = p[f"lf{k}"] * (t + p[f"lg{k}"] * G) + p[f"lp{k}"]
        frac = phi - np.floor(phi)
        mix = mix + p[f"la{k}"] * np.exp(p[f"ld{k}"] * t) * np.sin(2.0 * np.pi * frac)
    for j in range(1, NM + 1):
        phi = p[f"mf{j}"] * t
        frac = phi - np.floor(phi)
        mix = mix + p[f"ma{j}"] * np.exp(p[f"md{j}"] * t) * np.sin(2.0 * np.pi * frac)
    low = mix.copy()
    if p["rattleAmp"] > 0:
        mix = mix + p["rattleAmp"] * noise_hp(frames, sr, round(p["rattleHp"], 6)) * np.maximum(-low, 0.0) * np.exp(p["rattleDecay"] * t)
    mix = mix * attack
    if p["noiseAmp"] > 0:
        mix = mix + p["noiseAmp"] * noise_lp(frames, sr, round(p["noiseCutoff"], 6)) * np.exp(p["noiseDecay"] * t)
    if p["hissAmp"] > 0:
        mix = mix + p["hissAmp"] * noise_hp(frames, sr, round(p["hissCutoff"], 6)) * np.exp(p["hissDecay"] * t)
    y = np.tanh(p["drive"] * mix) / p["drive"] * p["outGain"]
    return y.astype(np.float32)


class HarmonicTracks:
    """Low-band mode term and diagnostic (keeps the HarmonicTracks name so
    deficit_table.py prints it): log magnitude of a long (8192) Hann window,
    hop 1024, bins 20..700 Hz — resolution ~6 Hz at 48 kHz, enough to place the
    inharmonic membrane modes the 256..4096 windows blur together. table() lists
    the strongest peaks per window for target and synth side by side."""
    def __init__(self, target, sr, window=8192, hop=1024, f_lo=20.0, f_hi=700.0):
        self.sr, self.window, self.hop = sr, window, hop
        self.win = np.hanning(window)
        fr = np.fft.rfftfreq(window, 1.0 / sr)
        self.bins = (fr >= f_lo) & (fr <= f_hi)
        self.fr = fr[self.bins]
        self.target = self.features(target)

    def features(self, x):
        n = len(x)
        pad = np.concatenate([np.zeros(self.window // 2), x.astype(np.float64), np.zeros(self.window)])
        starts = range(0, n, self.hop)
        frames = np.stack([pad[s:s + self.window] * self.win for s in starts])
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, self.bins] / (self.win.sum() / 2)
        return np.log(mag + HARMONIC_FLOOR)

    def loss(self, x):
        return float(np.mean(np.abs(self.features(x) - self.target)))

    def table(self, x, top=6):
        A, B = self.features(x), self.target
        lines = [f"  ms   target peaks Hz:dB (8192 win)              | synth peaks"]
        for j in range(A.shape[0]):
            def pk(S):
                idx = [i for i in range(1, len(S) - 1) if S[i] > S[i - 1] and S[i] >= S[i + 1] and S[i] > math.log(HARMONIC_FLOOR) + 0.7]
                idx = sorted(sorted(idx, key=lambda i: -S[i])[:top])
                return " ".join(f"{self.fr[i]:4.0f}:{20 * S[i] / math.log(10):4.0f}" for i in idx)
            lines.append(f"{j * self.hop / self.sr * 1000:5.0f} {pk(B[j]):46s} | {pk(A[j])}")
        return "\n".join(lines)


POOLED_WINDOWS = (256, 1024)
POOLED_EPSILON = 1e-6
POOLED_MIN_HZ = 2500.0
POOLED_WEIGHT = 0.0


class Objective:
    def __init__(self, target, sr, highpass_hz, windows=compare.WINDOWS, harmonics=None, harmonic_weight=0.0,
                 pooled_weight=0.0, bands=32):
        self.sr, self.frames, self.hp, self.windows = sr, len(target), highpass_hz, windows
        self.target = compare.capture_highpass(target, sr, highpass_hz)
        self.feat = {w: self.features(self.target, w) for w in windows}
        self.pooled_weight = pooled_weight
        self.pool = {}
        if pooled_weight > 0:
            edges = np.geomspace(150.0, min(20000.0, sr / 2), bands + 1)
            for w in POOLED_WINDOWS:
                fr = np.fft.rfftfreq(w, 1.0 / sr)
                idx = np.searchsorted(edges, fr, side="right") - 1
                m = np.zeros((len(fr), bands))
                ok = (idx >= 0) & (idx < bands)
                m[np.arange(len(fr))[ok], idx[ok]] = 1.0
                m = m[:, (m.sum(axis=0) > 0) & (edges[:-1] >= POOLED_MIN_HZ)]
                self.pool[w] = m / np.maximum(m.sum(axis=0, keepdims=True), 1)
            self.pooled_feat = {w: self.pooled_features(self.target, w) for w in POOLED_WINDOWS}
        self.harmonics = harmonics
        self.harmonic_weight = harmonic_weight
        self.evals = 0

    def features(self, signal, w):
        hop = w // 4
        win = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, w)[::hop]
        scale = max(float(win.sum()) / 2.0, 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1)) / scale
        return np.log(mag + compare.LOG_EPSILON)

    def pooled_features(self, signal, w):
        hop = w // 4
        win = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, w)[::hop]
        scale = max(float(win.sum()) / 2.0, 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1)) / scale
        return 0.5 * np.log((mag ** 2) @ self.pool[w] + POOLED_EPSILON ** 2)

    def distance(self, signal):
        return sum(float(np.mean(np.abs(self.features(signal, w) - self.feat[w]))) for w in self.windows)

    def pooled_distance(self, signal):
        return sum(float(np.mean(np.abs(self.pooled_features(signal, w) - self.pooled_feat[w]))) for w in POOLED_WINDOWS)

    def evaluate(self, p, parts=False):
        self.evals += 1
        y = render(p, self.frames, self.sr)
        pk = float(np.max(np.abs(y)))
        if pk > 0.9:
            y = y * np.float32(0.9 / pk)
        yh = compare.capture_highpass(y, self.sr, self.hp)
        spec = self.distance(yh)
        harm = self.harmonics.loss(y) if self.harmonics is not None else 0.0
        pooled = self.pooled_distance(yh) if self.pooled_weight > 0 else 0.0
        if parts:
            return spec, harm, pooled
        return spec + self.harmonic_weight * harm + self.pooled_weight * pooled


def coordinate_refine(obj, start, passes, steps, contraction=0.55, span_scale=1.0, order=ORDER, log=print):
    p = dict(start)
    best = obj.evaluate(p)
    for k in range(passes):
        c = contraction ** k
        for name in order:
            lo, hi, _ = BOUNDS[name]
            zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
            center = transformed(name, p[name])
            span = (zh - zl) * c * span_scale
            a, b = max(zl, center - span / 2), min(zh, center + span / 2)
            lv, lb = p[name], best
            for z in np.linspace(a, b, steps):
                cand = dict(p); cand[name] = natural(name, float(z))
                d = obj.evaluate(cand)
                if d < lb:
                    lb, lv = d, cand[name]
            if lb < best:
                p[name], best = lv, lb
        log(f"  pass {k}: {best:.6f} (evals {obj.evals})")
    return p, best


def resample_fft(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    m = min(len(X), len(Y)); Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Assets/r8-kick03.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--final-passes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", help="params json to refine from (still scored against restarts)")
    ap.add_argument("--highpass-hz", type=float, default=compare.DEFAULT_HIGHPASS_HZ)
    ap.add_argument("--harmonic-weight", type=float, default=HARMONIC_WEIGHT)
    ap.add_argument("--pooled-weight", type=float, default=POOLED_WEIGHT)
    ap.add_argument("--only", help="comma list of params to refine (others frozen at --start); skips restarts")
    ap.add_argument("--bound", action="append", default=[], help="override a bound: name=lo:hi (repeatable); recorded in the report")
    args = ap.parse_args()
    for spec in args.bound:
        name, rng = spec.split("=")
        lo, hi = (float(v) for v in rng.split(":"))
        BOUNDS[name] = (lo, hi, BOUNDS[name][2])
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "run.log"), "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    target, sr_in = compare.read_wav(args.target)
    if sr_in != args.sr:
        target = resample_fft(target, sr_in, args.sr)
    sr = args.sr
    frames = len(target)
    reference.write_wav(os.path.join(args.out, "target.wav"), target, sr)
    tracks = HarmonicTracks(target, sr)
    obj = Objective(target, sr, args.highpass_hz, TRAIN_WINDOWS, tracks, args.harmonic_weight, args.pooled_weight)
    gate = Objective(target, sr, args.highpass_hz, compare.WINDOWS)
    base = midpoints()
    baseline = obj.evaluate(base)
    gate_baseline = gate.evaluate(base)
    reference.write_wav(os.path.join(args.out, "initial.wav"), render(base, frames, sr), sr)
    log(f"frames={frames} sr={sr} baseline(midpoints)={baseline:.6f} gate_baseline={gate_baseline:.6f}")

    rng = np.random.default_rng(args.seed)
    cands = [("measured", dict(MEASURED)), ("midpoint", dict(base))]
    if args.start:
        with open(args.start) as f:
            s = json.load(f)
        s = s.get("params", s)
        for name, (lo, hi, _) in BOUNDS.items():
            s.setdefault(name, base[name])
            zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
            s[name] = natural(name, min(max(transformed(name, s[name]), zl), zh))
        cands.append(("start", s))
    for i in range(args.restarts):
        p = {}
        for name, (lo, hi, _) in BOUNDS.items():
            zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
            p[name] = natural(name, float(rng.uniform(zl + 0.05 * (zh - zl), zh - 0.05 * (zh - zl))))
        # random draws keep the measured mode frequencies and glide: a random mode set never lands in the basin
        for name in FREQ_KEYS:
            p[name] = MEASURED[name] * math.exp(rng.normal(0, 0.03))
        cands.append((f"random{i}", p))
    order = ORDER
    if args.only:
        order = [n for n in args.only.split(",") if n in BOUNDS]
        cands = [c for c in cands if c[0] == "start"]
        log(f"refining only {order} from --start")
    scored = sorted(((obj.evaluate(p), n, p) for n, p in cands), key=lambda x: x[0])
    for d, n, _ in scored[:8]:
        log(f"restart {n}: {d:.6f}")
    starts = [p for _, _, p in scored[:args.keep]]
    results = []
    for i, s in enumerate(starts):
        t0 = time.time()
        log(f"refine start {i}")
        p, d = coordinate_refine(obj, s, args.passes, args.steps, order=order, log=log)
        log(f"  start {i}: {d:.6f} improvement {1 - d / baseline:.2%} ({time.time() - t0:.0f}s)")
        results.append((d, p))
    results.sort(key=lambda x: x[0])
    d, p = results[0]
    if args.final_passes:
        log("final refine")
        p, d = coordinate_refine(obj, p, args.final_passes, 17, contraction=0.7, span_scale=0.3, order=order, log=log)
    y = render(p, frames, sr)
    pk = float(np.max(np.abs(y)))
    if pk > 0.9:
        y = y * np.float32(0.9 / pk)
    reference.write_wav(os.path.join(args.out, "learned.wav"), y, sr)
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    reference.write_wav(os.path.join(args.out, "ab.wav"), np.concatenate([target, gap, y, gap, target, gap, y]), sr)
    gate_learned = gate.evaluate(p)
    spec, harm, pooled = obj.evaluate(p, parts=True)
    pinned = [n for n in ORDER if min(abs(transformed(n, p[n]) - transformed(n, b)) for b in BOUNDS[n][:2] if (b != 0 or BOUNDS[n][2] == "linear")) < 1e-6]
    report = {"params": p, "learned": d, "baseline": baseline, "improvement": 1 - d / baseline,
              "train": {"spectral": spec, "harmonic": harm, "pooled": pooled, "harmonicWeight": args.harmonic_weight, "pooledWeight": args.pooled_weight},
              "gate": {"learned": gate_learned, "baseline": gate_baseline, "improvement": 1 - gate_learned / gate_baseline},
              "sampleRate": sr, "frames": frames, "pinned": pinned, "target": args.target,
              "bounds": {k: list(v[:2]) for k, v in BOUNDS.items()}, "boundOverrides": args.bound}
    with open(os.path.join(args.out, "recovered_params.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"RESULT train learned={d:.6f} (spectral {spec:.6f} lowband {harm:.6f} pooled {pooled:.6f}) baseline={baseline:.6f} improvement={1 - d / baseline:.2%} pinned={pinned}")
    log(f"GATE   learned={gate_learned:.6f} baseline={gate_baseline:.6f} improvement={1 - gate_learned / gate_baseline:.2%}")
    log("low-band peaks, target | synth:\n" + tracks.table(y))
    log(json.dumps({k: round(v, 5) for k, v in p.items()}))


if __name__ == "__main__":
    main()
