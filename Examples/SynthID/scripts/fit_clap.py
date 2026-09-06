#!/usr/bin/env python3
"""SynthID clap voice: NumPy reference renderer + CPU fit against the
independent MR-STFT metric (compare.py), for Assets/808-clap-r8.wav.

Topology (all documented scalars, no target-derived tables):
  noise (DGen xorshift, [-1,1)) -> two RBJ bandpasses (fc1/q1, fc2/q2, mix g2)
  burst train: 4 onsets at 0, sp1, sp1+sp2, sp1+sp2+sp3 ms; levels 1,l2,l3,l4;
    each burst an exponential (bDecay) plus a sub-burst echo (subDelay, subGain)
  tail from the last onset: fast exponential (tA1,d1) on the bandpassed source
    + slow exponential (tA2,d2) on a lowpassed (tailLpf) copy
  tanh(drive * x) * outGain

Fit protocol mirrors refine_rung3.py: baseline = spec midpoints; restarts are
scored by the metric; coordinate descent in transformed space with contraction.
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare
import render_reference as reference

BOUNDS = {
    "fc1": (300.0, 3000.0, "log"),
    "q1": (0.5, 6.0, "log"),
    "fc2": (800.0, 6000.0, "log"),
    "q2": (0.5, 6.0, "log"),
    "g2": (0.0, 4.0, "linear"),
    "sp1": (4.0, 16.0, "log"),
    "sp2": (4.0, 16.0, "log"),
    "sp3": (4.0, 16.0, "log"),
    "bDecay": (-1500.0, -100.0, "logneg"),
    "l2": (0.2, 3.0, "linear"),
    "l3": (0.2, 1.5, "linear"),
    "l4": (0.0, 1.5, "linear"),
    "subDelay": (1.5, 6.0, "log"),
    "subGain": (0.0, 2.0, "linear"),
    "burstAmp": (0.05, 20.0, "log"),
    "tA1": (0.0, 10.0, "linear"),
    "d1": (-150.0, -10.0, "logneg"),
    "tA2": (0.0, 10.0, "linear"),
    "d2": (-40.0, -4.0, "logneg"),
    "tailLpf": (500.0, 12000.0, "log"),
    "hpFc": (500.0, 10000.0, "log"),
    "bHp": (0.0, 1.5, "linear"),
    "tHp": (0.0, 1.5, "linear"),
    "outHp": (20.0, 2000.0, "log"),
    "drive": (0.5, 4.0, "log"),
    "outGain": (0.05, 2.0, "log"),
}
ORDER = ["fc1", "q1", "fc2", "q2", "g2", "sp1", "sp2", "sp3", "bDecay", "l2", "l3", "l4",
         "subDelay", "subGain", "burstAmp", "tA1", "d1", "tA2", "d2", "tailLpf", "hpFc", "bHp", "tHp", "outHp", "drive", "outGain"]

# measurement-informed start (scripts/analysis/analyze_808_clap.py)
MEASURED = {
    "fc1": 1000.0, "q1": 1.5, "fc2": 2100.0, "q2": 2.0, "g2": 0.5,
    "sp1": 9.7, "sp2": 11.3, "sp3": 8.3, "bDecay": -500.0, "l2": 0.75, "l3": 0.65, "l4": 0.7,
    "subDelay": 3.5, "subGain": 0.3, "burstAmp": 4.0,
    "tA1": 2.4, "d1": -36.0, "tA2": 1.2, "d2": -15.5, "tailLpf": 3000.0,
    "drive": 1.0, "outGain": 1.0, "hpFc": 3500.0, "bHp": 0.3, "tHp": 0.3, "outHp": 300.0,
}
TRAIN_WINDOWS = (64, 128, 256, 512, 1024, 2048)
# The R-8 MkII plays its samples at 26.04 kHz: nothing above 13 kHz leaves the
# machine. A fixed 4th-order output lowpass (two Q=0.707 biquads) is part of
# the voice, not a fitted parameter.
OUTPUT_LPF_HZ = 12000.0


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
    # RBJ cookbook; mode "bp" = constant 0 dB peak gain bandpass (DGen mode 2),
    # "lp" = lowpass (DGen mode 0).
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
    xl = x.tolist()
    for i, v in enumerate(xl):
        out = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = out
        x2, x1 = x1, v
        y2, y1 = y1, out
    return y


@functools.lru_cache(maxsize=512)
def noise_bp(frames, sr, fc, q):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    return biquad(n, "bp", fc, q, sr)


@functools.lru_cache(maxsize=512)
def noise_hp(frames, sr, fc):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    return biquad(n, "hp", fc, 0.707, sr)


@functools.lru_cache(maxsize=512)
def noise_bp_lp(frames, sr, fc, q, lpf):
    return biquad(noise_bp(frames, sr, fc, q), "lp", lpf, 0.707, sr)


def render(p, frames, sr):
    t = np.arange(frames, dtype=np.float64) / sr
    src = noise_bp(frames, sr, round(p["fc1"], 6), round(p["q1"], 6))
    if p["g2"] > 0:
        src = src + p["g2"] * noise_bp(frames, sr, round(p["fc2"], 6), round(p["q2"], 6))
    onsets = [0.0, p["sp1"], p["sp1"] + p["sp2"], p["sp1"] + p["sp2"] + p["sp3"]]
    levels = [1.0, p["l2"], p["l3"], p["l4"]]
    env_b = np.zeros(frames)
    for tk, lk in zip(onsets, levels):
        for delay, g in ((0.0, 1.0), (p["subDelay"], p["subGain"])):
            if g <= 0:
                continue
            dt = t - (tk + delay) * 1e-3
            env_b += lk * g * np.where(dt >= 0, np.exp(p["bDecay"] * np.maximum(dt, 0)), 0.0)
    t0 = onsets[-1] * 1e-3
    dt = np.maximum(t - t0, 0.0)
    on = (t >= t0).astype(np.float64)
    tail_fast = on * p["tA1"] * np.exp(p["d1"] * dt)
    tail_slow = on * p["tA2"] * np.exp(p["d2"] * dt)
    lp_src = noise_bp_lp(frames, sr, round(p["fc1"], 6), round(p["q1"], 6), round(p["tailLpf"], 6))
    hp_src = noise_hp(frames, sr, round(p["hpFc"], 6)) if (p["bHp"] > 0 or p["tHp"] > 0) else 0.0
    x = (src + p["bHp"] * hp_src) * env_b * p["burstAmp"] + (src + p["tHp"] * hp_src) * tail_fast + lp_src * tail_slow
    y = np.tanh(p["drive"] * x) * p["outGain"]
    y = biquad(biquad(y, "lp", OUTPUT_LPF_HZ, 0.707, sr), "lp", OUTPUT_LPF_HZ, 0.707, sr)
    y = biquad(y, "hp", p["outHp"], 0.707, sr)
    return y.astype(np.float32)


class Objective:
    def __init__(self, target, sr, highpass_hz, windows=compare.WINDOWS, pooled=False, bands=32):
        self.sr, self.frames, self.hp, self.windows = sr, len(target), highpass_hz, windows
        self.pooled = pooled
        self.target = compare.capture_highpass(target, sr, highpass_hz)
        self.pool = {}
        if pooled:
            edges = np.geomspace(150.0, min(20000.0, sr / 2), bands + 1)
            for w in windows:
                f = np.fft.rfftfreq(w, 1.0 / sr)
                idx = np.searchsorted(edges, f, side="right") - 1
                m = np.zeros((len(f), bands))
                ok = (idx >= 0) & (idx < bands)
                m[np.arange(len(f))[ok], idx[ok]] = 1.0
                m = m[:, m.sum(axis=0) > 0]           # drop bands with no bins at this window
                self.pool[w] = m / np.maximum(m.sum(axis=0, keepdims=True), 1)
        self.feat = {w: self.features(self.target, w) for w in windows}
        self.evals = 0

    def features(self, signal, w):
        hop = w // 4
        win = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, w)[::hop]
        scale = max(float(win.sum()) / 2.0, 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1)) / scale
        if self.pooled:
            # band-pooled log power: averaging bins inside a band removes most
            # of the per-bin Rayleigh variance of a noise signal, so the loss
            # follows the spectral envelope and time envelope instead of the
            # particular noise realization.
            power = (mag ** 2) @ self.pool[w]
            return 0.5 * np.log(power + compare.LOG_EPSILON ** 2)
        return np.log(mag + compare.LOG_EPSILON)

    def distance(self, signal):
        return sum(float(np.mean(np.abs(self.features(signal, w) - self.feat[w]))) for w in self.windows)

    def evaluate(self, p):
        self.evals += 1
        y = render(p, self.frames, self.sr)
        pk = float(np.max(np.abs(y)))
        if pk > 0.9:
            y = y * np.float32(0.9 / pk)
        return self.distance(compare.capture_highpass(y, self.sr, self.hp))


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
    ap.add_argument("--target", default="Assets/808-clap-r8.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--final-passes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", help="params json to refine from (skips restarts)")
    ap.add_argument("--highpass-hz", type=float, default=compare.DEFAULT_HIGHPASS_HZ)
    ap.add_argument("--gate-windows", action="store_true", help="train on the gate's 4 windows only")
    args = ap.parse_args()
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
    obj = Objective(target, sr, args.highpass_hz, compare.WINDOWS if args.gate_windows else TRAIN_WINDOWS, pooled=not args.gate_windows)
    gate = Objective(target, sr, args.highpass_hz, compare.WINDOWS)
    base = midpoints()
    baseline = obj.evaluate(base)
    gate_baseline = gate.evaluate(base)
    reference.write_wav(os.path.join(args.out, "initial.wav"), render(base, frames, sr), sr)
    log(f"frames={frames} sr={sr} baseline(midpoints)={baseline:.6f}")

    rng = np.random.default_rng(args.seed)
    if False:
        pass
    else:
        cands = [("measured", dict(MEASURED)), ("midpoint", dict(base))]
        if args.start:
            with open(args.start) as f:
                s = json.load(f)
            s = s.get("params", s)
            for name in BOUNDS:
                s.setdefault(name, base[name])
            cands.append(("start", s))
        for i in range(args.restarts):
            p = {}
            for name, (lo, hi, _) in BOUNDS.items():
                zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
                p[name] = natural(name, float(rng.uniform(zl + 0.05 * (zh - zl), zh - 0.05 * (zh - zl))))
            cands.append((f"random{i}", p))
        scored = sorted(((obj.evaluate(p), n, p) for n, p in cands), key=lambda x: x[0])
        for d, n, _ in scored[:8]:
            log(f"restart {n}: {d:.6f}")
        starts = [p for _, _, p in scored[:args.keep]]
    results = []
    for i, s in enumerate(starts):
        t0 = time.time()
        log(f"refine start {i}")
        p, d = coordinate_refine(obj, s, args.passes, args.steps, log=log)
        log(f"  start {i}: {d:.6f} improvement {1 - d / baseline:.2%} ({time.time() - t0:.0f}s)")
        results.append((d, p))
    results.sort(key=lambda x: x[0])
    d, p = results[0]
    if args.final_passes:
        log("final refine")
        p, d = coordinate_refine(obj, p, args.final_passes, 17, contraction=0.7, span_scale=0.3, log=log)
    y = render(p, frames, sr)
    pk = float(np.max(np.abs(y)))
    if pk > 0.9:
        y = y * np.float32(0.9 / pk)
    reference.write_wav(os.path.join(args.out, "learned.wav"), y, sr)
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    reference.write_wav(os.path.join(args.out, "ab.wav"), np.concatenate([target, gap, y, gap, target, gap, y]), sr)
    gate_learned = gate.evaluate(p)
    pinned = [n for n in ORDER if min(abs(transformed(n, p[n]) - transformed(n, b)) for b in BOUNDS[n][:2] if (b != 0 or BOUNDS[n][2] == "linear")) < 1e-6]
    report = {"params": p, "learned": d, "baseline": baseline, "improvement": 1 - d / baseline,
              "gate": {"learned": gate_learned, "baseline": gate_baseline, "improvement": 1 - gate_learned / gate_baseline},
              "sampleRate": sr, "frames": frames, "pinned": pinned, "target": args.target}
    with open(os.path.join(args.out, "recovered_params.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"RESULT train learned={d:.6f} baseline={baseline:.6f} improvement={1 - d / baseline:.2%} pinned={pinned}")
    log(f"GATE   learned={gate_learned:.6f} baseline={gate_baseline:.6f} improvement={1 - gate_learned / gate_baseline:.2%}")
    log(json.dumps({k: round(v, 4) for k, v in p.items()}))


if __name__ == "__main__":
    main()
