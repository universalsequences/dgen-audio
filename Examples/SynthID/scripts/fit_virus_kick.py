#!/usr/bin/env python3
"""SynthID Access Virus B BassDrum_23 voice (redo): NumPy reference renderer +
CPU fit against the independent MR-STFT metric (compare.py) plus an explicit
harmonic-track loss, for Assets/access-virus-b-bassdrum-23.wav.

Why this exists: the v1-v3 profile (Swift, sine sweep + decorative H2/H3/H5)
matched the fundamental and its envelope but missed the ladder of swept
harmonics that gives the sound its character. Measurement (harmonic
heterodyne against a zero-crossing phase track) shows harmonics 1..20 of the
sweep explain the target below 1 kHz to within 0.1 dB in every window; the
ladder is H2 ~ -25 dB, H3 ~ -40, H4 ~ -43, H5 ~ -50, H7 ~ -47 (late), each
with its own envelope, and the single-exponential pitch model sat 30-40 %
below the true pitch between 30 and 70 ms.

Topology (all documented scalars, no target-derived tables):
  phase   phi(t) = fEnd*t + a1/r1*(e^{r1 t}-1) + a2/r2*(e^{r2 t}-1)
  body    env(t) = attack(t; attackTime) * exp(ampDecay t + ampCurve t^2)
  bank    sum_k a_k * exp(d_k t) * sin(2 pi k frac(phi)),  k = 1..10
          (a_1 = bodyAmp, d_1 = 0; a_k, d_k free for k = 2..10)
  hiss    hissAmp * HP(noise, hissCutoff) -> fixed 16 kHz LP, * exp(hissDecay t)
  click   clickAmp * sin(2 pi clickFreq t) * exp(clickDecay t)   (capped: the target has no tick)
  noise   noiseAmp * LP(noise, noiseCutoff) * exp(noiseDecay t)
  out     tanh(drive * mix) / drive * outGain   (drive shapes, does not gain)

Fit protocol mirrors fit_clap.py: baseline = spec midpoints; restarts scored;
coordinate descent in transformed space with contraction. The gate stays
compare.py; training adds a 4096 window and the harmonic-track term.
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare
import render_reference as reference

NH = 10  # harmonics in the bank (1 = fundamental)

BOUNDS = {
    "fEnd": (40.0, 60.0, "log"),
    "a1": (400.0, 3000.0, "log"),
    "r1": (-400.0, -60.0, "logneg"),
    "a2": (50.0, 1500.0, "log"),
    "r2": (-150.0, -15.0, "logneg"),
    "attackTime": (0.003, 0.3, "log"),
    "ampDecay": (-40.0, -0.5, "logneg"),
    "ampCurve": (-100.0, 0.0, "linear"),
    "bodyAmp": (0.2, 2.5, "log"),
    "clickFreq": (300.0, 4000.0, "log"),
    "clickAmp": (0.0, 0.02, "linear"),   # the target has no tick: its first ms is a -40 dB wiggle
    "clickDecay": (-8000.0, -50.0, "logneg"),
    "noiseCutoff": (300.0, 12000.0, "log"),
    "noiseAmp": (0.0, 0.05, "linear"),
    "noiseDecay": (-800.0, -5.0, "logneg"),
    "hissCutoff": (2000.0, 12000.0, "log"),
    "hissAmp": (0.0, 0.01, "linear"),
    "hissDecay": (-60.0, -2.0, "logneg"),
    "drive": (0.05, 4.0, "log"),   # tanh must be allowed to go linear: the target's odd harmonics are not saturation products
    "outGain": (0.05, 5.0, "log"),
}
for k in range(2, NH + 1):
    BOUNDS[f"h{k}"] = (1e-4, 0.3, "log")      # level relative to bodyAmp (linear amplitude)
    BOUNDS[f"d{k}"] = (-80.0, 20.0, "linear")  # extra decay rate 1/s on top of the body envelope
ORDER = list(BOUNDS.keys())

# measurement-informed start (scratch analysis 2026-09-03, see HANDOFF)
MEASURED = {
    "fEnd": 48.06, "a1": 1457.7, "r1": -157.1, "a2": 573.2, "r2": -61.4,
    "attackTime": 0.03, "ampDecay": -8.0, "ampCurve": -30.0, "bodyAmp": 1.0,
    "clickFreq": 800.0, "clickAmp": 0.01, "clickDecay": -300.0,
    "noiseCutoff": 3000.0, "noiseAmp": 0.003, "noiseDecay": -40.0,
    "hissCutoff": 6000.0, "hissAmp": 0.001, "hissDecay": -11.0,
    "drive": 1.0, "outGain": 1.0,
    "h2": 0.056, "d2": -12.0, "h3": 0.012, "d3": -10.0, "h4": 0.007, "d4": -5.0,
    "h5": 0.0035, "d5": 0.0, "h6": 0.0016, "d6": -30.0, "h7": 0.004, "d7": 0.0,
    "h8": 0.0006, "d8": -10.0, "h9": 0.0006, "d9": -20.0, "h10": 0.0007, "d10": 0.0,
}
TRAIN_WINDOWS = (256, 512, 1024, 2048, 4096)
HARMONIC_WEIGHT = 0.3
HARMONIC_FLOOR = 1e-3   # -60 dBFS, same floor as the gate's log epsilon


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


# The Virus B runs at 32.5 kHz: nothing above 16.25 kHz leaves the machine. A
# fixed 4th-order 16 kHz lowpass on the hiss is a machine property, not a param.
HISS_LPF_HZ = 16000.0


@functools.lru_cache(maxsize=512)
def noise_hp(frames, sr, fc):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    y = biquad(n, "hp", fc, 0.707, sr)
    return biquad(biquad(y, "lp", HISS_LPF_HZ, 0.707, sr), "lp", HISS_LPF_HZ, 0.707, sr)


@functools.lru_cache(maxsize=512)
def noise_lp(frames, sr, fc):
    n = reference.dgen_noise(frames).astype(np.float64) * 2.0 - 1.0
    return biquad(n, "lp", fc, 0.707, sr)


def sweep_phase(p, t):
    return (p["fEnd"] * t
            + p["a1"] / p["r1"] * (np.exp(p["r1"] * t) - 1.0)
            + p["a2"] / p["r2"] * (np.exp(p["r2"] * t) - 1.0))


def render(p, frames, sr):
    t = np.arange(frames, dtype=np.float64) / sr
    phi = sweep_phase(p, t)
    frac = phi - np.floor(phi)
    tau = p["attackTime"]
    attack = (1.0 - np.exp(-t / tau)) / (1.0 - math.exp(-0.05 / tau))
    env = attack * np.exp(p["ampDecay"] * t + p["ampCurve"] * t * t)
    mix = np.sin(2.0 * np.pi * frac)
    for k in range(2, NH + 1):
        mix = mix + p[f"h{k}"] * np.exp(p[f"d{k}"] * t) * np.sin(2.0 * np.pi * k * frac)
    mix = mix * env * p["bodyAmp"]
    if p["hissAmp"] > 0:
        # recording/machine hiss: high-passed noise with its own slow decay
        # (target >4 kHz RMS: -67 dBFS at onset, -74 at 100 ms, -85 at 200 ms)
        mix = mix + p["hissAmp"] * noise_hp(frames, sr, round(p["hissCutoff"], 6)) * np.exp(p["hissDecay"] * t)
    if p["clickAmp"] > 0:
        mix = mix + p["clickAmp"] * np.sin(2.0 * np.pi * p["clickFreq"] * t) * np.exp(p["clickDecay"] * t)
    if p["noiseAmp"] > 0:
        mix = mix + p["noiseAmp"] * noise_lp(frames, sr, round(p["noiseCutoff"], 6)) * np.exp(p["noiseDecay"] * t)
    # gain-normalised saturator: drive sets the shape only, so the fit can trade
    # saturation harmonics against the bank without moving the level
    y = np.tanh(p["drive"] * mix) / p["drive"] * p["outGain"]
    return y.astype(np.float32)


def zero_crossing_phase(x, sr):
    """Deterministic phase track from positive zero crossings of a 25 Hz..2.5 kHz band copy."""
    n = len(x)
    X = np.fft.rfft(x)
    F = np.fft.rfftfreq(n, 1.0 / sr)
    xl = np.fft.irfft(X * ((F > 25) & (F < 2500)), n)
    zc = np.where((xl[:-1] <= 0) & (xl[1:] > 0))[0]
    zc = zc + (-xl[zc]) / (xl[zc + 1] - xl[zc])
    per = np.diff(zc) / sr
    tc = (zc[:-1] + zc[1:]) / 2 / sr
    f = 1.0 / per
    ok = (f > 30) & (f < 3000)
    t = np.arange(n) / sr
    f0 = np.interp(t, tc[ok], f[ok])
    return f0, np.cumsum(f0) / sr


class HarmonicTracks:
    """Heterodyne amplitude of harmonics 1..NH along the target's own phase track,
    sampled every 2 ms with a two-period Hann window. Used both as a training term
    and as the diagnostic that the gate metric lacks."""
    def __init__(self, target, sr, t_max=0.30):
        n = len(target)
        self.sr = sr
        self.f0, phi = zero_crossing_phase(target, sr)
        self.times = np.arange(0.003, min(t_max, n / sr - 0.01), 0.002)
        W = np.zeros((len(self.times), n))
        for j, ti in enumerate(self.times):
            i = int(ti * sr)
            w = int(np.clip(2.0 / self.f0[i], 0.0015, 0.045) * sr)
            a, b = max(0, i - w // 2), min(n, i + w // 2)
            ker = np.hanning(b - a)
            W[j, a:b] = ker / ker.sum()
        self.W = W
        self.demod = np.stack([np.exp(-2j * np.pi * k * phi) for k in range(1, NH + 1)])
        self.target = self.tracks(target)

    def tracks(self, x):
        return 2.0 * np.abs((self.demod * x[None, :]) @ self.W.T)  # (NH, T)

    def loss(self, x):
        return float(np.mean(np.abs(np.log(self.tracks(x) + HARMONIC_FLOOR) - np.log(self.target + HARMONIC_FLOOR))))

    def table(self, x, times_ms=(4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250)):
        A = 20 * np.log10(self.tracks(x) + 1e-7)
        B = 20 * np.log10(self.target + 1e-7)
        lines = ["  ms   f0  " + " ".join(f"H{k:<6}" for k in range(1, NH + 1)) + "   (synth-target dB)"]
        for ms in times_ms:
            j = int(np.argmin(np.abs(self.times - ms / 1000)))
            lines.append(f"{ms:4d} {self.f0[int(self.times[j] * self.sr)]:5.0f}  "
                         + " ".join(f"{A[k, j] - B[k, j]:+6.1f}" for k in range(NH)))
        return "\n".join(lines)


POOLED_WINDOWS = (256, 1024)
POOLED_EPSILON = 1e-6   # -120 dB per bin: the -70 dBFS hiss is ~-90 dB per bin
POOLED_MIN_HZ = 2500.0  # the pooled term is the hiss objective: high bands only
POOLED_WEIGHT = 0.3


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
        # band-pooled log power removes the per-bin Rayleigh variance of the
        # noise, so this term follows the hiss's spectral shape and envelope
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
    ap.add_argument("--target", default="Assets/access-virus-b-bassdrum-23.wav")
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
            # a seed outside a tightened bound must be clamped, or the search keeps it
            s[name] = natural(name, min(max(transformed(name, s[name]), zl), zh))
        cands.append(("start", s))
    for i in range(args.restarts):
        p = {}
        for name, (lo, hi, _) in BOUNDS.items():
            zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
            p[name] = natural(name, float(rng.uniform(zl + 0.05 * (zh - zl), zh - 0.05 * (zh - zl))))
        # random draws keep the measured pitch curve: a random sweep never lands near the basin
        for name in ("fEnd", "a1", "r1", "a2", "r2"):
            p[name] = MEASURED[name] * math.exp(rng.normal(0, 0.05))
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
              "sampleRate": sr, "frames": frames, "pinned": pinned, "target": args.target}
    with open(os.path.join(args.out, "recovered_params.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"RESULT train learned={d:.6f} (spectral {spec:.6f} harmonic {harm:.6f} pooled {pooled:.6f}) baseline={baseline:.6f} improvement={1 - d / baseline:.2%} pinned={pinned}")
    log(f"GATE   learned={gate_learned:.6f} baseline={gate_baseline:.6f} improvement={1 - gate_learned / gate_baseline:.2%}")
    log("harmonic tracks, synth minus target (dB):\n" + tracks.table(y))
    log(json.dumps({k: round(v, 5) for k, v in p.items()}))


if __name__ == "__main__":
    main()
