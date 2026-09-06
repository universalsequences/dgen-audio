#!/usr/bin/env python3
"""SynthID E-mu Orbit-9090 tom (sample "66.wav", tags EMU / EMU Orbit-9090):
NumPy reference renderer + CPU fit against the independent MR-STFT metric
(compare.py) plus a partial-track loss, for Assets/emu-orbit9090-tom-66.wav.

Measurement (2026-09-03): 44.1 kHz 16-bit mono, 222 ms, peak -0.7 dBFS, 1.5 ms
of silence before the onset. The instantaneous frequency sweeps from ~3.5 kHz
to ~300 Hz in the first 8 ms and settles at ~155 Hz by 40 ms; from 22 ms on it
wobbles with an 11.5 ms period. Long-window spectra of the tail show partials
at 68 / 155 / 242 / 329 / 415 / 502 Hz: evenly spaced by ~87 Hz but offset from
zero, i.e. FM sidebands fc + n*fm (fc ~155, fm ~87, index ~1.3), with the
upper sidebands falling -3 / -11 / -21 / -32 dB and one lower sideband at
-4 dB. Only one lower sideband is visible, so the bank is free rather than
Bessel-locked. The spacing itself drifts 92 -> 87 Hz while the carrier
settles 177 -> 155 Hz (25 -> 80 ms).

Topology (all documented scalars, no target-derived tables):
  clock   t' = t - onset, gated at t >= onset
  carrier phic(t') = cEnd t' + ca1/cr1 (e^{cr1 t'} - 1) + ca2/cr2 (e^{cr2 t'} - 1)
  mod     phim(t') = mEnd t' + ma1/mr1 (e^{mr1 t'} - 1)
  bank    sum_k h_k e^{d_k t'} sin(2 pi frac(phic + (k-1) phim + p_k)),  k = 0..6
          (k = 1 is the carrier: h_1 = bodyAmp, d_1 = 0, p_1 = 0;
           k = 0 is the lower sideband fc - fm, k >= 2 the upper ones)
  body    env = attack(t'; attackTime) * exp(ampDecay t' + ampCurve t'^2)
  click   clickAmp sin(2 pi clickFreq t') e^{clickDecay t'}   (capped)
  noise   noiseAmp LP(noise, noiseCutoff) e^{noiseDecay t'}    (capped)
  out     tanh(drive * mix) / drive * outGain
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare
import render_reference as reference

PARTIALS = (0, 2, 3, 4, 5, 6)   # bank members besides the carrier (k = 1)
NH = 7                           # tracks: k = 0..6

BOUNDS = {
    "onset": (0.0, 0.004, "linear"),
    "cEnd": (100.0, 250.0, "log"),
    "ca1": (500.0, 8000.0, "log"),
    "cr1": (-1200.0, -100.0, "logneg"),
    "hold": (0.0, 0.006, "linear"),
    "ca2": (10.0, 1200.0, "log"),
    "cr2": (-250.0, -10.0, "logneg"),
    "ratio": (0.3, 0.9, "log"),          # modulator / carrier, ratio-locked (FM-tom style)
    "ma1": (0.1, 600.0, "log"),          # extra modulator-only fall on top of ratio x carrier
    "mr1": (-1500.0, -20.0, "logneg"),
    "attackTime": (0.0002, 0.03, "log"),
    "sbAttack": (0.0002, 0.03, "log"),   # sideband (modulation index) rise time; 0.0002 = instant
    "ampDecay": (-40.0, -0.5, "logneg"),
    "ampCurve": (-150.0, 0.0, "linear"),
    "bodyAmp": (0.1, 2.5, "log"),
    "clickFreq": (300.0, 6000.0, "log"),
    "clickAmp": (0.0, 0.02, "linear"),
    "clickDecay": (-8000.0, -300.0, "logneg"),
    "noiseCutoff": (300.0, 12000.0, "log"),
    "noiseAmp": (0.0, 0.15, "linear"),
    "noiseDecay": (-2000.0, -100.0, "logneg"),
    "drive": (0.05, 4.0, "log"),
    "outGain": (0.05, 5.0, "log"),
}
for k in PARTIALS:
    BOUNDS[f"h{k}"] = (1e-4, 1.5, "log")       # level relative to the carrier
    BOUNDS[f"d{k}"] = (-80.0, 20.0, "linear")  # extra decay 1/s on top of the body envelope
    BOUNDS[f"p{k}"] = (0.0, 1.0, "linear")     # phase offset, cycles
ORDER = list(BOUNDS.keys())

# measurement-informed start (scratch analysis 2026-09-03, see HANDOFF)
MEASURED = {
    # v1 winner (gate 0.1555) with the measured onset plateau put back: the
    # instantaneous frequency sits at ~2.9 kHz from 2 to 4.5 ms, falls to
    # 330 Hz by 9 ms (r ~ -490/s), then drifts 300 -> 155 Hz over 30 ms.
    # v4: carrier from a direct least-squares pre-fit of the instantaneous-
    # frequency track (output/emu_tom_carrier_prefit.json), modulator locked
    # to the carrier at the tail's 87/155 ratio (sideband spacing / carrier is
    # ~0.65 at 5 ms, ~0.5-0.56 from 8 ms on: an FM tom, both operators on one
    # pitch envelope).
    "onset": 0.0015, "hold": 0.004,
    "cEnd": 154.0, "ca1": 2033.0, "cr1": -1120.0, "ca2": 482.0, "cr2": -81.0,
    "ratio": 0.561, "ma1": 0.39, "mr1": -918.0, "sbAttack": 0.004,
    "attackTime": 0.0008, "ampDecay": -10.3, "ampCurve": -43.0, "bodyAmp": 0.77,
    "clickFreq": 3000.0, "clickAmp": 0.0, "clickDecay": -1000.0,
    "noiseCutoff": 4000.0, "noiseAmp": 0.0, "noiseDecay": -300.0,
    "drive": 1.05, "outGain": 0.66,
    "h0": 1.15, "d0": -4.8, "p0": 0.35,
    "h2": 0.75, "d2": -1.0, "p2": 0.043,
    "h3": 0.27, "d3": -0.7, "p3": 0.105,
    "h4": 0.082, "d4": -0.5, "p4": 0.16,
    "h5": 0.023, "d5": 0.14, "p5": 0.18,
    "h6": 0.012, "d6": -1.4, "p6": 0.23,
}
SWEEP_KEYS = ("cEnd", "ca1", "cr1", "hold", "ca2", "cr2", "ratio")
SWEEP_JITTER = 0.15   # random restarts: log-normal jitter on the sweep keys (was 0.05; the coupled sweep needs the spread)
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




def held_fall(a, r, hold, t):
    """Phase of a frequency term that sits at `a` for `hold` seconds, then
    falls as a e^{r (t - hold)}: a*t before the hold ends, a*hold + a/r (e^{r(t-hold)} - 1) after."""
    th = np.maximum(t - hold, 0.0)
    return a * np.minimum(t, hold) + a / r * (np.exp(r * th) - 1.0)


def sweep_phase(p, t):
    return (p["cEnd"] * t
            + held_fall(p["ca1"], p["cr1"], p["hold"], t)
            + p["ca2"] / p["cr2"] * (np.exp(p["cr2"] * t) - 1.0))


def mod_phase(p, t, phic=None):
    if phic is None:
        phic = sweep_phase(p, t)
    return p["ratio"] * phic + p["ma1"] / p["mr1"] * (np.exp(p["mr1"] * t) - 1.0)


def local_time(p, frames, sr):
    t = np.arange(frames, dtype=np.float64) / sr - p["onset"]
    return np.maximum(t, 0.0), (t >= 0.0).astype(np.float64)


def render(p, frames, sr):
    t, gate = local_time(p, frames, sr)
    phic = sweep_phase(p, t)
    phim = mod_phase(p, t, phic)
    tau = p["attackTime"]
    attack = (1.0 - np.exp(-t / tau)) / (1.0 - math.exp(-0.05 / tau))
    env = gate * attack * np.exp(p["ampDecay"] * t + p["ampCurve"] * t * t)
    frac = phic - np.floor(phic)
    mix = np.sin(2.0 * np.pi * frac)
    # the sidebands grow in over the first milliseconds (the modulation index
    # rises; the sample's first cycles are nearly a bare sine)
    sb_rise = 1.0 - np.exp(-t / p["sbAttack"])
    for k in PARTIALS:
        ph = phic + (k - 1) * phim + p[f"p{k}"]
        mix = mix + sb_rise * p[f"h{k}"] * np.exp(p[f"d{k}"] * t) * np.sin(2.0 * np.pi * (ph - np.floor(ph)))
    mix = mix * env * p["bodyAmp"]
    if p["clickAmp"] > 0:
        mix = mix + gate * p["clickAmp"] * np.sin(2.0 * np.pi * p["clickFreq"] * t) * np.exp(p["clickDecay"] * t)
    if p["noiseAmp"] > 0:
        mix = mix + gate * p["noiseAmp"] * noise_lp(frames, sr, round(p["noiseCutoff"], 6)) * np.exp(p["noiseDecay"] * t)
    y = np.tanh(p["drive"] * mix) / p["drive"] * p["outGain"]
    return y.astype(np.float32)


class HarmonicTracks:
    """Heterodyne amplitude of the bank's partials k = 0..6 along the MEASURED
    seed's carrier/modulator phase tracks (phic + (k-1) phim), sampled every
    2 ms with a window of two modulator periods (the sidebands are fm apart).
    The tracks come from the documented seed, not from the target, so target
    and synth are demodulated identically. Training term + the diagnostic the
    gate metric lacks."""
    def __init__(self, target, sr, t_max=0.30, params=None):
        n = len(target)
        self.sr = sr
        p = dict(MEASURED if params is None else params)
        t, _ = local_time(p, n, sr)
        phic = sweep_phase(p, t)
        phim = mod_phase(p, t, phic)
        self.f0 = p["cEnd"] + p["ca1"] * np.exp(p["cr1"] * np.maximum(t - p["hold"], 0.0)) + p["ca2"] * np.exp(p["cr2"] * t)
        fm = p["ratio"] * self.f0 + p["ma1"] * np.exp(p["mr1"] * t)
        self.times = np.arange(0.006, min(t_max, n / sr - 0.01), 0.002)
        W = np.zeros((len(self.times), n))
        for j, ti in enumerate(self.times):
            i = int(ti * sr)
            w = int(np.clip(2.0 / fm[i], 0.004, 0.045) * sr)
            a, b = max(0, i - w // 2), min(n, i + w // 2)
            ker = np.hanning(b - a)
            W[j, a:b] = ker / ker.sum()
        self.W = W
        self.demod = np.stack([np.exp(-2j * np.pi * (phic + (k - 1) * phim)) for k in range(0, NH)])
        self.target = self.tracks(target)

    def tracks(self, x):
        return 2.0 * np.abs((self.demod * x[None, :]) @ self.W.T)  # (NH, T)

    def loss(self, x):
        return float(np.mean(np.abs(np.log(self.tracks(x) + HARMONIC_FLOOR) - np.log(self.target + HARMONIC_FLOOR))))

    def table(self, x, times_ms=(6, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200)):
        A = 20 * np.log10(self.tracks(x) + 1e-7)
        B = 20 * np.log10(self.target + 1e-7)
        lines = ["  ms   fc  " + " ".join(f"P{k:<6}" for k in range(0, NH)) + "   (synth-target dB; P1 = carrier, P0 = fc-fm, Pk = fc+(k-1)fm)"]
        for ms in times_ms:
            j = int(np.argmin(np.abs(self.times - ms / 1000)))
            lines.append(f"{ms:4d} {self.f0[int(self.times[j] * self.sr)]:5.0f}  "
                         + " ".join(f"{A[k, j] - B[k, j]:+6.1f}" for k in range(NH)))
        lines.append("target dBFS: " + " | ".join(f"{ms}ms " + " ".join(f"{B[k, int(np.argmin(np.abs(self.times - ms / 1000)))]:.0f}" for k in range(NH)) for ms in (12, 30, 60, 120)))
        return "\n".join(lines)


POOLED_WINDOWS = (256, 1024)
POOLED_EPSILON = 1e-6   # -120 dB per bin: the -70 dBFS hiss is ~-90 dB per bin
POOLED_MIN_HZ = 2500.0  # the pooled term is the hiss objective: high bands only
POOLED_WEIGHT = 0.0   # no hiss on this target: the pooled term only chased the 16-bit floor (v2)


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
    ap.add_argument("--target", default="Assets/emu-orbit9090-tom-66.wav")
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
        for name in SWEEP_KEYS:
            p[name] = MEASURED[name] * math.exp(rng.normal(0, SWEEP_JITTER))
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
