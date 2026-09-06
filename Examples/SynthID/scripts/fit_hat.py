#!/usr/bin/env python3
"""SynthID 909 open-hat voice: NumPy reference renderer + CPU fit against the
independent MR-STFT metric (compare.py), for Assets/909-open-hat.wav
(TR-909 HHOD0: the hat ROM played through the open-hat VCA at decay 0).

Topology (all documented scalars, no target-derived tables):
  noise (DGen xorshift, [-1,1)) ->
    broad: (RBJ bandpass fc1/q1 + gHp * highpass hpFc) * exp(swAmp * lowpass(noise, swRate))
           + g2 * bandpass fc2/q2
    modes: sum_k mkg * exp(mkd t) * sin(2 pi mkf t)   (12 struck-once metal modes)
  envelope: ramp = min(1, t/atk); hold-then-decay exp(d * max(t - hold, 0))
    broad: ramp * (exp(dTail*.) + aFast * exp(dFast * t))     * aB
    modes: ramp *  exp(dMode*.)
  + clickAmp * exp(clickDecay t) * lowpass(noise, clickFc)   (onset thump)
  tanh(drive * x) / drive * outGain -> highpass outHp

Fit protocol mirrors fit_clap.py: baseline = spec midpoints; restarts are
scored by the band-pooled log-power loss (random draws keep the measured mode
frequencies); coordinate descent in transformed space with contraction.
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare
import render_reference as reference

MODE_SEEDS = [647.3, 1486.2, 1548.4, 3407.4, 3940.8, 4272.3, 5023.1, 5225.1, 8311.5, 9207.4, 12552.7, 13433.0]
N_MODES = len(MODE_SEEDS)

BOUNDS = {
    "fc1": (1500.0, 12000.0, "log"),
    "q1": (0.3, 4.0, "log"),
    "fc2": (300.0, 16000.0, "log"),
    "q2": (0.1, 4.0, "log"),
    "g2": (0.0, 4.0, "linear"),
    "hpFc": (2000.0, 20000.0, "log"),
    "gHp": (0.0, 4.0, "linear"),
    "aB": (0.05, 20.0, "log"),
    "atk": (0.1, 10.0, "log"),
    "hold": (1.0, 80.0, "log"),
    "dTail": (-60.0, -4.0, "logneg"),
    "aFast": (0.0, 20.0, "linear"),
    "dFast": (-200.0, -15.0, "logneg"),
    "dMode": (-60.0, -2.0, "logneg"),
    "swRate": (3.0, 60.0, "log"),             # swish: slow undulation of the high wash (beating partials), zero-default
    "swAmp": (0.0, 2000.0, "linear"),
    "clickAmp": (0.0, 2.0, "linear"),         # onset thump: lowpassed noise burst, zero-default
    "clickDecay": (-3000.0, -600.0, "logneg"),
    "clickFc": (150.0, 1500.0, "log"),
    "outHp": (20.0, 2000.0, "log"),
    "drive": (0.02, 4.0, "log"),
    "outGain": (0.05, 2.0, "log"),
}
for k, f in enumerate(MODE_SEEDS, 1):
    BOUNDS[f"m{k}f"] = (f / 1.005, f * 1.005, "log")   # measured to 1 Hz; no spectral-average loss can rank where a sine sits, the measurement is the information
    BOUNDS[f"m{k}d"] = (-80.0, -1.0, "logneg")   # the mode's own free ring (the VCA decay is on top)
    BOUNDS[f"m{k}g"] = (0.001, 3.0, "log")
ORDER = list(BOUNDS)

# measurement-informed start (scripts/analysis/analyze_909_open_hat.py)
MEASURED = {
    "fc1": 6000.0, "q1": 0.7, "fc2": 1300.0, "q2": 1.5, "g2": 0.6, "hpFc": 9000.0, "gHp": 0.5,
    "aB": 1.0, "atk": 2.0, "hold": 20.0, "dTail": -16.0, "aFast": 0.5, "dFast": -40.0, "dMode": -12.0,
    "swRate": 15.0, "swAmp": 0.0,
    "clickAmp": 0.0, "clickDecay": -800.0, "clickFc": 400.0,
    "outHp": 200.0, "drive": 1.0, "outGain": 1.0,
}
for k, f in enumerate(MODE_SEEDS, 1):
    MEASURED[f"m{k}f"] = f
    MEASURED[f"m{k}d"] = -12.0
    MEASURED[f"m{k}g"] = 0.05
TRAIN_WINDOWS = (64, 128, 256, 512, 1024, 2048, 4096)
POOL_BANDS = 256         # cap; per-window count = clamp(w/16, 32, cap): wide enough to pool Rayleigh variance, narrow enough to see a metal mode
NOISE_SKIP = 0           # set >0 to render with a different noise realisation (metric floor)


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
    xl = x.tolist()
    for i, v in enumerate(xl):
        out = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = out
        x2, x1 = x1, v
        y2, y1 = y1, out
    return y


@functools.lru_cache(maxsize=8)
def noise(frames, skip):
    return reference.dgen_noise(frames + skip).astype(np.float64)[skip:] * 2.0 - 1.0


@functools.lru_cache(maxsize=2048)
def noise_bp(frames, sr, fc, q, skip):
    return biquad(noise(frames, skip), "bp", fc, q, sr)


@functools.lru_cache(maxsize=256)
def noise_lp(frames, sr, fc, skip):
    return biquad(noise(frames, skip), "lp", fc, 0.707, sr)


@functools.lru_cache(maxsize=256)
def noise_slow(frames, sr, fc, skip):
    """Two cascaded one-pole lowpasses (k = 1 - exp(-2 pi fc / sr)) on the
    noise. Not a biquad: at a few Hz a biquad's 1 - cos(w0) is one float32 ulp
    and the dgen runtime's coefficients are wrong by tens of percent."""
    k = 1.0 - math.exp(-2.0 * math.pi * fc / sr)
    x = noise(frames, skip); y = np.empty(frames); a = b = 0.0
    for i, v in enumerate(x.tolist()):
        a += k * (v - a); b += k * (a - b); y[i] = b
    return y


@functools.lru_cache(maxsize=256)
def noise_hp(frames, sr, fc, skip):
    return biquad(noise(frames, skip), "hp", fc, 0.707, sr)


def render(p, frames, sr):
    skip = NOISE_SKIP
    t = np.arange(frames, dtype=np.float64) / sr
    r6 = lambda v: round(v, 6)
    broad = noise_bp(frames, sr, r6(p["fc1"]), r6(p["q1"]), skip)
    if p["gHp"] > 0:
        broad = broad + p["gHp"] * noise_hp(frames, sr, r6(p["hpFc"]), skip)
    if p["swAmp"] > 0:
        # the swish: the high wash undulates like beating cymbal partials —
        # exp(swAmp * lowpassed noise) is a positive, dB-scale slow modulator
        broad = broad * np.exp(p["swAmp"] * noise_slow(frames, sr, r6(p["swRate"]), skip))
    if p["g2"] > 0:
        broad = broad + p["g2"] * noise_bp(frames, sr, r6(p["fc2"]), r6(p["q2"]), skip)
    modes = np.zeros(frames)
    for k in range(1, N_MODES + 1):
        # struck-once metal mode: a decaying sine (phase wrapped before the sine
        # so the float32 dgenlisp port stays exact)
        ph = t * p[f"m{k}f"]
        modes += p[f"m{k}g"] * np.exp(p[f"m{k}d"] * t) * np.sin(2.0 * np.pi * (ph - np.floor(ph)))
    ramp = np.minimum(1.0, t / (p["atk"] * 1e-3))
    th = np.maximum(t - p["hold"] * 1e-3, 0.0)
    env_b = ramp * (np.exp(p["dTail"] * th) + p["aFast"] * np.exp(p["dFast"] * t))
    env_m = ramp * np.exp(p["dMode"] * th)
    x = p["aB"] * broad * env_b + modes * env_m
    if p["clickAmp"] > 0:
        x = x + p["clickAmp"] * np.exp(p["clickDecay"] * t) * noise_lp(frames, sr, r6(p["clickFc"]), skip)
    y = np.tanh(p["drive"] * x) / p["drive"] * p["outGain"]
    y = biquad(y, "hp", p["outHp"], 0.707, sr)
    return y.astype(np.float32)


class ModeTracks:
    """Heterodyned amplitude of each measured mode along the target: the
    diagnostic and loss term the spectral averages lack. 32 ms Hann window
    (main lobe narrower than the 62 Hz between the 1486/1548 modes), 5 ms hop,
    -60 dBFS floor."""
    def __init__(self, target, sr, freqs=MODE_SEEDS, win_ms=32.0, hop_ms=5.0, floor_db=-60.0):
        self.sr, self.freqs = sr, list(freqs)
        self.w = int(win_ms * 1e-3 * sr); self.hop = int(hop_ms * 1e-3 * sr)
        self.win = np.hanning(self.w) / (np.hanning(self.w).sum() / 2)
        self.floor = floor_db
        n = len(target); self.starts = list(range(0, n - self.w, self.hop))
        t = np.arange(n) / sr
        self.osc = [np.exp(-2j * np.pi * f * t) for f in self.freqs]
        self.ref = self.tracks(target)

    def tracks(self, x):
        out = np.empty((len(self.freqs), len(self.starts)))
        for i, o in enumerate(self.osc):
            z = x[:len(o)] * o
            for j, s0 in enumerate(self.starts):
                a = abs(np.sum(z[s0:s0 + self.w] * self.win))
                out[i, j] = max(20 * np.log10(a + 1e-12), self.floor)
        return out

    def loss(self, y):
        d = self.tracks(y) - self.ref
        m = self.ref > self.floor + 3
        return float(np.mean(np.abs(d[m]))) / 10.0 if m.any() else 0.0

    def table(self, y):
        d = self.tracks(y) - self.ref
        cols = [j for j in range(0, len(self.starts), max(1, len(self.starts) // 8))]
        lines = ["  mode Hz | " + " ".join(f"{self.starts[j] * 1000 // self.sr:4d}ms" for j in cols)]
        for i, f in enumerate(self.freqs):
            lines.append(f"  {f:7.0f} | " + " ".join(("  ·  " if self.ref[i, j] <= self.floor + 3 else f"{d[i, j]:+5.1f}") for j in cols))
        return "\n".join(lines)


HarmonicTracks = ModeTracks   # deficit_table.py hook


class SwishStats:
    """Envelope-modulation statistics of the wash, per band: the std of the
    detrended 1 ms dB envelope over 20-150 ms and the log energy of its
    modulation spectrum in 5-30 Hz and 30-150 Hz. Realisation-free, so it can
    see beating texture the spectral losses zero out."""
    BANDS = ((2000, 4000), (4000, 6000), (6000, 12000), (12000, 20000))
    def __init__(self, target, sr):
        self.sr = sr; self.n = len(target)
        self.f = np.fft.rfftfreq(self.n, 1.0 / sr)
        self.ref = self.stats(target)

    def stats(self, x):
        X = np.fft.rfft(x[:self.n]); out = []
        w = int(0.001 * self.sr); a, b = int(0.02 * self.sr), int(0.15 * self.sr)
        for lo, hi in self.BANDS:
            band = np.fft.irfft(X * ((self.f >= lo) & (self.f < hi)), self.n)
            e = np.sqrt(np.mean(band[a:b].reshape(-1, w) ** 2, axis=1)) if (b - a) % w == 0 else np.array([np.sqrt(np.mean(band[i:i + w] ** 2)) for i in range(a, b - w, w)])
            db = 20 * np.log10(e + 1e-9); tt = np.arange(len(db)); r = db - np.polyval(np.polyfit(tt, db, 1), tt)
            R = np.abs(np.fft.rfft(r * np.hanning(len(r)))) ** 2; fr = np.fft.rfftfreq(len(r), 1e-3)
            out += [np.log(r.std() + 1e-3), 0.5 * np.log(R[(fr >= 5) & (fr < 30)].sum() + 1e-6), 0.5 * np.log(R[(fr >= 30) & (fr < 150)].sum() + 1e-6)]
        return np.array(out)

    def loss(self, y):
        return float(np.mean(np.abs(self.stats(y) - self.ref)))

    def table(self, y):
        s, r = self.stats(y), self.ref
        return "\n".join(f"  {lo}-{hi}: std {np.exp(r[3*i]):.2f}/{np.exp(s[3*i]):.2f} dB  5-30Hz {r[3*i+1]:.2f}/{s[3*i+1]:.2f}  30-150Hz {r[3*i+2]:.2f}/{s[3*i+2]:.2f} (target/synth)" for i, (lo, hi) in enumerate(self.BANDS))


class BandBalanceStats:
    """Broad-band energy shares keep a dynamics fit from highpassing away body.

    The per-bin log floor in the main objective hides quiet but audible low-mid
    energy. These integrated powers are relative to the whole window, so global
    gain cannot buy a better spectral balance. Windowing avoids boundary leakage.
    """
    WINDOWS = ((0, 10), (10, 30), (30, 60), (60, 120), (120, 240))
    EDGES = (200, 400, 800, 1500, 3000, 6000, 10000, 16000, 22000)

    def __init__(self, target, sr):
        self.windows = []
        for a, b in self.WINDOWS:
            start, end = round(a * sr / 1000), round(b * sr / 1000)
            if end > len(target):
                continue
            freq = np.fft.rfftfreq(end - start, 1 / sr)
            masks = [(freq >= lo) & (freq < hi)
                     for lo, hi in zip(self.EDGES, self.EDGES[1:])]
            self.windows.append((slice(start, end), np.hanning(end - start), masks))
        self.ref = self.stats(target)

    def stats(self, x):
        result = []
        for window, taper, masks in self.windows:
            power = np.abs(np.fft.rfft(x[window] * taper)) ** 2
            total = max(float(power.sum()), 1e-30)
            result.append([np.log(max(float(power[m].sum()) / total, 1e-6))
                           for m in masks])
        return np.array(result)

    def loss(self, y):
        return float(np.mean(np.abs(self.stats(y) - self.ref)))

    def table(self, y):
        deficit = (self.ref - self.stats(y)) * 10 / np.log(10)
        return '\n'.join(f'  {a:3d}-{b:3d}ms: ' + ' '.join(f'{v:+5.1f}' for v in row)
                         for (a, b), row in zip(self.WINDOWS, deficit))


class DynamicsStats:
    """Local level and amplitude-distribution shape, independent of noise phase.

    Spectral pooling cannot distinguish a clipped noise attack from the source.
    Short-window fourth moments expose that flattening; RMS keeps removing the
    clipping from becoming a gain change. Aggregate neighbouring windows to avoid
    fitting individual random peaks. No waveform samples enter the parameter set.
    """
    WINDOWS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80),
               (80, 120), (120, 180), (180, 240))

    def __init__(self, target, sr):
        self.slices = [slice(round(a * sr / 1000), round(b * sr / 1000))
                       for a, b in self.WINDOWS if round(b * sr / 1000) <= len(target)]
        self.ref = self.stats(target)

    def stats(self, x):
        out = []
        for window in self.slices:
            y = x[window].astype(np.float64)
            power = max(float(np.mean(y * y)), 1e-12)
            kurtosis = float(np.mean(y ** 4)) / (power * power)
            out.append((0.5 * np.log(power), np.log(max(kurtosis, 1e-6))))
        return np.array(out)

    def loss(self, y):
        error = np.abs(self.stats(y) - self.ref)
        return float(np.mean(error[:, 0]) + 2 * np.mean(error[:, 1]))

    def table(self, y):
        s = self.stats(y)
        return '\n'.join(f'  {a:3d}-{b:3d}ms: RMS {r[0]*20/np.log(10):.1f}/{v[0]*20/np.log(10):.1f} dB, kurtosis {np.exp(r[1]):.2f}/{np.exp(v[1]):.2f} (target/synth)'
                         for (a, b), r, v in zip(self.WINDOWS, self.ref, s))


class Objective:
    def __init__(self, target, sr, highpass_hz, windows=compare.WINDOWS, pooled=False, bands=POOL_BANDS,
                 fine_weight=0.0, fine_windows=(4096, 8192), fine_band=(500.0, 15000.0), mode_weight=0.0, swish_weight=0.0, pooled_weight=1.0, dynamics_weight=0.0, balance_weight=0.0):
        self.sr, self.frames, self.hp, self.windows = sr, len(target), highpass_hz, windows
        self.mode_weight, self.swish_weight, self.pooled_weight = mode_weight, swish_weight, pooled_weight
        self.balance_weight = balance_weight
        self.balance = BandBalanceStats(target, sr) if balance_weight > 0 else None
        self.dynamics_weight = dynamics_weight
        self.dynamics = DynamicsStats(target, sr) if dynamics_weight > 0 else None
        self.swish = SwishStats(target, sr) if swish_weight > 0 else None
        self.modes = ModeTracks(target, sr) if mode_weight > 0 else None
        self.pooled = pooled
        # per-bin log magnitude at long windows: the only view that separates
        # metal modes 60 Hz apart (the pooled loss dilutes one 2% band to 1/256
        # of a window's error and happily trades a mode for broad fill)
        self.fine_weight, self.fine_windows = fine_weight, tuple(w for w in fine_windows if w <= len(target))
        self.fine_mask = {w: (lambda f: (f >= fine_band[0]) & (f <= fine_band[1]))(np.fft.rfftfreq(w, 1.0 / sr)) for w in self.fine_windows}
        self.target = compare.capture_highpass(target, sr, highpass_hz)
        self.pool = {}
        if pooled:
            for w in windows:
                # finer bands at the long windows so a narrow metal mode registers;
                # coarse bands at the short ones keep the Rayleigh variance pooled
                nb = int(min(bands, max(32, w // 16)))
                edges = np.geomspace(250.0, min(20000.0, sr / 2), nb + 1)   # the -47 dBFS rumble under 250 Hz is not the hat
                bands_w = nb
                f = np.fft.rfftfreq(w, 1.0 / sr)
                idx = np.searchsorted(edges, f, side="right") - 1
                m = np.zeros((len(f), bands_w))
                ok = (idx >= 0) & (idx < bands_w)
                m[np.arange(len(f))[ok], idx[ok]] = 1.0
                m = m[:, m.sum(axis=0) > 0]
                self.pool[w] = m / np.maximum(m.sum(axis=0, keepdims=True), 1)
        self.feat = {w: self.features(self.target, w) for w in windows}
        self.fine_feat = {w: self.fine_features(self.target, w) for w in self.fine_windows}
        self.evals = 0

    def fine_features(self, signal, w):
        hop = w // 8
        win = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, w)[::hop]
        scale = max(float(win.sum()) / 2.0, 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1)) / scale
        return np.log(mag[:, self.fine_mask[w]] + compare.LOG_EPSILON)

    def features(self, signal, w):
        hop = w // 4
        win = np.hanning(w).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, w)[::hop]
        scale = max(float(win.sum()) / 2.0, 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1)) / scale
        if self.pooled:
            power = (mag ** 2) @ self.pool[w]
            return 0.5 * np.log(power + compare.LOG_EPSILON ** 2)
        return np.log(mag + compare.LOG_EPSILON)

    def distance(self, signal):
        d = self.pooled_weight * sum(float(np.mean(np.abs(self.features(signal, w) - self.feat[w]))) for w in self.windows)
        if self.balance_weight > 0:
            d += self.balance_weight * self.balance.loss(signal)
        if self.dynamics_weight > 0:
            d += self.dynamics_weight * self.dynamics.loss(signal)
        if self.swish_weight > 0:
            d += self.swish_weight * self.swish.loss(signal)
        if self.fine_weight > 0:
            d += self.fine_weight * sum(float(np.mean(np.abs(self.fine_features(signal, w) - self.fine_feat[w]))) for w in self.fine_windows)
        if self.mode_weight > 0:
            d += self.mode_weight * self.modes.loss(signal)
        return d

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


def clamp(p):
    out = {}
    for name, (lo, hi, _) in BOUNDS.items():
        v = p.get(name, MEASURED[name])
        out[name] = min(max(v, min(lo, hi)), max(lo, hi))
    return out


def main():
    global NOISE_SKIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Assets/909-open-hat.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--final-passes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", help="params json to refine from")
    ap.add_argument("--only", help="comma list: refine only these scalars from --start, rest frozen")
    ap.add_argument("--highpass-hz", type=float, default=compare.DEFAULT_HIGHPASS_HZ)
    ap.add_argument("--bands", type=int, default=POOL_BANDS)
    ap.add_argument("--fine-weight", type=float, default=0.0, help="weight of the per-bin long-window term (diluted by noise bins; kept for experiments)")
    ap.add_argument("--mode-weight", type=float, default=1.0, help="weight of the heterodyned mode-track term (mean |dB| / 10)")
    ap.add_argument("--swish-weight", type=float, default=0.0, help="weight of the envelope-modulation statistics term")
    ap.add_argument("--pooled-weight", type=float, default=1.0)
    ap.add_argument("--balance-weight", type=float, default=0.0,
                    help="broad-band energy shares: protect body-to-hiss balance")
    ap.add_argument("--dynamics-weight", type=float, default=0.0,
                    help="local RMS and kurtosis: protect attack dynamics from saturation")
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
    obj = Objective(target, sr, args.highpass_hz, compare.WINDOWS if args.gate_windows else TRAIN_WINDOWS, pooled=not args.gate_windows, bands=args.bands, fine_weight=args.fine_weight, mode_weight=args.mode_weight, swish_weight=args.swish_weight, pooled_weight=args.pooled_weight, dynamics_weight=args.dynamics_weight, balance_weight=args.balance_weight)
    gate = Objective(target, sr, args.highpass_hz, compare.WINDOWS)
    base = midpoints()
    baseline = obj.evaluate(base)
    gate_baseline = gate.evaluate(base)
    reference.write_wav(os.path.join(args.out, "initial.wav"), render(base, frames, sr), sr)
    log(f"frames={frames} sr={sr} baseline(midpoints)={baseline:.6f} gate baseline={gate_baseline:.6f}")

    rng = np.random.default_rng(args.seed)
    order = ORDER
    if args.only:
        assert args.start, "--only needs --start"
        order = [n for n in args.only.split(",") if n]
        with open(args.start) as f:
            s = clamp(json.load(f).get("params", json.load(open(args.start))))
        starts = [s]
        log(f"refining only {order}")
    else:
        cands = [("measured", clamp(MEASURED)), ("midpoint", dict(base))]
        if args.start:
            with open(args.start) as f:
                s = json.load(f)
            cands.append(("start", clamp(s.get("params", s))))
        for i in range(args.restarts):
            p = {}
            for name, (lo, hi, _) in BOUNDS.items():
                zl, zh = sorted((transformed(name, lo), transformed(name, hi)))
                p[name] = natural(name, float(rng.uniform(zl + 0.05 * (zh - zl), zh - 0.05 * (zh - zl))))
            for k in range(1, N_MODES + 1):
                p[f"m{k}f"] = MEASURED[f"m{k}f"]     # random draws keep the measured modes
            cands.append((f"random{i}", p))
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
    # metric noise floor: the same patch with a different noise realisation
    NOISE_SKIP = 4099
    gate_floor = gate.evaluate(p)
    NOISE_SKIP = 0
    pinned = [n for n in ORDER if min(abs(transformed(n, p[n]) - transformed(n, b)) for b in BOUNDS[n][:2] if (b != 0 or BOUNDS[n][2] == "linear")) < 1e-6]
    report = {"params": p, "learned": d, "baseline": baseline, "improvement": 1 - d / baseline,
              "gate": {"learned": gate_learned, "baseline": gate_baseline, "improvement": 1 - gate_learned / gate_baseline,
                       "floor": gate_floor, "excess": gate_learned - gate_floor},
              "sampleRate": sr, "frames": frames, "pinned": pinned, "target": args.target}
    with open(os.path.join(args.out, "recovered_params.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"RESULT train learned={d:.6f} baseline={baseline:.6f} improvement={1 - d / baseline:.2%} pinned={pinned}")
    log(f"GATE   learned={gate_learned:.6f} baseline={gate_baseline:.6f} floor={gate_floor:.6f} excess={gate_learned - gate_floor:.4f}")
    log(json.dumps({k: round(v, 4) for k, v in p.items()}))
    if obj.dynamics:
        log(obj.dynamics.table(y))
    if obj.balance:
        log('Band-share deficits dB (positive = missing): ' + str(obj.balance.EDGES))
        log(obj.balance.table(y))


if __name__ == "__main__":
    main()
