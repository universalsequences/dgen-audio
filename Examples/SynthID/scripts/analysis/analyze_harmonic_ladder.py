#!/usr/bin/env python3
"""Pitched-drum measurement: what the harmonics of a swept hit actually are.

    python3 Examples/SynthID/scripts/analysis/analyze_harmonic_ladder.py Assets/<file>.wav

Answers, before any DSP is written (lessons from the Virus B kick, 2026-09-03):
  1. f0(t) from positive zero crossings, and how well one- vs two-exponential
     pitch models fit it (rms log error) — one exponential missed that sweep
     by 30-40 % for 40 ms and the MR-STFT gate never noticed.
  2. Harmonic ladder: each harmonic's level relative to H1 per time window,
     heterodyned along the measured phase with a TWO-PERIOD Hann window. A
     fixed short window leaks between neighbours once f0 is low and fakes a
     flat "pulse" spectrum; a long FFT shows the fundamental's Hann sidelobes
     as fake ridges. Both fooled a previous round.
  3. Whether Hk-H1 decays with H1 (oscillator content -> additive bank) or with
     H1's power (static waveshaper) — the slope column.
  4. Residual after resynthesising harmonics 1..20: a band table of target /
     harmonics / residual. Residual 20+ dB down below 1 kHz = the sound IS
     harmonics; anything left is transient or recording texture.
  5. Recording texture: >4 kHz RMS per 25 ms and 1/3-octave shape per window.
     This sits under the gate's -60 dB floor, so the gate cannot fit it; it
     must be measured here and fitted alone (fit_virus_kick.py --only).
  6. The first 60 samples: a fitted "click" can turn into an impulse the
     sample never had.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import compare  # noqa: E402


def zero_crossing_f0(x, sr, lo=25.0, hi=2500.0):
    n = len(x)
    X = np.fft.rfft(x)
    F = np.fft.rfftfreq(n, 1.0 / sr)
    xl = np.fft.irfft(X * ((F > lo) & (F < hi)), n)
    zc = np.where((xl[:-1] <= 0) & (xl[1:] > 0))[0]
    zc = zc + (-xl[zc]) / (xl[zc + 1] - xl[zc])
    per = np.diff(zc) / sr
    tc = (zc[:-1] + zc[1:]) / 2 / sr
    f = 1.0 / per
    ok = (f > 30) & (f < 3000)
    t = np.arange(n) / sr
    return tc[ok], f[ok], np.interp(t, tc[ok], f[ok])


def fit_pitch_models(tc, f, t_max=0.25, seed=0):
    m = (tc > 0.0005) & (tc < t_max)
    t, y = tc[m], f[m]

    def two(p, t):
        fe, a1, r1, a2, r2 = p
        return fe + a1 * np.exp(r1 * t) + a2 * np.exp(r2 * t)

    def loss(p):
        v = two(p, t)
        return np.inf if np.any(v <= 0) else float(np.mean((np.log(v) - np.log(y)) ** 2))

    rng = np.random.default_rng(seed)
    fe0 = float(np.median(y[t > 0.6 * t_max])) if np.any(t > 0.6 * t_max) else float(y[-1])
    best = None
    for _ in range(4000):
        p = [fe0, 10 ** rng.uniform(2.0, 3.6), -10 ** rng.uniform(1.5, 2.8), 10 ** rng.uniform(1.0, 3.2), -10 ** rng.uniform(0.5, 2.2)]
        l = loss(p)
        if best is None or l < best[0]:
            best = (l, p)
    l, p = best
    for _ in range(4000):
        q = [v * np.exp(rng.normal(0, 0.03)) if i else v + rng.normal(0, 0.2) for i, v in enumerate(p)]
        lq = loss(q)
        if lq < l:
            l, p = lq, q
    # one-exponential: a2 = 0
    best1 = None
    for _ in range(4000):
        q = [fe0, 10 ** rng.uniform(2.0, 3.6), -10 ** rng.uniform(1.5, 2.8), 0.0, -10.0]
        lq = loss(q)
        if best1 is None or lq < best1[0]:
            best1 = (lq, q)
    l1, p1 = best1
    for _ in range(3000):
        q = [p1[0] + rng.normal(0, 0.2), p1[1] * np.exp(rng.normal(0, 0.03)), p1[2] * np.exp(rng.normal(0, 0.03)), 0.0, -10.0]
        lq = loss(q)
        if lq < l1:
            l1, p1 = lq, q
    return (p, np.sqrt(l)), (p1, np.sqrt(l1)), two


def heterodyne_tracks(x, sr, f0, phi, K=14, periods=2.0, step_s=0.002, t_max=0.30):
    n = len(x)
    times = np.arange(0.003, min(t_max, n / sr - 0.01), step_s)
    A = np.zeros((K, len(times)))
    for j, ti in enumerate(times):
        i = int(ti * sr)
        w = int(np.clip(periods / f0[i], 0.0015, 0.045) * sr)
        a, b = max(0, i - w // 2), min(n, i + w // 2)
        ker = np.hanning(b - a)
        ker /= ker.sum()
        seg = x[a:b] * ker
        for k in range(1, K + 1):
            A[k - 1, j] = 2 * abs(np.sum(seg * np.exp(-2j * np.pi * k * phi[a:b])))
    return times, 20 * np.log10(A + 1e-7)


def resynthesis_residual(x, sr, f0, phi, K=20):
    n = len(x)
    t = np.arange(n) / sr
    resyn = np.zeros(n)
    idx = np.arange(0, n, 8)
    for k in range(1, K + 1):
        z = x * np.exp(-2j * np.pi * k * phi)
        out = np.zeros(len(idx), complex)
        for m, i in enumerate(idx):
            w = int(np.clip(2.0 / f0[i], 0.0015, 0.045) * sr)
            a, b = max(0, i - w // 2), min(n, i + w // 2)
            ker = np.hanning(b - a)
            out[m] = np.sum(z[a:b] * ker) / ker.sum()
        c = np.interp(t, idx / sr, out.real) + 1j * np.interp(t, idx / sr, out.imag)
        resyn += 2 * np.abs(c) * np.cos(2 * np.pi * k * phi + np.angle(c))
    return resyn, x - resyn


def band_table(sig, sr, label, edges, wins):
    print(f"\n{label}: band power dBFS   cols=" + " ".join(f"{a}-{b}" for a, b in edges))
    for a, b in wins:
        seg = sig[int(a * sr / 1000):int(b * sr / 1000)]
        w = np.hanning(len(seg))
        S = np.abs(np.fft.rfft(seg * w)) ** 2 / (w.sum() / 2) ** 2
        F = np.fft.rfftfreq(len(seg), 1 / sr)
        print(f"{a:3d}-{b:3d}ms " + " ".join(f"{10 * np.log10(S[(F >= lo) & (F < hi)].sum() + 1e-14):7.1f}" for lo, hi in edges))


def main():
    path = sys.argv[1]
    x, sr = compare.read_wav(path)
    n = len(x)
    print(f"{path}: sr {sr}, {n} frames, {n / sr * 1000:.1f} ms, peak {20 * np.log10(np.abs(x).max()):.2f} dBFS")

    tc, f, f0 = zero_crossing_f0(x, sr)
    phi = np.cumsum(f0) / sr
    print("\nzero-crossing f0 (ms:Hz): " + " ".join(f"{a * 1000:.1f}:{b:.0f}" for a, b in list(zip(tc, f))[:16]) + f" ... tail {np.median(f[tc > 0.6 * tc[-1]]):.1f}")
    (p2, e2), (p1, e1), model = fit_pitch_models(tc, f)
    print(f"pitch: two-exp fEnd={p2[0]:.2f} a1={p2[1]:.1f} r1={p2[2]:.1f} a2={p2[3]:.1f} r2={p2[4]:.1f}  rms log err {e2:.4f}")
    print(f"       one-exp fEnd={p1[0]:.2f} a1={p1[1]:.1f} r1={p1[2]:.1f}                       rms log err {e1:.4f}")
    for ms in (3, 8, 15, 25, 40, 60, 90):
        i = np.argmin(abs(tc - ms / 1000))
        print(f"   {ms:3d} ms measured {f[i]:7.1f}  two {model(p2, tc[i]):7.1f}  one {model(p1, tc[i]):7.1f}")

    K = 12
    times, A = heterodyne_tracks(x, sr, f0, phi, K=K)
    rel = A - A[:1]
    wins = [(0.004, 0.015), (0.015, 0.04), (0.04, 0.08), (0.08, 0.15), (0.15, 0.25)]
    print("\nharmonic ladder Hk-H1 (dB, two-period heterodyne), and dB/s slope of Hk-H1 over 70-250 ms")
    print("k    " + "  ".join(f"{a * 1000:.0f}-{b * 1000:.0f}ms" for a, b in wins) + "   slope")
    m = (times >= 0.07) & (times < 0.25)
    for k in range(2, K + 1):
        row = [rel[k - 1, (times >= a) & (times < b)].mean() for a, b in wins]
        sl = np.polyfit(times[m], rel[k - 1, m], 1)[0] if m.sum() > 3 else float("nan")
        print(f"H{k:<2}  " + "  ".join(f"{v:8.1f}" for v in row) + f"   {sl:8.1f}")
    print("H1 dBFS: " + " ".join(f"{ti * 1000:.0f}:{A[0, j]:.1f}" for j, ti in enumerate(times) if j % 5 == 0))
    print("(slope ~ 0 or ~ H1's own slope -> oscillator content, additive bank; slope ~ (k-1) x H1 slope -> static waveshaper)")

    resyn, resid = resynthesis_residual(x, sr, f0, phi)
    edges = [(30, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
    bwins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 300), (300, min(450, int(n / sr * 1000) - 1))]
    band_table(x, sr, "TARGET", edges, bwins)
    band_table(resyn, sr, "HARMONICS 1-20", edges, bwins)
    band_table(resid, sr, "RESIDUAL (not harmonics of f0)", edges, bwins)

    X = np.fft.rfft(x)
    F = np.fft.rfftfreq(n, 1 / sr)
    hp = np.fft.irfft(X * (F > 4000), n)
    print("\nrecording texture: >4 kHz RMS dBFS per 25 ms: " + " ".join(
        f"{a}:{20 * np.log10(np.sqrt(np.mean(hp[int(a * sr / 1000):int((a + 25) * sr / 1000)] ** 2)) + 1e-9):.0f}"
        for a in range(0, int(n / sr * 1000) - 25, 25)))
    oct_edges = [1500, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
    for a, b in [(20, 60), (60, 120), (120, 200), (200, 300)]:
        seg = x[int(a * sr / 1000):int(b * sr / 1000)]
        w = np.hanning(len(seg))
        S = np.abs(np.fft.rfft(seg * w)) ** 2 / (w.sum() / 2) ** 2
        Fs = np.fft.rfftfreq(len(seg), 1 / sr)
        print(f"  {a}-{b}ms 1/3-oct dB per 100 Hz: " + " ".join(
            f"{lo}:{10 * np.log10(S[(Fs >= lo) & (Fs < hi)].sum() / (hi - lo) * 100 + 1e-15):.0f}" for lo, hi in zip(oct_edges[:-1], oct_edges[1:]) if hi <= sr / 2))
    print("(texture that decays with the body but is not harmonic = machine/recording hiss: zero-default HP-noise block, fitted ALONE)")

    print("\nonset, first 60 samples x1000: " + str(np.round(x[:60] * 1000).astype(int).tolist()))
    print("first sample above 0.01:", int(np.argmax(np.abs(x) > 0.01)))


if __name__ == "__main__":
    main()
