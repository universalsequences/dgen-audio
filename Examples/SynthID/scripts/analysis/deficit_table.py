#!/usr/bin/env python3
"""What the synth still lacks: target minus learned, per time window and band.

    python3 Examples/SynthID/scripts/analysis/deficit_table.py output/<run> [--fit-module scripts/fit_<sound>.py]

Prints the target's band levels (dBFS) and the signed deficit (target − learned)
per time window, the first 60 samples of both (a fitted click can become an
impulse the sample never had), the >4 kHz RMS envelopes (recording texture the
gate cannot see), and — with --fit-module — the synth-minus-target harmonic
table from that module's HarmonicTracks. Read this after every round; the gate
number alone hid a missing harmonic ladder and a missing hiss on the Virus B kick.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import compare  # noqa: E402

EDGES = [(30, 100), (100, 200), (200, 400), (400, 800), (800, 1500), (1500, 3000), (3000, 6000), (6000, 12000), (12000, 24000)]
WINS = [(0, 3), (3, 8), (8, 15), (15, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 450)]


def bands(s, sr, a, b):
    seg = s[int(a * sr / 1000):int(b * sr / 1000)]
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w)) ** 2 / (w.sum() / 2) ** 2
    F = np.fft.rfftfreq(len(seg), 1 / sr)
    return [10 * np.log10(S[(F >= lo) & (F < hi)].sum() + 1e-14) for lo, hi in EDGES if lo < sr / 2]


def hp_env(s, sr, cut=4000.0, step=50):
    X = np.fft.rfft(s)
    F = np.fft.rfftfreq(len(s), 1 / sr)
    h = np.fft.irfft(X * (F > cut), len(s))
    return " ".join(f"{a}:{20 * np.log10(np.sqrt(np.mean(h[int(a * sr / 1000):int((a + step) * sr / 1000)] ** 2)) + 1e-9):.0f}"
                    for a in range(0, int(len(s) / sr * 1000) - step, step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fit-module")
    ap.add_argument("--learned", default="learned.wav")
    args = ap.parse_args()
    x, sr = compare.read_wav(os.path.join(args.run, "target.wav"))
    y, sr2 = compare.read_wav(os.path.join(args.run, args.learned))
    assert sr == sr2, "sample-rate mismatch"
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    edges = [e for e in EDGES if e[0] < sr / 2]
    print("target dBFS | deficit (target − learned) dB;  bands: " + " ".join(f"{lo}-{hi}" for lo, hi in edges))
    for a, b in WINS:
        if b * sr / 1000 > n:
            break
        T, L = bands(x, sr, a, b), bands(y, sr, a, b)
        print(f"{a:3d}-{b:3d}ms T: " + " ".join(f"{v:6.0f}" for v in T) + "  | D: " + " ".join(f"{t - l:+5.1f}" for t, l in zip(T, L)))
    print("(positive = synth is short there; a uniformly short high band from 30 ms on is recording hiss, fit it alone)")
    print("\nonset x1000, target : " + str(np.round(x[:60] * 1000).astype(int).tolist()))
    print("onset x1000, learned: " + str(np.round(y[:60] * 1000).astype(int).tolist()))
    print(f"first sample > 0.01: target {int(np.argmax(np.abs(x) > 0.01))}, learned {int(np.argmax(np.abs(y) > 0.01))}")
    print("\n>4 kHz RMS dBFS per 50 ms, target : " + hp_env(x, sr))
    print(">4 kHz RMS dBFS per 50 ms, learned: " + hp_env(y, sr))
    if args.fit_module:
        spec = importlib.util.spec_from_file_location("fit_voice", args.fit_module)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.fit_module)))
        spec.loader.exec_module(mod)
        if hasattr(mod, "HarmonicTracks"):
            tr = mod.HarmonicTracks(x, sr)
            print("\nharmonic tracks, synth minus target (dB):\n" + tr.table(y))
            print(f"harmonic-track loss {tr.loss(y):.4f}")


if __name__ == "__main__":
    main()
