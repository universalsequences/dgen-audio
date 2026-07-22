#!/usr/bin/env python3
"""CPU pitch fit + deterministic initial params for a subtractive-bass target.

Produces the playbook's deterministic baseline: spec-table midpoints (exact
transformed-space midpoints, matching Params.swift) + the CPU pitch fit as
frozen subF0, + measured note-off as frozen subNoteOff.

Usage: prepare_subtractive_initial.py <prepared_target.wav> --out-dir <dir>
Writes pitch_fit.json and initial_params.json into --out-dir.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare  # noqa: E402


def precise_f0(mono, rate):
    """Weighted least-squares fit of f0 to harmonic peak positions.

    Long-FFT peak positions of H1..H8 are refined with parabolic
    interpolation; f0 = argmin sum_n w_n (f_n - n f0)^2 with magnitude
    weights, i.e. f0 = sum(w n f_n) / sum(w n^2).
    """
    n = 1 << 18
    seg = mono[: min(len(mono), int(0.55 * rate))]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    # coarse f0 via harmonic score
    cand = np.where((freqs >= 25) & (freqs <= 300))[0]
    score = np.zeros(len(cand))
    for h in range(1, 9):
        score += mag[np.minimum(cand * h, len(mag) - 1)] / h
    f0c = freqs[cand[np.argmax(score)]]
    num = den = 0.0
    peaks = []
    for h in range(1, 9):
        lo = int((h * f0c * 0.96) * n / rate)
        hi = int((h * f0c * 1.04) * n / rate)
        k = lo + int(np.argmax(mag[lo:hi]))
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + c
        dk = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        fh = (k + dk) * rate / n
        w = float(b)
        peaks.append({"harmonic": h, "hz": fh, "mag": w})
        num += w * h * fh
        den += w * h * h
    return num / den, f0c, peaks


def note_off(mono, rate):
    """Knee of the release: time of maximum steepening of the log envelope."""
    win = int(0.030 * rate)
    hop = win // 4
    count = (len(mono) - win) // hop
    t = np.array([(i * hop + win / 2) / rate for i in range(count)])
    rms = np.array(
        [np.sqrt(np.mean(mono[i * hop:i * hop + win] ** 2)) for i in range(count)])
    logdb = 20 * np.log10(np.maximum(rms / rms.max(), 1e-6))
    # slope over +/-2 hops; knee = where slope first drops below 2x the
    # median pre-knee decay slope, searched after the envelope peak
    slope = np.gradient(logdb, t)
    peak_i = int(np.argmax(logdb))
    body = slope[peak_i + 4:]
    tb = t[peak_i + 4:]
    med = np.median(body[: max(4, len(body) // 3)])
    for i, s in enumerate(body):
        if s < 3.0 * med and logdb[peak_i + 4 + i] < -20:
            return float(tb[i]), float(med)
    return float(t[-1]), float(med)


MIDPOINTS = {
    # (min, max, mode) mirrored from Params.swift KickParamSpecs.subtractiveBass
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
}


def midpoint(lo, hi, mode):
    if mode == "log":
        return math.sqrt(lo * hi)
    if mode == "log1p":
        return math.exp((math.log1p(lo) + math.log1p(hi)) / 2.0) - 1.0
    return (lo + hi) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--note-off", type=float, default=None,
                    help="override measured note-off (seconds)")
    args = ap.parse_args()

    mono, rate = compare.read_wav(str(args.target))
    f0, coarse, peaks = precise_f0(mono, rate)
    measured_off, body_slope = note_off(mono, rate)
    off = args.note_off if args.note_off is not None else measured_off

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pitch = {
        "f0": f0, "coarse_f0": coarse, "harmonic_peaks": peaks,
        "noteOff_measured": measured_off, "noteOff_used": off,
        "body_decay_db_per_s": body_slope,
    }
    (args.out_dir / "pitch_fit.json").write_text(json.dumps(pitch, indent=1))

    params = {name: midpoint(*spec) for name, spec in MIDPOINTS.items()}
    params["subF0"] = round(f0, 4)
    params["subNoteOff"] = round(off, 4)
    (args.out_dir / "initial_params.json").write_text(
        json.dumps(params, indent=1, sort_keys=True))
    print(f"f0={f0:.4f} Hz (coarse {coarse:.2f})  noteOff={off:.4f}s "
          f"(measured {measured_off:.4f}, body slope {body_slope:.1f} dB/s)")
    print(f"wrote {args.out_dir}/pitch_fit.json, initial_params.json")


if __name__ == "__main__":
    main()
