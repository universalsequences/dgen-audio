#!/usr/bin/env python3
"""Phase 1 measurements for Assets/monologue-bass.wav (NEW_TARGET_PLAYBOOK.md).

Prints a measurement report and writes a JSON summary next to --out.
Measurements: housekeeping/provenance, effective length, pitch contour,
amplitude envelope (sustain-plateau detection), harmonic tracks over time,
spectral-centroid track (filter EG evidence), attack, noise floor, beating.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load(path: Path):
    rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    return rate, audio


def db(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def housekeeping(rate, audio, report):
    stereo = audio.ndim == 2
    mono = audio.mean(axis=1) if stereo else audio
    report["sample_rate"] = rate
    report["frames"] = int(len(mono))
    report["duration_s"] = len(mono) / rate
    report["stereo"] = bool(stereo)
    if stereo:
        l, r = audio[:, 0], audio[:, 1]
        report["stereo_max_abs_delta"] = float(np.max(np.abs(l - r)))
        report["stereo_correlation"] = float(
            np.corrcoef(l, r)[0, 1]) if l.std() > 0 else 1.0
    report["peak_dbfs"] = float(db(np.max(np.abs(audio))))
    report["clipped_samples"] = int(np.sum(np.abs(audio) >= 0.999))
    report["dc_offset"] = float(mono.mean())
    # Sub-20 Hz and >20.5 kHz band energy (provenance: 320 kbps mp3 shelves ~20.5 kHz)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / rate)
    total = np.sum(spec**2)
    for name, lo, hi in [("sub20", 0, 20), ("sub30", 0, 30),
                         ("band_20500_up", 20500, rate / 2),
                         ("band_18k_20k", 18000, 20000)]:
        m = (freqs >= lo) & (freqs < hi)
        report[f"energy_frac_{name}"] = float(np.sum(spec[m]**2) / total)
    return mono


def envelope(rate, mono, report):
    # Window must span at least one fundamental period (35 Hz -> 28.6 ms),
    # otherwise RMS oscillates within the period.
    win = int(0.030 * rate)
    hop = win // 2
    n = (len(mono) - win) // hop
    t = np.array([(i * hop + win / 2) / rate for i in range(n)])
    rms = np.array([np.sqrt(np.mean(mono[i * hop:i * hop + win]**2)) for i in range(n)])
    peak = rms.max()
    report["envelope_t"] = t.tolist()
    report["envelope_db"] = db(rms / peak).tolist()
    # effective length: last time envelope is above -60 dB rel peak
    above = np.where(rms / peak > 1e-3)[0]
    report["effective_length_s"] = float(t[above[-1]]) if len(above) else 0.0
    # attack time: first crossing of -3 dB rel peak
    up = np.where(rms / peak > 0.707)[0]
    report["attack_to_minus3db_s"] = float(t[up[0]]) if len(up) else None
    report["peak_time_s"] = float(t[np.argmax(rms)])
    return t, rms


def pitch_contour(rate, mono, report):
    """Harmonic-score (HPS-style) pitch track — robust at very low f0
    where short-window autocorrelation fails."""
    win = int(0.100 * rate)
    hop = int(0.010 * rate)
    nfft = 1 << 17
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    cand = np.where((freqs >= 25) & (freqs <= 300))[0]
    times, f0s = [], []
    env_gate = np.max(np.abs(mono)) * 0.02
    hann = np.hanning(win)
    for start in range(0, len(mono) - win, hop):
        seg = mono[start:start + win]
        if np.max(np.abs(seg)) < env_gate:
            continue
        mag = np.abs(np.fft.rfft(seg * hann, nfft))
        score = np.zeros(len(cand))
        for h in range(1, 10):
            score += mag[np.minimum(cand * h, len(mag) - 1)] / h
        times.append((start + win / 2) / rate)
        f0s.append(float(freqs[cand[np.argmax(score)]]))
    times, f0s = np.array(times), np.array(f0s)
    report["pitch_t"] = times.tolist()
    report["pitch_hz"] = f0s.tolist()
    if len(f0s):
        report["pitch_median_hz"] = float(np.median(f0s))
        report["pitch_std_hz"] = float(np.std(f0s))
        report["pitch_first_hz"] = float(f0s[0])
        report["pitch_last_hz"] = float(f0s[-1])
    return times, f0s


def harmonics(rate, mono, f0, report):
    """Per-harmonic dB tracks (H1..H16) via 46 ms Hann DFT projections."""
    win = int(0.046 * rate)
    hop = int(0.010 * rate)
    hann = np.hanning(win)
    n_h = 16
    times, tracks = [], []
    tt = np.arange(win) / rate
    env_gate = np.max(np.abs(mono)) * 0.01
    for start in range(0, len(mono) - win, hop):
        seg = mono[start:start + win]
        if np.max(np.abs(seg)) < env_gate:
            continue
        w = seg * hann
        row = []
        for h in range(1, n_h + 1):
            ph = 2 * np.pi * f0 * h * tt
            amp = 2.0 * np.abs(np.dot(w, np.exp(-1j * ph))) / np.sum(hann)
            row.append(amp)
        times.append((start + win / 2) / rate)
        tracks.append(row)
    times = np.array(times)
    tracks = np.array(tracks)  # [frame, harmonic]
    report["harm_t"] = times.tolist()
    report["harm_db"] = db(tracks).tolist()
    # early vs late relative-to-H1 profile
    def profile(mask):
        m = tracks[mask].mean(axis=0)
        return (db(m) - db(m[0])).tolist()
    early = (times >= 0.02) & (times <= 0.10)
    late = (times >= 0.35) & (times <= 0.60)
    if early.any():
        report["harm_rel_db_early_20_100ms"] = profile(early)
    if late.any():
        report["harm_rel_db_late_350_600ms"] = profile(late)
    # odd/even balance (pulse vs saw evidence), late window
    if late.any():
        m = tracks[late].mean(axis=0)
        odd = np.sum(m[0::2] ** 2)   # H1,H3,..
        even = np.sum(m[1::2] ** 2)  # H2,H4,..
        report["late_even_to_odd_db"] = float(10 * np.log10(even / odd))
    return times, tracks


def beating(times, tracks, report):
    """Non-monotonic per-harmonic wiggles = two-oscillator beating.

    For each harmonic, measure the depth of the largest dip-and-recovery in
    its dB track before 0.45 s (pre-release). A closing filter is monotonic;
    a null that recovers by >3 dB is a beat null at n*detuneHz.
    """
    out = {}
    pre = times < 0.45
    for h in range(16):
        tr = db(tracks[pre, h])
        # largest recovery after a local minimum
        best = 0.0
        null_t = None
        for i in range(1, len(tr) - 1):
            drop = tr[:i].max() - tr[i]
            recover = tr[i:].max() - tr[i]
            if min(drop, recover) > best:
                best = min(drop, recover)
                null_t = float(times[pre][i])
        if best > 3.0:
            out[f"H{h+1}"] = {"depth_db": round(float(best), 1), "null_t_s": null_t}
    report["beat_nulls"] = out


def centroid_track(rate, mono, report):
    """Spectral centroid over time — the filter-sweep fingerprint."""
    win, hop = 2048, 512
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / rate)
    times, cents = [], []
    env_gate = np.max(np.abs(mono)) * 0.01
    for start in range(0, len(mono) - win, hop):
        seg = mono[start:start + win]
        if np.max(np.abs(seg)) < env_gate:
            continue
        mag = np.abs(np.fft.rfft(seg * hann))
        times.append((start + win / 2) / rate)
        cents.append(float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)))
    report["centroid_t"] = times
    report["centroid_hz"] = cents
    return np.array(times), np.array(cents)


def noise_floor(rate, mono, report, eff_len):
    tail_start = int(min(eff_len + 0.05, len(mono) / rate - 0.05) * rate)
    tail = mono[tail_start:]
    if len(tail) > 64:
        report["tail_rms_dbfs"] = float(db(np.sqrt(np.mean(tail**2))))
    # broadband noise during sustain: energy between harmonics 2-6 kHz
    report["quantization_floor_dbfs"] = float(db(1.0 / 32768))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rate, audio = load(args.wav)
    report = {"file": str(args.wav)}
    mono = housekeeping(rate, audio, report)
    envelope(rate, mono, report)
    pitch_contour(rate, mono, report)
    f0 = report.get("pitch_median_hz", 110.0)
    h_times, h_tracks = harmonics(rate, mono, f0, report)
    beating(h_times, h_tracks, report)
    centroid_track(rate, mono, report)
    noise_floor(rate, mono, report, report["effective_length_s"])

    # Print scalar summary (skip long arrays)
    for k, v in report.items():
        if isinstance(v, list):
            continue
        print(f"{k:32s} {v}")
    for key in ["harm_rel_db_early_20_100ms", "harm_rel_db_late_350_600ms"]:
        if key in report:
            vals = " ".join(f"{x:6.1f}" for x in report[key])
            print(f"{key}: {vals}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
