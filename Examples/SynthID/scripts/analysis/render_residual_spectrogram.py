#!/usr/bin/env python3
"""Target / learned / residual spectrogram figure for the monologue paper.

Renders a 3-panel figure: target and learned log-frequency STFT magnitude
spectrograms plus their log-magnitude DIFFERENCE (target dB - learned dB).
The fit is phase-blind, so a time-domain subtraction does not cancel and
is NOT a meaningful residual; the magnitude difference is the quantity the
MR-STFT metric actually scores. Optionally also writes the (phase-sensitive,
diagnostic-only) time-domain difference as a wav.

Usage:
  render_residual_spectrogram.py --target t.wav --learned l.wav \
      --out-png fig.png [--out-wav residual.wav] [--title "v2"]
"""

import argparse
import sys
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import compare  # noqa: E402


def stft_db(x, rate, w=1024):
    hop = w // 4
    hann = np.hanning(w).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, w)[::hop]
    mag = np.abs(np.fft.rfft(frames * hann, axis=1)) / max(hann.sum() / 2, 1e-12)
    t = np.arange(len(frames)) * hop / rate
    f = np.fft.rfftfreq(w, 1.0 / rate)
    return t, f, 20 * np.log10(mag.T + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-wav")
    ap.add_argument("--title", default="")
    ap.add_argument("--fmax", type=float, default=8000.0)
    args = ap.parse_args()

    target, rate = compare.read_wav(args.target)
    learned, lrate = compare.read_wav(args.learned)
    assert rate == lrate, f"rate mismatch {rate} vs {lrate}"
    n = min(len(target), len(learned))
    target, learned = target[:n], learned[:n]

    for name, x in [("target", target), ("learned", learned)]:
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        print(f"{name}: rms {20*np.log10(rms):+.1f} dBFS  "
              f"peak {20*np.log10(np.max(np.abs(x)) + 1e-12):+.1f} dBFS")

    if args.out_wav:
        diff = target - learned
        print(f"time-domain diff rms {20*np.log10(float(np.sqrt(np.mean(diff**2)) + 1e-12)):+.1f} dBFS "
              "(phase-sensitive; diagnostic only)")
        with wave.open(args.out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(
                (np.clip(diff, -1, 1) * 32767).astype(np.int16).tobytes())
        print(f"time-domain diff wav -> {args.out_wav}")

    tt, ff, target_db = stft_db(target, rate)
    _, _, learned_db = stft_db(learned, rate)
    diff_db = np.clip(target_db, -90, 0) - np.clip(learned_db, -90, 0)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for ax, label, db in [(axes[0], "Target", target_db),
                          (axes[1], "Learned (v2 circuit)", learned_db)]:
        im_mag = ax.pcolormesh(tt, ff, db, vmin=-90, vmax=0, cmap="magma",
                               shading="auto", rasterized=True)
        ax.set_title(label, fontsize=10)
    im_diff = axes[2].pcolormesh(tt, ff, diff_db, vmin=-30, vmax=30,
                                 cmap="RdBu_r", shading="auto", rasterized=True)
    axes[2].set_title("Residual: target dB $-$ learned dB", fontsize=10)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(30, args.fmax)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("frequency (Hz)", fontsize=9)
    cb1 = fig.colorbar(im_mag, ax=axes[:2], pad=0.012, aspect=28)
    cb1.set_label("dBFS", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im_diff, ax=axes[2:], pad=0.025, aspect=28)
    cb2.set_label("dB (red = target louder)", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.savefig(args.out_png, dpi=170, bbox_inches="tight")
    print(f"figure -> {args.out_png}")


if __name__ == "__main__":
    main()
