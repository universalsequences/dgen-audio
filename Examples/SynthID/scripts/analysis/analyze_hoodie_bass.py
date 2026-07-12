#!/usr/bin/env python3
"""Measure the eseq sample set tagged `hoodie bass` for SynthID."""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def mono_float(path: Path):
    sample_rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    stereo_delta = float(np.max(np.abs(audio[:, 0] - audio[:, 1]))) if audio.ndim == 2 else 0.0
    return sample_rate, audio.mean(axis=1) if audio.ndim == 2 else audio, stereo_delta


def estimate_f0(audio, sample_rate):
    segment = audio[int(0.25 * sample_rate):int(1.5 * sample_rate)]
    fft_size = 131072
    magnitude = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), fft_size))
    frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    candidates = np.where((frequencies >= 25) & (frequencies <= 260))[0]
    score = np.zeros(len(candidates))
    for harmonic in range(1, 9):
        score += magnitude[np.minimum(candidates * harmonic, len(magnitude) - 1)] / harmonic
    return float(frequencies[candidates[np.argmax(score)]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eseq", type=Path, help="path to the eseq repository")
    args = parser.parse_args()
    db_path = args.eseq / "crates/sequencer/samples.db"
    sample_dir = args.eseq / "crates/sequencer/samples"
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """SELECT s.hash, s.title FROM samples s
               JOIN sample_tags st ON st.sample_id = s.id
               JOIN tags t ON t.id = st.tag_id
               WHERE lower(t.name) = 'hoodie bass' ORDER BY s.id""").fetchall()
    for sample_hash, title in rows:
        rate, audio, stereo_delta = mono_float(sample_dir / f"{sample_hash}.wav")
        print(
            f"{title:>3} {sample_hash[:8]} rate={rate} frames={len(audio)} "
            f"peak={np.max(np.abs(audio)):.6f} dc={np.mean(audio):+.3e} "
            f"stereoDelta={stereo_delta:.3e} f0={estimate_f0(audio, rate):.3f}")


if __name__ == "__main__":
    main()
