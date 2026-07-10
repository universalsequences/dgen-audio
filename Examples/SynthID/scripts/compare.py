#!/usr/bin/env python3
"""Independent Rung 3 MR-STFT comparison and spectrogram artifact generator."""

import argparse
import json
import os
import struct

import matplotlib.pyplot as plt
import numpy as np


WINDOWS = (256, 512, 1024, 2048)
LOG_EPSILON = 1e-3
REQUIRED_IMPROVEMENT = 0.80


def read_wav(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    offset = 12
    fmt = None
    data_chunk = None
    while offset + 8 <= len(blob):
        chunk_id = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        end = start + size
        if chunk_id == b"fmt ":
            audio_format, channels, rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", blob, start)
            fmt = audio_format, channels, rate, bits
        elif chunk_id == b"data":
            data_chunk = blob[start:end]
        offset = end + (size & 1)

    if fmt is None or data_chunk is None:
        raise ValueError(f"{path} is missing fmt or data chunk")

    audio_format, channels, rate, bits = fmt
    if audio_format == 1 and bits == 16:
        data = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 1 and bits == 24:
        raw = np.frombuffer(data_chunk, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        data = values.astype(np.float32) / 8388608.0
    elif audio_format == 3 and bits == 32:
        data = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format={audio_format} bits={bits}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def same_length(*signals):
    n = min(map(len, signals))
    if n < max(WINDOWS):
        raise ValueError(f"comparison signals are too short: {n} frames")
    return tuple(signal[:n] for signal in signals)


def mrstft(a, b, windows=WINDOWS, epsilon=LOG_EPSILON):
    a, b = same_length(a, b)
    total = 0.0
    for win in windows:
        hop = max(1, win // 4)
        window = np.hanning(win).astype(np.float32)
        values = []
        for start in range(0, len(a) - win + 1, hop):
            aa = np.fft.rfft(a[start:start + win] * window)
            bb = np.fft.rfft(b[start:start + win] * window)
            delta = np.abs(
                np.log(np.abs(aa) + epsilon) - np.log(np.abs(bb) + epsilon))
            values.append(float(np.mean(delta)))
        total += float(np.mean(values))
    return total


def spectrogram(signal, sample_rate, window_size=2048, hop=128):
    window = np.hanning(window_size).astype(np.float32)
    frames = []
    for start in range(0, len(signal) - window_size + 1, hop):
        frame = np.fft.rfft(signal[start:start + window_size] * window)
        frames.append(np.abs(frame))
    magnitude = np.stack(frames, axis=1)
    frequencies = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    times = (np.arange(magnitude.shape[1]) * hop + window_size / 2) / sample_rate
    return magnitude, frequencies, times


def normalized_db(magnitude, reference):
    return np.clip(20.0 * np.log10(np.maximum(magnitude, 1e-8) / reference), -80.0, 0.0)


def write_comparison_png(path, target, initial, learned, sample_rate):
    target_mag, frequencies, times = spectrogram(target, sample_rate)
    initial_mag, _, _ = spectrogram(initial, sample_rate)
    learned_mag, _, _ = spectrogram(learned, sample_rate)
    reference = max(float(target_mag.max()), float(initial_mag.max()), float(learned_mag.max()), 1e-8)
    target_db = normalized_db(target_mag, reference)
    initial_db = normalized_db(initial_mag, reference)
    learned_db = normalized_db(learned_mag, reference)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    panels = (
        (axes[0, 0], target_db, "Target"),
        (axes[0, 1], initial_db, "Initialization"),
        (axes[1, 0], learned_db, "Learned"),
    )
    for axis, values, title in panels:
        axis.pcolormesh(times, frequencies, values, shading="auto", cmap="magma", vmin=-80, vmax=0)
        axis.set_title(f"{title} (-80 to 0 dB)")
        axis.set_yscale("log")
        axis.set_ylim(25, min(8000, sample_rate / 2))
        axis.grid(alpha=0.15, which="both")

    # Literal overlay: target contributes cyan, learned contributes magenta,
    # and matching energy appears white.
    target_level = np.clip((target_db + 80.0) / 80.0, 0.0, 1.0)
    learned_level = np.clip((learned_db + 80.0) / 80.0, 0.0, 1.0)
    overlay = np.stack(
        [learned_level, target_level, np.maximum(target_level, learned_level)], axis=-1)
    axes[1, 1].imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], frequencies[0], frequencies[-1]))
    axes[1, 1].set_title("Overlay: target cyan, learned magenta, match white")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(25, min(8000, sample_rate / 2))
    axes[1, 1].grid(alpha=0.15, which="both")

    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency (Hz)")
    for axis in axes[1, :]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("SynthID Rung 3 spectral comparison")
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.91, wspace=0.14, hspace=0.20)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--out", required=True, help="spectrogram comparison PNG")
    parser.add_argument("--json", required=True, help="machine-readable comparison report")
    parser.add_argument("--required-improvement", type=float, default=REQUIRED_IMPROVEMENT)
    args = parser.parse_args()

    target, target_rate = read_wav(args.target)
    initial, initial_rate = read_wav(args.initial)
    learned, learned_rate = read_wav(args.learned)
    if len({target_rate, initial_rate, learned_rate}) != 1:
        raise SystemExit(
            f"sample-rate mismatch: target={target_rate}, initial={initial_rate}, learned={learned_rate}")
    target, initial, learned = same_length(target, initial, learned)

    initial_distance = mrstft(initial, target)
    learned_distance = mrstft(learned, target)
    improvement = 1.0 - learned_distance / max(initial_distance, 1e-12)
    report = {
        "initialDistance": initial_distance,
        "learnedDistance": learned_distance,
        "improvement": improvement,
        "requiredImprovement": args.required_improvement,
        "logEpsilon": LOG_EPSILON,
        "windows": list(WINDOWS),
        "pass": improvement >= args.required_improvement,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    write_comparison_png(args.out, target, initial, learned, target_rate)
    print(
        f"initial_mrstft={initial_distance:.6f} learned_mrstft={learned_distance:.6f} "
        f"improvement={improvement:.2%} pass={str(report['pass']).lower()}")


if __name__ == "__main__":
    main()
