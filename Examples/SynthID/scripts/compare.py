#!/usr/bin/env python3
"""Independent Rung 3 MR-STFT comparison and spectrogram artifact generator."""

import argparse
import json
import os
import struct
import zlib

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


WINDOWS = (256, 512, 1024, 2048)
LOG_EPSILON = 1e-3
REQUIRED_IMPROVEMENT = 0.80
DEFAULT_HIGHPASS_HZ = 30.0


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


def capture_highpass(signal, sample_rate, cutoff_hz):
    """Zero-phase capture-policy high-pass with a short cosine transition."""
    if cutoff_hz <= 0:
        return signal
    pad = min(len(signal) - 1, 8192)
    padded = np.concatenate(
        (signal[pad:0:-1], signal, signal[-2:-pad - 2:-1]))
    frequencies = np.fft.rfftfreq(len(padded), 1.0 / sample_rate)
    low = cutoff_hz * 0.8
    high = cutoff_hz * 1.2
    response = np.clip((frequencies - low) / max(high - low, 1e-12), 0.0, 1.0)
    response = 0.5 - 0.5 * np.cos(np.pi * response)
    filtered = np.fft.irfft(np.fft.rfft(padded) * response, len(padded))
    return filtered[pad:pad + len(signal)].astype(np.float32)


def mrstft(a, b, windows=WINDOWS, epsilon=LOG_EPSILON):
    a, b = same_length(a, b)
    total = 0.0
    for win in windows:
        hop = max(1, win // 4)
        window = np.hanning(win).astype(np.float32)
        # Convert raw FFT magnitudes to sinusoid amplitude. Without coherent-gain
        # normalization, a fixed epsilon moves by 18 dB between 256- and 2048-bin
        # windows and the documented -60 dBFS floor is not actually a dBFS floor.
        magnitude_scale = max(float(window.sum()) / 2.0, 1e-12)
        values = []
        for start in range(0, len(a) - win + 1, hop):
            aa = np.abs(np.fft.rfft(a[start:start + win] * window)) / magnitude_scale
            bb = np.abs(np.fft.rfft(b[start:start + win] * window)) / magnitude_scale
            delta = np.abs(
                np.log(aa + epsilon) - np.log(bb + epsilon))
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

    if plt is None:
        write_comparison_png_fallback(
            path, target_db, initial_db, learned_db, frequencies, sample_rate)
        return

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


def write_comparison_png_fallback(
        path, target_db, initial_db, learned_db, frequencies, sample_rate):
    """Write the required visual artifact with only NumPy + stdlib.

    Matplotlib remains the preferred labeled rendering. This fallback keeps the
    CLI self-contained in Swift/Python environments that have NumPy but no plot
    stack; panels are target, initialization, learned, and target/learned overlay
    in reading order. Frequency is logarithmic from 25 Hz to 8 kHz and time runs
    left-to-right in every panel.
    """
    panel_width = 520
    panel_height = 320
    gap = 8
    canvas = np.zeros(
        (panel_height * 2 + gap, panel_width * 2 + gap, 3), dtype=np.uint8)

    target_level = np.clip((target_db + 80.0) / 80.0, 0.0, 1.0)
    initial_level = np.clip((initial_db + 80.0) / 80.0, 0.0, 1.0)
    learned_level = np.clip((learned_db + 80.0) / 80.0, 0.0, 1.0)

    def resample(values):
        max_hz = min(8000.0, sample_rate / 2.0)
        display_hz = np.geomspace(25.0, max_hz, panel_height)
        rows = np.searchsorted(frequencies, display_hz)
        rows = np.clip(rows, 0, len(frequencies) - 1)
        columns = np.linspace(0, values.shape[1] - 1, panel_width).astype(int)
        # Low frequencies belong at the bottom of the image.
        return values[rows[:, None], columns[None, :]][::-1]

    def magma(level):
        # Compact perceptual-ish black/purple/orange ramp; exact colors are not
        # part of the gate, while the shared scaling across panels is.
        r = np.clip(2.6 * level - 0.55, 0.0, 1.0)
        g = np.clip(2.2 * level - 1.15, 0.0, 1.0)
        b = np.clip(1.8 * level, 0.0, 1.0)
        return np.stack((r, g, b), axis=-1)

    panels = [magma(resample(level)) for level in (
        target_level, initial_level, learned_level)]
    target_panel = resample(target_level)
    learned_panel = resample(learned_level)
    overlay = np.stack(
        (learned_panel, target_panel, np.maximum(target_panel, learned_panel)),
        axis=-1)
    panels.append(overlay)

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        y = row * (panel_height + gap)
        x = column * (panel_width + gap)
        canvas[y:y + panel_height, x:x + panel_width] = np.round(
            np.clip(panel, 0.0, 1.0) * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_rgb_png(path, canvas)


def write_rgb_png(path, pixels):
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype != np.uint8:
        raise ValueError("PNG fallback expects uint8 RGB pixels")

    def chunk(kind, payload):
        body = kind + payload
        return (
            struct.pack(">I", len(payload)) + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    scanlines = b"".join(
        b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.write(chunk(b"IHDR", header))
        output.write(chunk(b"IDAT", zlib.compress(scanlines, level=9)))
        output.write(chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--out", required=True, help="spectrogram comparison PNG")
    parser.add_argument("--json", required=True, help="machine-readable comparison report")
    parser.add_argument("--required-improvement", type=float, default=REQUIRED_IMPROVEMENT)
    parser.add_argument("--highpass-hz", type=float, default=DEFAULT_HIGHPASS_HZ)
    args = parser.parse_args()

    target, target_rate = read_wav(args.target)
    initial, initial_rate = read_wav(args.initial)
    learned, learned_rate = read_wav(args.learned)
    if len({target_rate, initial_rate, learned_rate}) != 1:
        raise SystemExit(
            f"sample-rate mismatch: target={target_rate}, initial={initial_rate}, learned={learned_rate}")
    target, initial, learned = same_length(target, initial, learned)
    target = capture_highpass(target, target_rate, args.highpass_hz)
    initial = capture_highpass(initial, target_rate, args.highpass_hz)
    learned = capture_highpass(learned, target_rate, args.highpass_hz)

    initial_distance = mrstft(initial, target)
    learned_distance = mrstft(learned, target)
    improvement = 1.0 - learned_distance / max(initial_distance, 1e-12)
    report = {
        "initialDistance": initial_distance,
        "learnedDistance": learned_distance,
        "improvement": improvement,
        "requiredImprovement": args.required_improvement,
        "logEpsilon": LOG_EPSILON,
        "magnitudeNormalization": "hann coherent gain (sum(window) / 2)",
        "windows": list(WINDOWS),
        "highpassHz": args.highpass_hz,
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
