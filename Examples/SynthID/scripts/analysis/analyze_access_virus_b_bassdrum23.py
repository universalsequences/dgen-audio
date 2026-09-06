#!/usr/bin/env python3
"""Measure the Access Virus B BassDrum_23 target before voice design."""

import argparse
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width == 2:
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - (1 << 24), values)
        signal = values.astype(np.float64) / float(1 << 23)
    elif sample_width == 4:
        signal = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")
    return sample_rate, signal.reshape(-1, channels)


def rms_db(signal):
    return 20.0 * np.log10(np.sqrt(np.mean(signal * signal)) + 1e-15)


def band_spectrum(signal, sample_rate, start_ms, duration_ms):
    start = round(start_ms * sample_rate / 1000.0)
    size = max(64, round(duration_ms * sample_rate / 1000.0))
    segment = np.zeros(size)
    available = signal[start:start + size]
    segment[:len(available)] = available
    window = np.hanning(size)
    spectrum = np.abs(np.fft.rfft(segment * window))
    frequencies = np.fft.rfftfreq(size, 1.0 / sample_rate)
    power = spectrum * spectrum
    total = max(float(np.sum(power)), 1e-30)
    thirds = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    bands = []
    for center in thirds:
        lower, upper = center / 2 ** (1 / 6), center * 2 ** (1 / 6)
        mask = (frequencies >= lower) & (frequencies < upper)
        if np.any(mask):
            bands.append((center, 10 * np.log10(float(np.sum(power[mask])) / total + 1e-15)))
    peaks = []
    candidates = np.argsort(spectrum)[::-1]
    for index in candidates:
        frequency = frequencies[index]
        if frequency < 20:
            continue
        if all(abs(frequency - prior[0]) > 25 for prior in peaks):
            peaks.append((float(frequency), 20 * np.log10(spectrum[index] / (np.max(spectrum) + 1e-30) + 1e-15)))
        if len(peaks) == 8:
            break
    return bands, peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    sample_rate, channels = load_wav(args.target)
    mono = np.mean(channels, axis=1)
    peak = float(np.max(np.abs(mono)))
    print(f"path {args.target}")
    print(f"sample_rate {sample_rate} Hz")
    print(f"shape {channels.shape}; duration {len(mono) / sample_rate * 1000:.3f} ms")
    print(f"peak {20 * np.log10(peak + 1e-30):.3f} dBFS; DC {np.mean(mono):.8f}")
    if channels.shape[1] == 2:
        print(f"L-R max abs {np.max(np.abs(channels[:, 0] - channels[:, 1])):.9f}")
    for threshold_db in (-20, -40, -60):
        indices = np.flatnonzero(np.abs(mono) >= peak * 10 ** (threshold_db / 20))
        print(f"{threshold_db} dB support {indices[0] / sample_rate * 1000:.3f}..{indices[-1] / sample_rate * 1000:.3f} ms")
    print(f"final 20 ms RMS {rms_db(mono[-round(.020 * sample_rate):]):.2f} dBFS")

    onset_candidates = np.flatnonzero(np.abs(mono) >= peak * 1e-3)
    onset = int(onset_candidates[0])
    signal = mono[onset:]
    window = max(8, round(0.0005 * sample_rate))
    hop = max(4, round(0.00025 * sample_rate))
    envelope = []
    for index in range(0, max(1, len(signal) - window + 1), hop):
        envelope.append((index / sample_rate, rms_db(signal[index:index + window])))
    envelope = np.asarray(envelope)
    maxima = []
    for index in range(1, len(envelope) - 1):
        if envelope[index, 1] > envelope[index - 1, 1] and envelope[index, 1] >= envelope[index + 1, 1]:
            if not maxima or envelope[index, 0] - maxima[-1][0] >= 0.001:
                maxima.append(tuple(envelope[index]))
    print("fine-envelope local peaks (first 30 ms):")
    print(" ".join(f"{time * 1000:.2f}ms/{db:.1f}dB" for time, db in maxima if time <= .030))

    print("log-RMS decay fits:")
    for start, end in ((.005, .030), (.030, .080), (.080, .160), (.160, .300), (.300, .450)):
        mask = (envelope[:, 0] >= start) & (envelope[:, 0] <= end)
        if np.count_nonzero(mask) >= 3:
            slope, intercept = np.polyfit(envelope[mask, 0], envelope[mask, 1] / 8.685889638, 1)
            print(f"  {start * 1000:.0f}-{end * 1000:.0f} ms: {slope:.3f} 1/s (T60 {-6.907755 / slope * 1000:.1f} ms)")

    crossings = np.flatnonzero((signal[:-1] < 0) & (signal[1:] >= 0))
    periods = np.diff(crossings) / sample_rate
    frequencies = 1.0 / periods
    times = crossings[:-1] / sample_rate
    print("zero-crossing pitch contour (10 ms bins):")
    points = []
    for start in np.arange(0, min(.35, len(signal) / sample_rate), .010):
        mask = (times >= start) & (times < start + .010) & (frequencies >= 20) & (frequencies <= 500)
        if np.any(mask):
            value = float(np.median(frequencies[mask]))
            points.append((start + .005, value))
            print(f"  {(start + .005) * 1000:6.1f} ms {value:8.3f} Hz")
    if len(points) >= 4:
        points = np.asarray(points)
        best = None
        for decay in np.linspace(-300, -1, 3000):
            design = np.column_stack((np.ones(len(points)), np.exp(decay * points[:, 0])))
            coefficients, *_ = np.linalg.lstsq(design, points[:, 1], rcond=None)
            residual = float(np.sqrt(np.mean((design @ coefficients - points[:, 1]) ** 2)))
            if best is None or residual < best[0]:
                best = residual, decay, coefficients
        print(f"pitch exponential fit: fStart {best[2].sum():.3f} Hz; fEnd {best[2][0]:.3f} Hz; decay {best[1]:.3f} 1/s; RMS {best[0]:.3f} Hz")

    print("time-band spectra (1/3 octave share; peaks relative to strongest):")
    for start_ms, duration_ms in ((0, 10), (10, 30), (40, 60), (100, 100), (250, 150)):
        bands, peaks = band_spectrum(signal, sample_rate, start_ms, duration_ms)
        print(f"  {start_ms:3d}-{start_ms + duration_ms:3d} ms bands: " + " ".join(f"{center:g}Hz={db:.1f}dB" for center, db in bands))
        print(" " * 15 + "peaks: " + " ".join(f"{frequency:.1f}Hz/{db:.1f}dB" for frequency, db in peaks))


if __name__ == "__main__":
    main()
