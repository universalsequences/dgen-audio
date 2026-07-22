#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def read_wav(path):
    data = Path(path).read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    offset = 12
    fmt = None
    pcm = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        if chunk_id == b"fmt ":
            audio_format, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, start
            )
            fmt = (audio_format, channels, sample_rate, bits)
        elif chunk_id == b"data":
            pcm = data[start:end]
        offset = end + (size & 1)

    if fmt is None or pcm is None:
        raise ValueError(f"{path}: missing fmt or data chunk")

    audio_format, channels, sample_rate, bits = fmt
    if audio_format == 1 and bits == 16:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    elif audio_format == 1 and bits == 24:
        vals = []
        for i in range(len(pcm) // 3):
            b0, b1, b2 = pcm[i * 3 : i * 3 + 3]
            raw = b0 | (b1 << 8) | (b2 << 16)
            if raw & 0x800000:
                raw |= ~0xFFFFFF
            vals.append(raw / 8388608.0)
        samples = np.asarray(vals, dtype=np.float64)
    elif audio_format == 3 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported WAV format={audio_format} bits={bits}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def normalize(x):
    peak = np.max(np.abs(x)) if len(x) else 0.0
    return x / peak if peak > 1e-12 else x.copy()


def align(a, b):
    n = min(len(a), len(b))
    return normalize(a[:n]), normalize(b[:n])


def stft_log_distance(a, b, sr, window, hop, end_ms=90.0):
    n = min(len(a), len(b), int(sr * end_ms / 1000.0))
    if n < window:
        return 0.0
    win = np.hanning(window)
    total = 0.0
    weight_sum = 0.0
    for start in range(0, n - window + 1, hop):
        mid_ms = (start + window * 0.5) / sr * 1000.0
        if mid_ms < 10.0:
            w = 2.4
        elif mid_ms < 35.0:
            w = 1.8
        else:
            w = 1.0
        aa = np.fft.rfft(a[start : start + window] * win)
        bb = np.fft.rfft(b[start : start + window] * win)
        la = np.log1p(np.abs(aa) * 80.0)
        lb = np.log1p(np.abs(bb) * 80.0)
        freqs = np.fft.rfftfreq(window, 1.0 / sr)
        band_weight = np.ones_like(freqs)
        band_weight[(freqs >= 1000.0) & (freqs < 4000.0)] *= 1.35
        band_weight[(freqs >= 4000.0) & (freqs < 16000.0)] *= 1.55
        dist = np.sqrt(np.mean(((la - lb) * band_weight) ** 2))
        total += dist * w
        weight_sum += w
    return total / max(weight_sum, 1e-12)


def biquad_band(x, sr, low, high):
    # Offline approximate Butterworth-ish one-pole cascades are enough for evaluator bands.
    y = x.copy()
    if low > 0:
        y = highpass(y, sr, low)
        y = highpass(y, sr, low)
    if high > low:
        y = lowpass(y, sr, high)
        y = lowpass(y, sr, high)
    return y


def lowpass(x, sr, cutoff):
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sr
    alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = prev + alpha * (v - prev)
        y[i] = prev
    return y


def highpass(x, sr, cutoff):
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    y = np.zeros_like(x)
    prev_y = 0.0
    prev_x = 0.0
    for i, v in enumerate(x):
        cur = alpha * (prev_y + v - prev_x)
        y[i] = cur
        prev_y = cur
        prev_x = v
    return y


def rms_db(x, eps=1e-12):
    return 20.0 * math.log10(float(np.sqrt(np.mean(x * x)) + eps))


def spectral_band_envelope(x, sr, low, high, start_ms, end_ms, window=256, hop=64):
    start = int(sr * start_ms / 1000.0)
    end = min(len(x), int(sr * end_ms / 1000.0))
    segment = x[start:end]
    if len(segment) < 2:
        return np.asarray([0.0], dtype=np.float64)
    if len(segment) < window:
        segment = np.pad(segment, (0, window - len(segment)))

    win = np.hanning(window)
    freqs = np.fft.rfftfreq(window, 1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.asarray([0.0], dtype=np.float64)

    env = []
    for frame_start in range(0, len(segment) - window + 1, hop):
        spec = np.fft.rfft(segment[frame_start : frame_start + window] * win)
        mag = np.abs(spec[mask])
        env.append(float(np.sqrt(np.mean(mag * mag) + 1e-18)))
    return np.asarray(env if env else [0.0], dtype=np.float64)


def band_delta(a, b, sr, low, high, start_ms, end_ms):
    aa = spectral_band_envelope(a, sr, low, high, start_ms, end_ms)
    bb = spectral_band_envelope(b, sr, low, high, start_ms, end_ms)
    return rms_db(aa) - rms_db(bb)


def envelope_distance(a, b, sr, low, high, start_ms, end_ms):
    if high <= 250.0:
        window, hop = 2048, 256
    elif high <= 1000.0:
        window, hop = 1024, 128
    else:
        window, hop = 256, 64
    aa = spectral_band_envelope(a, sr, low, high, start_ms, end_ms, window=window, hop=hop)
    bb = spectral_band_envelope(b, sr, low, high, start_ms, end_ms, window=window, hop=hop)
    n = min(len(aa), len(bb))
    aa = aa[:n]
    bb = bb[:n]
    scale = max(np.max(aa), np.max(bb), 1e-9)
    return float(np.sqrt(np.mean(((aa - bb) / scale) ** 2)))


def zero_crossings(x, sr, end_ms):
    n = min(len(x), int(sr * end_ms / 1000.0))
    s = np.signbit(x[:n])
    return int(np.count_nonzero(s[1:] != s[:-1]))


def metrics(learned, target, sr):
    learned, target = align(learned, target)
    d = {}
    d["mrstft_64"] = stft_log_distance(learned, target, sr, 64, 16)
    d["mrstft_128"] = stft_log_distance(learned, target, sr, 128, 32)
    d["mrstft_256"] = stft_log_distance(learned, target, sr, 256, 64)
    d["mrstft_512"] = stft_log_distance(learned, target, sr, 512, 128)
    d["mrstft_1024"] = stft_log_distance(learned, target, sr, 1024, 256)
    d["mrstft"] = (
        d["mrstft_64"] * 1.4
        + d["mrstft_128"] * 1.4
        + d["mrstft_256"] * 1.2
        + d["mrstft_512"]
        + d["mrstft_1024"] * 0.8
    ) / 5.8

    d["body_env"] = envelope_distance(learned, target, sr, 30.0, 900.0, 0.0, 85.0)
    d["sub_env"] = envelope_distance(learned, target, sr, 30.0, 180.0, 0.0, 85.0)
    d["crisp_env"] = envelope_distance(learned, target, sr, 4000.0, 16000.0, 0.0, 35.0)
    d["transient_env"] = envelope_distance(learned, target, sr, 1000.0, 16000.0, 0.0, 10.0)

    d["presence_delta"] = band_delta(learned, target, sr, 1000.0, 4000.0, 0.0, 30.0)
    d["air_delta"] = band_delta(learned, target, sr, 4000.0, 12000.0, 0.0, 30.0)
    d["hf_delta"] = band_delta(learned, target, sr, 2000.0, 16000.0, 0.0, 30.0)
    d["air_0_5_delta"] = band_delta(learned, target, sr, 4000.0, 12000.0, 0.0, 5.0)
    d["air_5_30_delta"] = band_delta(learned, target, sr, 4000.0, 12000.0, 5.0, 30.0)

    d["norm_mse"] = float(np.mean((learned - target) ** 2))
    d["transient_mse"] = float(np.mean((learned[:512] - target[:512]) ** 2))
    d["learned_zc_30"] = zero_crossings(learned, sr, 30.0)
    d["target_zc_30"] = zero_crossings(target, sr, 30.0)
    d["zc_delta"] = d["learned_zc_30"] - d["target_zc_30"]

    top_balance = (
        abs(d["presence_delta"]) * 0.08
        + abs(d["air_delta"]) * 0.14
        + abs(d["hf_delta"]) * 0.10
        + max(0.0, d["air_0_5_delta"] - 8.0) * 0.18
        + max(0.0, -d["air_5_30_delta"]) * 0.16
    )
    d["perceptual_score"] = (
        d["mrstft"] * 2.4
        + d["body_env"] * 1.3
        + d["sub_env"] * 0.8
        + d["crisp_env"] * 1.2
        + d["transient_env"] * 0.9
        + top_balance
        + d["norm_mse"] * 3.0
        + d["transient_mse"] * 2.0
        + abs(d["zc_delta"]) * 0.08
    )
    return d


def apply_gates(d, baseline=None):
    gates = {}
    gates["zero_crossings"] = abs(d["zc_delta"]) <= 4
    gates["front_air"] = d["air_0_5_delta"] <= 10.0
    gates["presence_not_worse"] = d["presence_delta"] <= 18.0
    gates["sustained_air"] = d["air_5_30_delta"] >= -4.0
    gates["waveform"] = d["norm_mse"] <= (baseline["norm_mse"] * 1.35 if baseline else 0.018)
    gates["transient"] = d["transient_mse"] <= (baseline["transient_mse"] * 1.45 if baseline else 0.04)
    gates["non_oracle"] = d["norm_mse"] >= 1e-4
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    learned, learned_sr = read_wav(args.learned)
    target, target_sr = read_wav(args.target)
    sr = min(learned_sr, target_sr)
    baseline = None
    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text())
    d = metrics(learned, target, sr)
    gates = apply_gates(d, baseline=baseline)
    d["gates"] = gates
    d["passes_gates"] = all(gates.values())
    if baseline:
        d["baseline_score"] = baseline["perceptual_score"]
        d["improvement"] = (baseline["perceptual_score"] - d["perceptual_score"]) / baseline["perceptual_score"]

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(d, indent=2, sort_keys=True))
    print(f"perceptual_score={d['perceptual_score']:.6f}")
    if baseline:
        print(f"improvement={d['improvement']:.6f}")
    print(f"passes_gates={str(d['passes_gates']).lower()}")
    for key in [
        "mrstft", "body_env", "sub_env", "crisp_env", "transient_env",
        "presence_delta", "air_delta", "hf_delta", "air_0_5_delta", "air_5_30_delta",
        "norm_mse", "transient_mse", "learned_zc_30", "target_zc_30", "zc_delta",
    ]:
        value = d[key]
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")
    for key, ok in gates.items():
        print(f"gate_{key}={str(ok).lower()}")


if __name__ == "__main__":
    main()
