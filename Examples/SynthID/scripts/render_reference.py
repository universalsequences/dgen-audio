#!/usr/bin/env python3
"""Independent NumPy renderer for SynthID rung 2.

The body and click phase formulas intentionally mirror Patch.swift's closed-form
phase convention. They do not use DGen or call back into the Swift executable.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def render(params, frames=32768, sample_rate=44100, enable_noise_filter=True):
    # Keep the signal path in float32 so the equivalence check measures the
    # independent formulas rather than avoidable float64-vs-float32 drift.
    sample_rate = np.float32(sample_rate)
    t = dgen_time_ramp(frames, sample_rate)
    f_start = np.float32(params["fStart"])
    f_end = np.float32(params["fEnd"])
    pitch_decay = np.float32(params["pitchDecay"])
    two_pi = np.float32(2.0 * math.pi)

    # Patch.swift outputs the accumulator's pre-update float32 state (nominally
    # t[n] = n / sr) and evaluates the exact pitch-envelope integral there.
    sweep_phase = (
        f_end * t
        + (f_start - f_end)
        / pitch_decay
        * (np.exp(pitch_decay * t) - np.float32(1.0))
    )
    body = (
        np.sin(two_pi * sweep_phase)
        * np.exp(np.float32(params["ampDecay"]) * t)
        * np.float32(params["bodyAmp"])
    )
    body_asymmetry = np.float32(params.get("bodyAsymmetry", 0.0))
    even_harmonic = (
        body_asymmetry
        * np.sin(np.float32(2.0) * two_pi * sweep_phase - np.float32(0.62))
        * np.exp(np.float32(params["ampDecay"]) * t)
        * np.float32(params["bodyAmp"])
        * np.exp(np.float32(-17.0) * t)
    )

    click_phase = np.float32(params["clickFreq"]) * t
    click = (
        np.sin(two_pi * click_phase)
        * np.exp(np.float32(params["clickDecay"]) * t)
        * np.float32(params["clickAmp"])
    )

    noise = dgen_noise(frames) * 2.0 - 1.0
    if enable_noise_filter:
        noise = lowpass_biquad(
            noise, params["noiseCutoff"], 0.707, 1.0, float(sample_rate)
        )
    noise_burst = (
        noise
        * np.exp(np.float32(params["noiseDecay"]) * t)
        * np.float32(params["noiseAmp"])
    )

    return (
        np.tanh(
            (body + even_harmonic + click + noise_burst)
            * np.float32(params["drive"])
        )
        * np.float32(params["outGain"])
    ).astype(np.float32)


def dgen_time_ramp(frames, sample_rate):
    """Match Signal.accum(1 / sampleRate)'s pre-update float32 state."""
    values = np.zeros(frames, dtype=np.float32)
    increment = np.float32(np.float32(1.0) / np.float32(sample_rate))
    state = np.float32(0.0)
    for i in range(frames):
        values[i] = state
        state = np.float32(state + increment)
    return values


def dgen_noise(frames):
    values = np.zeros(frames, dtype=np.float32)
    state = 0
    for i in range(frames):
        if state == 0:
            state = 1
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        values[i] = state / 4294967296.0
    return values


def lowpass_biquad(x, cutoff, q, gain, sample_rate):
    # Matches the lowpass coefficient path in DGen's lowered biquad helper.
    w0 = abs(cutoff) * (2.0 * math.pi / float(sample_rate))
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 * 0.5 / abs(q)
    a0 = 1.0 + alpha
    b0 = ((1.0 - cos_w0) * 0.5) * gain / a0
    b1 = (1.0 - cos_w0) * gain / a0
    b2 = ((1.0 - cos_w0) * 0.5) * gain / a0
    a1 = (-2.0 * cos_w0) / a0
    a2 = (1.0 - alpha) / a0

    y = np.zeros_like(x, dtype=np.float32)
    x1 = x2 = y1 = y2 = 0.0
    for i, value in enumerate(x):
        out = b0 * float(value) + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = out
        x2, x1 = x1, float(value)
        y2, y1 = y1, out
    return y


def write_wav(path, samples, sample_rate):
    path = Path(path)
    # Match AudioFile.save's IEEE float32 WAV output. PCM16 quantization is
    # inaudible but creates a large log-STFT floor in otherwise empty bins,
    # making the rung-2 loss-ratio gate impossible even at the true parameters.
    payload = np.clip(samples, -1.0, 1.0).astype("<f4").tobytes()
    channels = 1
    bytes_per_sample = 4
    byte_rate = int(sample_rate) * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(payload))
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            3,  # IEEE float
            channels,
            int(sample_rate),
            byte_rate,
            block_align,
            32,
        )
        + b"data"
        + struct.pack("<I", len(payload))
    )
    path.write_bytes(header + payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=32768)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--no-noise-filter", action="store_true")
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]

    write_wav(
        args.out,
        render(
            params,
            args.frames,
            args.sample_rate,
            enable_noise_filter=not args.no_noise_filter,
        ),
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
