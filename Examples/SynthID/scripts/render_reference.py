#!/usr/bin/env python3
"""Reference numpy renderer for SynthID rung 2."""

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def render(params, frames=32768, sample_rate=44100):
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    f_start = params["fStart"]
    f_end = params["fEnd"]
    pitch = f_end + (f_start - f_end) * np.exp(params["pitchDecay"] * t)
    phase = phasor(pitch, sample_rate)
    body = np.sin(2.0 * math.pi * phase) * np.exp(params["ampDecay"] * t) * params["bodyAmp"]

    click_phase = phasor(np.full(frames, params["clickFreq"], dtype=np.float32), sample_rate)
    click = np.sin(2.0 * math.pi * click_phase) * np.exp(params["clickDecay"] * t) * params["clickAmp"]

    noise = dgen_noise(frames)
    noise = lowpass_biquad(noise, params["noiseCutoff"], 0.707, 1.0, sample_rate)
    noise_burst = noise * np.exp(params["noiseDecay"] * t) * params["noiseAmp"]

    return np.tanh((body + click + noise_burst) * params["drive"]) * params["outGain"]


def phasor(freq, sample_rate):
    phase = np.zeros_like(freq, dtype=np.float32)
    state = 0.0
    for i, f in enumerate(freq):
        phase[i] = state
        state = (state + float(f) / float(sample_rate)) % 1.0
    return phase


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
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        f.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=32768)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]

    write_wav(args.out, render(params, args.frames, args.sample_rate), args.sample_rate)


if __name__ == "__main__":
    main()
