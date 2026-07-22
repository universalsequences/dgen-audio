#!/usr/bin/env python3
"""Independent NumPy renderer for SynthID rung 2.

The profile formulas intentionally mirror their deployment math and phase
conventions. They do not use DGen or call back into the Swift executable.
"""

import argparse
import functools
import json
import math
import struct
from pathlib import Path

import numpy as np

TR909_HARMONIC_CORRECTIONS = [
    ("d0h10s", 10, 0, False), ("d0h12s", 12, 0, False), ("d0h16c", 16, 0, True),
    ("d0h2c", 2, 0, True), ("d0h2s", 2, 0, False), ("d0h3c", 3, 0, True),
    ("d0h3s", 3, 0, False), ("d0h4c", 4, 0, True), ("d0h5c", 5, 0, True),
    ("d0h5s", 5, 0, False), ("d0h6s", 6, 0, False), ("d0h7c", 7, 0, True),
    ("d0h8c", 8, 0, True), ("d0h9c", 9, 0, True), ("d0h9s", 9, 0, False),
    ("d15h14s", 14, 15, False), ("d15h2c", 2, 15, True), ("d15h3c", 3, 15, True),
    ("d15h3s", 3, 15, False), ("d15h5c", 5, 15, True), ("d15h5s", 5, 15, False),
    ("d15h6c", 6, 15, True), ("d15h9s", 9, 15, False),
    ("d15h10s", 10, 15, False), ("d15h16s", 16, 15, False),
    ("d15h2s", 2, 15, False), ("d240h12s", 12, 240, False),
    ("d240h15c", 15, 240, True), ("d240h3s", 3, 240, False),
    ("d240h4s", 4, 240, False), ("d240h3c", 3, 240, True),
    ("d60h10s", 10, 60, False),
    ("d60h12s", 12, 60, False), ("d60h14c", 14, 60, True),
    ("d60h16c", 16, 60, True), ("d60h2c", 2, 60, True),
    ("d60h2s", 2, 60, False), ("d60h3c", 3, 60, True),
    ("d60h5s", 5, 60, False), ("d60h6s", 6, 60, False),
]

HOODIE_BASS_HARMONICS = [
    (name, harmonic, decay, cosine)
    for harmonic in range(1, 33)
    for name, decay, cosine in (
        (f"h{harmonic}s", 0, False), (f"h{harmonic}c", 0, True))
] + [
    (name, harmonic, decay, cosine)
    for harmonic in range(2, 33)
    for name, decay, cosine in (
        (f"mh{harmonic}s", 2, False), (f"mh{harmonic}c", 2, True))
] + [
    (name, harmonic, decay, cosine)
    for harmonic in range(2, 33)
    for name, decay, cosine in (
        (f"bh{harmonic}s", 1, False), (f"bh{harmonic}c", 1, True))
] + [
    (name, harmonic, decay, cosine)
    for harmonic in range(2, 33)
    for name, decay, cosine in (
        (f"fh{harmonic}s", 4, False), (f"fh{harmonic}c", 4, True))
]


def render(
        params, frames=32768, sample_rate=44100, enable_noise_filter=True,
        profile=None, oscillator_only=False):
    # Keep the signal path in float32 so the equivalence check measures the
    # independent formulas rather than avoidable float64-vs-float32 drift.
    sample_rate = np.float32(sample_rate)
    t = dgen_time_ramp(frames, sample_rate)
    if profile == "subtractive-bass":
        return render_subtractive_bass(
            params, t, sample_rate, enable_noise_filter, oscillator_only)
    if profile == "monologue-bass":
        return render_monologue_bass(
            params, t, sample_rate, enable_noise_filter, oscillator_only)
    if profile == "hoodie-bass":
        return render_hoodie_bass(params, t)
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
    amp_curve = np.float32(params.get("ampCurve", 0.0))
    body_env = np.exp(np.float32(params["ampDecay"]) * t + amp_curve * (t * t))
    body = (
        np.sin(two_pi * sweep_phase)
        * body_env
        * np.float32(params["bodyAmp"])
    )
    body_asymmetry = np.float32(params.get("bodyAsymmetry", 0.0))
    even_harmonic = (
        body_asymmetry
        * np.sin(np.float32(2.0) * two_pi * sweep_phase - np.float32(0.62))
        * body_env
        * np.float32(params["bodyAmp"])
        * np.exp(np.float32(-17.0) * t)
    )

    body_harmonic = np.float32(params.get("bodyHarmonic", 0.0))
    odd_harmonics = (
        body_harmonic
        * (
            np.sin(np.float32(3.0) * two_pi * sweep_phase) * np.float32(1.0 / 9.0)
            + np.sin(np.float32(5.0) * two_pi * sweep_phase) * np.float32(1.0 / 25.0)
        )
        * body_env
        * np.float32(params["bodyAmp"])
    )

    corrections = params.get("harmonicCorrections", {})
    harmonic_correction = np.zeros(frames, dtype=np.float32)
    for name, harmonic, decay, cosine in TR909_HARMONIC_CORRECTIONS:
        coefficient = np.float32(corrections.get(name, params.get(name, 0.0)))
        wave = np.cos(np.float32(harmonic) * two_pi * sweep_phase) if cosine else np.sin(
            np.float32(harmonic) * two_pi * sweep_phase)
        harmonic_correction += (coefficient * wave * body_env
                                * np.float32(params["bodyAmp"])
                                * np.exp(np.float32(-decay) * t))

    click_phase = np.float32(params["clickFreq"]) * t
    click = (
        np.sin(two_pi * click_phase)
        * np.exp(np.float32(params["clickDecay"]) * t)
        * np.float32(params["clickAmp"])
    )

    if enable_noise_filter:
        noise = filtered_dgen_noise(
            frames, float(params["noiseCutoff"]), float(sample_rate))
    else:
        noise = dgen_noise(frames) * 2.0 - 1.0
    noise_burst = (
        noise
        * np.exp(np.float32(params["noiseDecay"]) * t)
        * np.float32(params["noiseAmp"])
    )

    mixed = body + even_harmonic + odd_harmonics + harmonic_correction + click + noise_burst
    is_909 = profile == "909" or (profile is None and bool(params.get("harmonicCorrections")))
    if is_909:
        bias = np.float32(0.05)
        shifted = mixed * np.float32(params["drive"]) + bias
        shaped = shifted / (np.float32(1.0) + np.abs(shifted))
        shaped -= bias / (np.float32(1.0) + abs(bias))
        return (shaped * np.float32(params["outGain"])).astype(np.float32)
    return (np.tanh(mixed * np.float32(params["drive"]))
            * np.float32(params["outGain"])).astype(np.float32)


def render_hoodie_bass(params, t):
    """Independent mirror of Patch.swift's additive Hoodie Bass voice."""
    two_pi = np.float32(2.0 * math.pi)
    phase = np.float32(params["f0"]) * t * two_pi
    attack = np.float32(1.0) - np.exp(-t / np.float32(params["attackTime"]))
    sustain = np.float32(params["sustain"])
    decay = sustain + (np.float32(1.0) - sustain) * np.exp(
        -t / np.float32(params["decayTime"]))
    release = np.float32(1.0) / (
        np.float32(1.0)
        + np.exp((t - np.float32(params["noteOff"])) / np.float32(params["releaseTime"])))
    amplitude_envelope = attack * decay * release

    coefficients = params.get("harmonicCorrections", {})
    oscillator = np.zeros(len(t), dtype=np.float32)
    brightness_decay = np.float32(params["brightnessDecay"])
    for name, harmonic, decay, cosine in HOODIE_BASS_HARMONICS:
        coefficient = np.float32(coefficients.get(name, params.get(name, 0.0)))
        angle = np.float32(harmonic) * phase
        wave = np.cos(angle) if cosine else np.sin(angle)
        brightness = np.exp(-brightness_decay * np.float32(decay) * t)
        oscillator += coefficient * wave * brightness

    driven = oscillator * amplitude_envelope * np.float32(params["drive"])
    shaped = driven / (np.float32(1.0) + np.abs(driven))
    return (shaped * np.float32(params["outGain"])).astype(np.float32)


def render_subtractive_bass(
        params, t, sample_rate, enable_filter=True, oscillator_only=False):
    """Independent float32 transcription of the deployment PolyBLEP voice."""
    # subF0/subNoteOff are frozen documented scalars (pitch fit + measured
    # note-off); defaults preserve legacy artifacts' 110 Hz / 0.6 s constants.
    frequency = np.float32(params.get("subF0", 110.0))
    phase = dgen_phasor(len(t), frequency, sample_rate)
    dt = np.clip(frequency / sample_rate, np.float32(0.000001), np.float32(0.5))

    saw = phase * np.float32(2.0) - np.float32(1.0) - polyblep(phase, dt)
    width = np.clip(
        np.float32(params["pw"]), np.float32(0.01), np.float32(0.99))
    falling_phase = np.mod(phase - width, np.float32(1.0)).astype(np.float32)
    raw_pulse = (phase < width).astype(np.float32) * np.float32(2.0) - np.float32(1.0)
    pulse = raw_pulse + polyblep(phase, dt) - polyblep(falling_phase, dt)
    shape = np.float32(params["shape"])
    oscillator = ((np.float32(1.0) - shape) * saw + shape * pulse).astype(np.float32)
    if oscillator_only:
        return oscillator

    cutoff = (
        np.float32(params["fBase"])
        + np.float32(params["fAmt"])
        * np.exp(-t / np.float32(params["fDecay"])))
    filtered = time_varying_lowpass_biquad(
        oscillator, cutoff, np.float32(params["res"])) if enable_filter else oscillator

    attack = np.float32(1.0) - np.exp(-t / np.float32(params["attackTime"]))
    sustain = np.float32(params["sustain"])
    decay = sustain + (np.float32(1.0) - sustain) * np.exp(
        -t / np.float32(params["decayTime"]))
    release = np.float32(1.0) / (
        np.float32(1.0)
        + np.exp((t - np.float32(params.get("subNoteOff", 0.6)))
                 / np.float32(params["releaseTime"])))
    driven = filtered * attack * decay * release * np.float32(params["drive"])
    shaped = driven / (np.float32(1.0) + np.abs(driven))
    return (shaped * np.float32(params["outGain"])).astype(np.float32)


def render_monologue_bass(params, t, sample_rate, enable_filter=True,
                          oscillator_only=False):
    """Independent float32 transcription of MonologueVoice (Patch.swift):
    two detuned polyblep VCOs -> polynomial saturator -> ZDF SVF lowpass
    with softsign-saturated integrator states -> VCA -> softsign drive."""
    one = np.float32(1.0)

    def osc(phase, freq):
        dt = np.clip(np.float32(freq) / sample_rate,
                     np.float32(0.000001), np.float32(0.5))
        saw = phase * np.float32(2.0) - one - polyblep(phase, dt)
        width = np.clip(np.float32(params["pw"]), np.float32(0.01), np.float32(0.99))
        falling = np.mod(phase - width, one).astype(np.float32)
        raw_pulse = (phase < width).astype(np.float32) * np.float32(2.0) - one
        pulse = raw_pulse + polyblep(phase, dt) - polyblep(falling, dt)
        shape = np.float32(params["shape"])
        return ((one - shape) * saw + shape * pulse).astype(np.float32)

    f0 = np.float32(params.get("subF0", 110.0))
    # VCO2 phase is the closed-form offset wrap(phase1 - t*detune), matching
    # MonologueVoice (a trainable statefulPhasor frequency corrupts unrelated
    # gradients; see SVFBPTTScratchTests).
    phase1 = dgen_phasor(len(t), float(f0), float(sample_rate))
    detune = np.float32(params.get("vco2Detune", 0.316))
    phase2 = np.mod(phase1 - t * detune, one).astype(np.float32)
    osc1 = osc(phase1, f0)
    osc2 = osc(phase2, f0)
    lvl = np.float32(params.get("vco2Level", 0.0))
    mixed = ((osc1 + lvl * osc2) / (one + lvl)).astype(np.float32)
    if oscillator_only:
        return mixed

    g_sat = np.float32(params.get("satGain", 1.0))
    bias = np.float32(params.get("satBias", 0.0))
    a2s = np.float32(params.get("satA2", 0.0))
    a3s = np.float32(params.get("satA3", 0.0))
    a5s = np.float32(params.get("satA5", 0.0))
    y = g_sat * mixed + bias
    shaped_pre = y + a2s * (y * y) + a3s * (y * y * y) + a5s * (y * y * y * y * y)
    dc = bias + a2s * (bias * bias) + a3s * (bias * bias * bias) \
        + a5s * (bias * bias * bias * bias * bias)
    pre = (shaped_pre - dc).astype(np.float32)

    cutoff = np.clip(
        (np.float32(params["fBase"])
         + np.float32(params["fAmt"]) * np.exp(-t / np.float32(params["fDecay"]))),
        np.float32(20.0), np.float32(sample_rate) * np.float32(0.49)).astype(np.float32)
    if enable_filter:
        filtered = zdf_svf_lowpass_sat(
            pre, cutoff, np.float32(params["res"]),
            np.float32(params.get("filtSat", 0.0)), np.float32(sample_rate))
    else:
        filtered = pre

    attack = one - np.exp(-t / np.float32(params["attackTime"]))
    sustain = np.float32(params["sustain"])
    decay = sustain + (one - sustain) * np.exp(-t / np.float32(params["decayTime"]))
    release = one / (one + np.exp(
        (t - np.float32(params.get("subNoteOff", 0.6)))
        / np.float32(params["releaseTime"])))
    driven = filtered * attack * decay * release * np.float32(params["drive"])
    shaped = driven / (one + np.abs(driven))
    return (shaped * np.float32(params["outGain"])).astype(np.float32)


def zdf_svf_lowpass_sat(x, cutoff, q, k_sat, sample_rate):
    """Cytomic ZDF SVF (LP output) with softsign-saturated integrator state
    reads; k_sat=0 is the exact linear SVF. Float32 throughout to mirror the
    Metal render."""
    y = np.zeros_like(x, dtype=np.float32)
    ic1 = np.float32(0.0)
    ic2 = np.float32(0.0)
    one = np.float32(1.0)
    two = np.float32(2.0)
    k_damp = one / np.abs(np.float32(q))
    k_sat = np.float32(k_sat)
    pi_over_sr = np.float32(np.pi) / np.float32(sample_rate)
    for i in range(len(x)):
        g = np.tan(np.float32(cutoff[i]) * pi_over_sr, dtype=np.float32)
        a1 = one / (one + g * (g + k_damp))
        a2 = g * a1
        a3 = g * a2
        s1 = ic1 / (one + np.abs(k_sat * ic1))
        s2 = ic2 / (one + np.abs(k_sat * ic2))
        v3 = np.float32(x[i]) - s2
        v1 = a1 * s1 + a2 * v3
        v2 = s2 + a2 * s1 + a3 * v3
        ic1 = np.float32(two * v1 - s1)
        ic2 = np.float32(two * v2 - s2)
        y[i] = v2
    return y


def polyblep(phase, dt):
    """The eseq macro, including its strict lt/gt multiplicative gates."""
    left_x = phase / dt
    left = (
        np.float32(2.0) * left_x - left_x * left_x - np.float32(1.0))
    right_x = (phase - np.float32(1.0)) / dt
    right = right_x * right_x + np.float32(2.0) * right_x + np.float32(1.0)
    return (
        (phase < dt).astype(np.float32) * left
        + (phase > (np.float32(1.0) - dt)).astype(np.float32) * right).astype(np.float32)


@functools.lru_cache(maxsize=16)
def dgen_phasor(frames, frequency, sample_rate):
    """Match statefulPhasor's pre-update float32 accumulator output."""
    values = np.zeros(frames, dtype=np.float32)
    increment = np.float32(np.float32(frequency) / np.float32(sample_rate))
    state = np.float32(0.0)
    for i in range(frames):
        values[i] = state
        state = np.float32(state + increment)
        if state >= np.float32(1.0):
            state = np.float32(state - np.float32(1.0))
    return values


def time_varying_lowpass_biquad(x, cutoff, q):
    """Match DGen HigherOps' per-frame mode-0 coefficient graph."""
    y = np.zeros_like(x, dtype=np.float32)
    x1 = np.float32(0.0)
    x2 = np.float32(0.0)
    y1 = np.float32(0.0)
    y2 = np.float32(0.0)
    angular_scale = np.float32(0.00014247585730565955)
    one = np.float32(1.0)
    two = np.float32(2.0)
    half = np.float32(0.5)
    q = np.abs(np.float32(q))
    for i in range(len(x)):
        w0 = np.abs(np.float32(cutoff[i])) * angular_scale
        cos_w0 = np.cos(w0, dtype=np.float32)
        sin_w0 = np.sin(w0, dtype=np.float32)
        alpha = sin_w0 * half / q
        a0 = one + alpha
        b0 = ((one - cos_w0) * half) / a0
        b1 = (one - cos_w0) / a0
        b2 = b0
        a1 = (-two * cos_w0) / a0
        a2 = (one - alpha) / a0
        out = np.float32(
            b0 * np.float32(x[i]) + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2)
        y[i] = out
        x2, x1 = x1, np.float32(x[i])
        y2, y1 = y1, out
    return y


@functools.lru_cache(maxsize=16)
def dgen_time_ramp(frames, sample_rate):
    """Match Signal.accum(1 / sampleRate)'s pre-update float32 state."""
    values = np.zeros(frames, dtype=np.float32)
    increment = np.float32(np.float32(1.0) / np.float32(sample_rate))
    state = np.float32(0.0)
    for i in range(frames):
        values[i] = state
        state = np.float32(state + increment)
    return values


@functools.lru_cache(maxsize=16)
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


@functools.lru_cache(maxsize=256)
def filtered_dgen_noise(frames, cutoff, sample_rate):
    noise = dgen_noise(frames) * 2.0 - 1.0
    return lowpass_biquad(noise, cutoff, 0.707, 1.0, sample_rate)


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
    parser.add_argument(
        "--profile", choices=["808", "909", "hoodie-bass", "subtractive-bass"],
        default=None)
    parser.add_argument("--oscillator-only", action="store_true")
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
            profile=args.profile,
            oscillator_only=args.oscillator_only,
        ),
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
