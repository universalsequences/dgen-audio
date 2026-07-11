#!/usr/bin/env python3
"""Bounded scalar refinement for the real-target Rung 3 patch.

The GPU/autograd run remains the primary optimizer. This final deterministic
coordinate search handles the few jagged pitch/click/asymmetry directions for
which the independent metric and the training loss have demonstrably different
local basins. It optimizes only the 14 documented scalars; no waveform samples,
tables, FIR coefficients, or target-derived arrays become patch parameters.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import compare
import render_reference as reference


BOUNDS = {
    "fStart": (80.0, 180.0, "log"),
    "fEnd": (35.0, 60.0, "log"),
    "pitchDecay": (-80.0, -15.0, "logneg"),
    "bodyAmp": (0.5, 1.0, "linear"),
    "ampDecay": (-12.0, -3.0, "linear"),
    "clickFreq": (600.0, 3000.0, "log"),
    "clickAmp": (0.05, 1.5, "linear"),
    "clickDecay": (-1600.0, -200.0, "logneg"),
    "noiseCutoff": (1000.0, 20000.0, "log"),
    "noiseAmp": (1e-6, 0.3, "log"),
    "noiseDecay": (-400.0, -0.001, "logneg"),
    "drive": (1.0, 3.0, "linear"),
    "outGain": (0.4, 1.0, "linear"),
    "bodyAsymmetry": (-0.5, 0.5, "linear"),
}


def transformed(name, value):
    mode = BOUNDS[name][2]
    if mode == "log":
        return math.log(value)
    if mode == "logneg":
        return math.log(-value)
    return value


def natural(name, value):
    mode = BOUNDS[name][2]
    if mode == "log":
        return math.exp(value)
    if mode == "logneg":
        return -math.exp(value)
    return value


class Objective:
    def __init__(self, target, initial, sample_rate, highpass_hz):
        self.sample_rate = sample_rate
        self.frames = len(target)
        self.highpass_hz = highpass_hz
        self.target = compare.capture_highpass(target, sample_rate, highpass_hz)
        self.initial = compare.capture_highpass(initial, sample_rate, highpass_hz)
        self.target_features = {
            window: self.features(self.target, window)
            for window in compare.WINDOWS
        }
        self.initial_distance = self.distance_filtered(self.initial)

    @staticmethod
    def features(signal, window_size):
        hop = window_size // 4
        window = np.hanning(window_size).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::hop]
        scale = max(float(window.sum()) / 2.0, 1e-12)
        return np.log(
            np.abs(np.fft.rfft(frames * window, axis=1)) / scale
            + compare.LOG_EPSILON)

    def distance_filtered(self, signal):
        return sum(
            float(np.mean(np.abs(
                self.features(signal, window) - self.target_features[window])))
            for window in compare.WINDOWS)

    def evaluate(self, params):
        rendered = reference.render(
            params, self.frames, self.sample_rate, enable_noise_filter=True)
        peak = float(np.max(np.abs(rendered)))
        if peak > 0.9:
            rendered = rendered * np.float32(0.9 / peak)
        filtered = compare.capture_highpass(
            rendered, self.sample_rate, self.highpass_hz)
        return self.distance_filtered(filtered)


def coordinate_refine(objective, start, passes=6, steps=15):
    params = dict(start)
    params.setdefault("bodyAsymmetry", 0.0)
    best = objective.evaluate(params)
    order = [
        "bodyAsymmetry", "clickFreq", "clickDecay", "clickAmp",
        "noiseCutoff", "noiseDecay", "noiseAmp",
        "fStart", "pitchDecay", "fEnd", "ampDecay",
        "bodyAmp", "drive", "outGain",
    ]

    for pass_index in range(passes):
        contraction = 0.55 ** pass_index
        for name in order:
            lower, upper, _ = BOUNDS[name]
            z_lower, z_upper = sorted((
                transformed(name, lower), transformed(name, upper)))
            center = transformed(name, params[name])
            span = (z_upper - z_lower) * contraction
            search_lower = max(z_lower, center - span / 2.0)
            search_upper = min(z_upper, center + span / 2.0)
            local_value = params[name]
            local_best = best
            for candidate_z in np.linspace(search_lower, search_upper, steps):
                candidate = dict(params)
                candidate[name] = natural(name, float(candidate_z))
                distance = objective.evaluate(candidate)
                if distance < local_best:
                    local_best = distance
                    local_value = candidate[name]
            if local_best < best:
                params[name] = local_value
                best = local_best
        print(
            f"refine_pass={pass_index} learned_mrstft={best:.6f} "
            f"improvement={1.0 - best / objective.initial_distance:.2%}",
            flush=True)
    return params, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-params", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--highpass-hz", type=float, default=compare.DEFAULT_HIGHPASS_HZ)
    args = parser.parse_args()

    target, target_rate = compare.read_wav(args.target)
    initial, initial_rate = compare.read_wav(args.initial)
    if target_rate != initial_rate:
        raise SystemExit(
            f"sample-rate mismatch: target={target_rate} initial={initial_rate}")
    target, initial = compare.same_length(target, initial)
    with open(args.params, encoding="utf-8") as source:
        base = json.load(source)
    if "params" in base:
        base = base["params"]
    base.setdefault("bodyAsymmetry", 0.0)

    objective = Objective(target, initial, target_rate, args.highpass_hz)
    starts = [dict(base)]
    capture_floor = dict(base)
    capture_floor.update({
        "bodyAsymmetry": 0.1,
        "noiseAmp": 1e-4,
        "noiseDecay": -0.1,
        "noiseCutoff": 10000.0,
    })
    starts.append(capture_floor)

    best_params = None
    best_distance = math.inf
    for index, start in enumerate(starts):
        print(f"refine_start={index}", flush=True)
        params, distance = coordinate_refine(objective, start)
        if distance < best_distance:
            best_params, best_distance = params, distance

    improvement = 1.0 - best_distance / max(objective.initial_distance, 1e-12)
    report = {
        "initialDistance": objective.initial_distance,
        "learnedDistance": best_distance,
        "improvement": improvement,
        "highpassHz": args.highpass_hz,
        "pass": improvement >= compare.REQUIRED_IMPROVEMENT,
    }
    Path(args.out_params).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_params, "w", encoding="utf-8") as output:
        json.dump(best_params, output, indent=2, sort_keys=True)
        output.write("\n")
    with open(args.json, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"refined_initial_mrstft={objective.initial_distance:.6f} "
        f"refined_learned_mrstft={best_distance:.6f} "
        f"improvement={improvement:.2%} pass={str(report['pass']).lower()}")


if __name__ == "__main__":
    main()
