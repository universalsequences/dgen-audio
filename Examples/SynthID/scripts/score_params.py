#!/usr/bin/env python3
"""Score a rung-3 patch params.json against a target wav using the same
independent CPU MR-STFT metric compare.py and refine_rung3.py use.

Reused by Trainer.swift/main.swift (via subprocess) to rescore each
restart's FINAL learned params by the independent metric instead of GPU
training loss, without running any coordinate search. Prints a single-line
JSON object to stdout: {"distance": ..., "frames": ..., "sampleRate": ...}.
"""

import argparse
import json

import compare
import render_reference as reference


def score(target, params, sample_rate, highpass_hz, enable_noise_filter=True, profile="808"):
    frames = len(target)
    rendered = reference.render(
        params, frames, sample_rate, enable_noise_filter=enable_noise_filter, profile=profile)
    peak = float(max(abs(rendered.min()), abs(rendered.max())))
    if peak > 0.9:
        rendered = rendered * (0.9 / peak)
    filtered_target = compare.capture_highpass(target, sample_rate, highpass_hz)
    filtered_rendered = compare.capture_highpass(rendered, sample_rate, highpass_hz)
    return compare.mrstft(filtered_rendered, filtered_target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--frames", type=int, default=None,
                         help="frames to score at; defaults to the target wav's length")
    parser.add_argument("--sample-rate", type=int, default=None,
                         help="defaults to the target wav's sample rate")
    parser.add_argument("--highpass-hz", type=float, default=compare.DEFAULT_HIGHPASS_HZ)
    parser.add_argument(
        "--profile", choices=["808", "909", "hoodie-bass", "subtractive-bass", "monologue-bass"],
        default="808",
                         help="accepted for CLI parity with refine_rung3.py; the raw"
                              " synth render does not depend on the profile's bounds")
    parser.add_argument("--no-noise-filter", action="store_true")
    args = parser.parse_args()

    target, target_rate = compare.read_wav(args.target)
    sample_rate = args.sample_rate or int(round(target_rate))
    frames = args.frames or len(target)
    target = target[:frames]

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    if "params" in params and isinstance(params["params"], dict):
        params = params["params"]
    params.setdefault("bodyAsymmetry", 0.0)
    params.setdefault("bodyHarmonic", 0.0)
    params.setdefault("ampCurve", 0.0)

    distance = score(
        target, params, sample_rate, args.highpass_hz,
        enable_noise_filter=not args.no_noise_filter, profile=args.profile)
    print(json.dumps({
        "distance": distance,
        "frames": frames,
        "sampleRate": sample_rate,
    }))


if __name__ == "__main__":
    main()
