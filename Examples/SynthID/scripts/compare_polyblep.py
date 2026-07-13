#!/usr/bin/env python3
"""Independent MR-STFT gate for E2 training/deployment oscillator equivalence."""

import argparse
import json

import compare


DEFAULT_THRESHOLD = 0.00308  # 5% of the frozen E3 additive distance, 0.0616.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    training, training_rate = compare.read_wav(args.training)
    deployment, deployment_rate = compare.read_wav(args.deployment)
    if training_rate != deployment_rate:
        raise ValueError(
            f"sample-rate mismatch: training={training_rate}, deployment={deployment_rate}")

    distance = compare.mrstft(training, deployment)
    report = {
        "sampleRate": training_rate,
        "frames": min(len(training), len(deployment)),
        "windows": list(compare.WINDOWS),
        "logEpsilon": compare.LOG_EPSILON,
        "threshold": args.threshold,
        "distance": distance,
        "pass": distance < args.threshold,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
