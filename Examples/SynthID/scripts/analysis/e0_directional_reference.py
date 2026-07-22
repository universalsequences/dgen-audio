#!/usr/bin/env python3
"""Float64 reference for the E0 smooth-log spectral directional check."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_f32(path: Path, frames: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != frames:
        raise ValueError(f"{path}: expected {frames} floats, found {values.size}")
    return values.astype(np.float64)


def windowed_rfft(signal: np.ndarray, window_size: int) -> np.ndarray:
    hop = window_size // 4
    padded = np.pad(signal, (window_size - 1, 0))
    frames = np.lib.stride_tricks.sliding_window_view(padded, window_size)[::hop]
    hann = np.hanning(window_size)
    return np.fft.rfft(frames * hann[None, :], axis=1)


def loss_and_derivative(
    base: np.ndarray,
    tangent: np.ndarray,
    target: np.ndarray,
    alpha: float,
    windows: list[int],
    log_epsilon: float,
) -> tuple[float, float]:
    loss = 0.0
    derivative = 0.0
    student = base + alpha * tangent
    for window_size in windows:
        x = windowed_rfft(student, window_size)
        v = windowed_rfft(tangent, window_size)
        y = windowed_rfft(target, window_size)
        x_power = x.real * x.real + x.imag * x.imag
        y_power = y.real * y.real + y.imag * y.imag
        x_log = 0.5 * np.log(x_power + log_epsilon * log_epsilon)
        y_log = 0.5 * np.log(y_power + log_epsilon * log_epsilon)
        diff = x_log - y_log
        loss += float(np.sum(diff * diff)) / base.size
        dlog = np.real(np.conj(x) * v) / (x_power + log_epsilon * log_epsilon)
        derivative += float(np.sum(2.0 * diff * dlog)) / base.size
    return loss, derivative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32768)
    parser.add_argument("--param", default="shape")
    parser.add_argument("--windows", default="256,512,1024,2048")
    parser.add_argument("--log-eps", type=float, default=1e-3)
    parser.add_argument(
        "--fd-grid",
        default="0.000001,0.000002,0.000003,0.000005,0.00001,0.00002,0.00003,0.00005,0.0001,0.00015,0.0002,0.0003,0.0005,0.001",
    )
    args = parser.parse_args()

    windows = [int(value) for value in args.windows.split(",")]
    grid = [float(value) for value in args.fd_grid.split(",")]
    base = read_f32(args.dir / "base_signal.f32", args.frames)
    tangent = read_f32(args.dir / f"{args.param}_tangent.f32", args.frames)
    target = read_f32(args.dir / "target_signal.f32", args.frames)

    base_loss, analytic = loss_and_derivative(
        base, tangent, target, 0.0, windows, args.log_eps
    )
    rows = []
    for epsilon in grid:
        minus, _ = loss_and_derivative(
            base, tangent, target, -epsilon, windows, args.log_eps
        )
        plus, _ = loss_and_derivative(
            base, tangent, target, epsilon, windows, args.log_eps
        )
        fd = (plus - minus) / (2.0 * epsilon)
        relative_error = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-300)
        rows.append(
            {
                "epsilon": epsilon,
                "loss_minus": minus,
                "loss_plus": plus,
                "finite_difference": fd,
                "analytic_derivative": analytic,
                "relative_error": relative_error,
            }
        )

    result = {
        "frames": args.frames,
        "windows": windows,
        "log_epsilon": args.log_eps,
        "base_loss": base_loss,
        "analytic_derivative": analytic,
        "rows": rows,
    }
    (args.dir / "float64_directional_reference.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (args.dir / "float64_directional_reference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
