#!/usr/bin/env python3
"""
Plot DDSPE2E decoder control dumps produced by --dump-controls-every.

Usage examples:
  python3 Examples/DDSPE2E/scripts/plot_controls.py --run runs/controls_dump_smoke
  python3 Examples/DDSPE2E/scripts/plot_controls.py --controls-dir runs/<run>/logs/controls --step 30
  python3 Examples/DDSPE2E/scripts/plot_controls.py --run runs/<run> --steps 0,10,20,50
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


SUMMARY_RE = re.compile(r"step_(\d{6})_control_summary\.csv$")
FRAME_FILE_RE = re.compile(r"step_(\d{6})_b(\d+)_f(\d+)_(harmonics|wavetable|noise_filter)\.csv$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot DDSPE2E control dumps")
    p.add_argument("--run", type=str, default="", help="Run directory (contains logs/controls)")
    p.add_argument("--controls-dir", type=str, default="", help="Direct path to logs/controls")
    p.add_argument(
        "--step",
        type=str,
        default="latest",
        help="Step number (e.g. 30) or 'latest' (default)",
    )
    p.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated step list for multi-step compare (e.g. 0,10,20,50) or latestN (e.g. latest4)",
    )
    p.add_argument("--batch", type=int, default=0, help="Batch index from control_summary.csv")
    p.add_argument(
        "--compare-frame",
        type=int,
        default=-1,
        help="Frame index for cross-step harmonic/wavetable compare (-1: use middle frame)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: <controls-dir>/plots/step_xxxxxx_controls.png)",
    )
    p.add_argument("--dpi", type=int, default=160, help="Output DPI")
    return p.parse_args()


def resolve_controls_dir(args: argparse.Namespace) -> str:
    if args.controls_dir:
        return args.controls_dir
    if args.run:
        return os.path.join(args.run, "logs", "controls")
    raise SystemExit("Provide either --run or --controls-dir")


def available_steps(controls_dir: str) -> List[int]:
    steps = set()
    for path in glob.glob(os.path.join(controls_dir, "step_*_control_summary.csv")):
        m = SUMMARY_RE.search(os.path.basename(path))
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def resolve_step(step_arg: str, steps: List[int]) -> int:
    if not steps:
        raise SystemExit("No control summary CSV files found")
    if step_arg == "latest":
        return steps[-1]
    try:
        step = int(step_arg)
    except ValueError as exc:
        raise SystemExit(f"Invalid --step value: {step_arg}") from exc
    if step not in steps:
        raise SystemExit(f"Step {step} not found. Available: {steps[:8]}{'...' if len(steps) > 8 else ''}")
    return step


def resolve_steps(step_args: str, steps: List[int]) -> List[int]:
    if not step_args:
        return []
    if not steps:
        raise SystemExit("No control summary CSV files found")

    m = re.fullmatch(r"latest(\d+)", step_args.strip())
    if m:
        n = max(1, int(m.group(1)))
        return steps[-n:]

    out: List[int] = []
    for token in step_args.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            step = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid step token in --steps: {token}") from exc
        if step not in steps:
            raise SystemExit(f"Step {step} not found. Available: {steps[:8]}{'...' if len(steps) > 8 else ''}")
        out.append(step)

    if not out:
        raise SystemExit("--steps produced an empty list")
    return sorted(set(out))


def read_summary(summary_csv: str, batch: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "frame": [],
        "f0_norm": [],
        "loudness_norm": [],
        "uv": [],
        "harmonic_gain": [],
        "noise_gain": [],
        "amp_sum": [],
        "amp_max": [],
        "amp_argmax": [],
    }

    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["batch"]) != batch:
                continue
            out["frame"].append(float(row["frame"]))
            out["f0_norm"].append(float(row["f0_norm"]))
            out["loudness_norm"].append(float(row["loudness_norm"]))
            out["uv"].append(float(row["uv"]))
            out["harmonic_gain"].append(float(row["harmonic_gain"]))
            out["noise_gain"].append(float(row["noise_gain"]))
            out["amp_sum"].append(float(row["amp_sum"]))
            out["amp_max"].append(float(row["amp_max"]))
            out["amp_argmax"].append(float(row["amp_argmax"]))

    if not out["frame"]:
        raise SystemExit(f"No rows found for batch={batch} in {summary_csv}")

    return out


def read_xy_csv(path: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys


def load_frame_files(
    controls_dir: str,
    step: int,
    batch: int,
) -> Tuple[Dict[int, Tuple[List[float], List[float]]], Dict[int, Tuple[List[float], List[float]]]]:
    harm: Dict[int, Tuple[List[float], List[float]]] = {}
    wave: Dict[int, Tuple[List[float], List[float]]] = {}
    tag = f"{step:06d}"

    for path in glob.glob(os.path.join(controls_dir, f"step_{tag}_b{batch}_f*_*.csv")):
        name = os.path.basename(path)
        m = FRAME_FILE_RE.match(name)
        if not m:
            continue
        frame = int(m.group(3))
        kind = m.group(4)
        data = read_xy_csv(path)
        if kind == "harmonics":
            harm[frame] = data
        elif kind == "wavetable":
            wave[frame] = data

    return harm, wave


def pick_default_output(controls_dir: str, step: int) -> str:
    out_dir = os.path.join(controls_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"step_{step:06d}_controls.png")


def pick_default_compare_output(controls_dir: str, steps: List[int]) -> str:
    out_dir = os.path.join(controls_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(
        out_dir,
        f"steps_{steps[0]:06d}_{steps[-1]:06d}_controls_compare.png",
    )


def load_frame_kind_for_step(
    controls_dir: str,
    step: int,
    batch: int,
    kind: str,
    target_frame: int,
) -> Tuple[Optional[int], Tuple[List[float], List[float]]]:
    tag = f"{step:06d}"
    candidates: Dict[int, str] = {}
    for path in glob.glob(os.path.join(controls_dir, f"step_{tag}_b{batch}_f*_{kind}.csv")):
        name = os.path.basename(path)
        m = FRAME_FILE_RE.match(name)
        if not m:
            continue
        frame = int(m.group(3))
        candidates[frame] = path

    if not candidates:
        return None, ([], [])

    frames = sorted(candidates.keys())
    if target_frame < 0:
        chosen = frames[len(frames) // 2]
    else:
        chosen = min(frames, key=lambda f: abs(f - target_frame))
    return chosen, read_xy_csv(candidates[chosen])


def plot_single_step(
    controls_dir: str,
    step: int,
    batch: int,
    out_path: str,
    dpi: int,
) -> None:
    summary_path = os.path.join(controls_dir, f"step_{step:06d}_control_summary.csv")
    summary = read_summary(summary_path, batch=batch)
    harmonics, wavetables = load_frame_files(controls_dir, step=step, batch=batch)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    frames = summary["frame"]

    # Panel 1: main controls over time
    ax0.plot(frames, summary["harmonic_gain"], label="harmonic_gain", linewidth=1.8)
    ax0.plot(frames, summary["noise_gain"], label="noise_gain", linewidth=1.6)
    ax0.plot(frames, summary["uv"], label="uv", linewidth=1.4, alpha=0.9)
    ax0.set_title("Control Gains / UV vs Frame")
    ax0.set_xlabel("Frame")
    ax0.set_ylabel("Value")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    # Panel 2: harmonic stats over time
    ax1.plot(frames, summary["amp_sum"], label="amp_sum", linewidth=1.8)
    ax1.plot(frames, summary["amp_max"], label="amp_max", linewidth=1.4)
    ax1.plot(frames, summary["amp_argmax"], label="amp_argmax", linewidth=1.2, alpha=0.85)
    ax1.set_title("Harmonic Stats vs Frame")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Value / Index")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    # Panel 3: harmonic distributions for selected frames
    if harmonics:
        for frame in sorted(harmonics.keys()):
            xs, ys = harmonics[frame]
            ax2.plot(xs, ys, marker="o", markersize=2.3, linewidth=1.0, label=f"frame {frame}")
        ax2.set_title("Harmonic Amplitudes (selected frames)")
        ax2.set_xlabel("Harmonic index")
        ax2.set_ylabel("Amplitude")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No *_harmonics.csv files found", ha="center", va="center")
        ax2.set_axis_off()

    # Panel 4: synthetic wavetable snapshots
    if wavetables:
        for frame in sorted(wavetables.keys()):
            xs, ys = wavetables[frame]
            ax3.plot(xs, ys, linewidth=1.2, label=f"frame {frame}")
        ax3.set_title("Synthetic Wavetable (selected frames)")
        ax3.set_xlabel("Sample (one cycle)")
        ax3.set_ylabel("Value")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="best")
    else:
        ax3.text(0.5, 0.5, "No *_wavetable.csv files found", ha="center", va="center")
        ax3.set_axis_off()

    fig.suptitle(f"DDSPE2E Controls - step {step:06d} - batch {batch}", fontsize=13)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def plot_multi_step(
    controls_dir: str,
    steps: List[int],
    batch: int,
    compare_frame: int,
    out_path: str,
    dpi: int,
) -> None:
    summaries: Dict[int, Dict[str, List[float]]] = {}
    for step in steps:
        summary_path = os.path.join(controls_dir, f"step_{step:06d}_control_summary.csv")
        summaries[step] = read_summary(summary_path, batch=batch)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel 1: harmonic gain over frame for each step
    for step in steps:
        s = summaries[step]
        ax0.plot(s["frame"], s["harmonic_gain"], linewidth=1.5, label=f"step {step}")
    ax0.set_title("Harmonic Gain vs Frame (across steps)")
    ax0.set_xlabel("Frame")
    ax0.set_ylabel("harmonic_gain")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=8)

    # Panel 2: amp_sum over frame for each step
    for step in steps:
        s = summaries[step]
        ax1.plot(s["frame"], s["amp_sum"], linewidth=1.5, label=f"step {step}")
    ax1.set_title("Amp Sum vs Frame (across steps)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("amp_sum")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    # Panel 3: harmonic amplitudes at a selected frame for each step
    harm_labels: List[str] = []
    for step in steps:
        chosen, (xs, ys) = load_frame_kind_for_step(
            controls_dir=controls_dir,
            step=step,
            batch=batch,
            kind="harmonics",
            target_frame=compare_frame,
        )
        if xs:
            label = f"step {step} (f{chosen})"
            harm_labels.append(label)
            ax2.plot(xs, ys, linewidth=1.3, marker="o", markersize=2.2, label=label)
    if harm_labels:
        ax2.set_title("Harmonic Amplitudes at Compare Frame (across steps)")
        ax2.set_xlabel("Harmonic index")
        ax2.set_ylabel("Amplitude")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="best", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No harmonics CSV found for requested steps", ha="center", va="center")
        ax2.set_axis_off()

    # Panel 4: wavetable at selected frame for each step
    wave_labels: List[str] = []
    for step in steps:
        chosen, (xs, ys) = load_frame_kind_for_step(
            controls_dir=controls_dir,
            step=step,
            batch=batch,
            kind="wavetable",
            target_frame=compare_frame,
        )
        if xs:
            label = f"step {step} (f{chosen})"
            wave_labels.append(label)
            ax3.plot(xs, ys, linewidth=1.2, label=label)
    if wave_labels:
        ax3.set_title("Synthetic Wavetable at Compare Frame (across steps)")
        ax3.set_xlabel("Sample (one cycle)")
        ax3.set_ylabel("Value")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="best", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No wavetable CSV found for requested steps", ha="center", va="center")
        ax3.set_axis_off()

    title_steps = ",".join(str(s) for s in steps)
    fig.suptitle(
        f"DDSPE2E Controls Compare - steps [{title_steps}] - batch {batch}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    controls_dir = resolve_controls_dir(args)
    steps = available_steps(controls_dir)
    compare_steps = resolve_steps(args.steps, steps)
    if compare_steps:
        out_path = args.output if args.output else pick_default_compare_output(controls_dir, compare_steps)
        plot_multi_step(
            controls_dir=controls_dir,
            steps=compare_steps,
            batch=args.batch,
            compare_frame=args.compare_frame,
            out_path=out_path,
            dpi=args.dpi,
        )
        print(f"Wrote plot: {out_path}")
    else:
        step = resolve_step(args.step, steps)
        out_path = args.output if args.output else pick_default_output(controls_dir, step)
        plot_single_step(
            controls_dir=controls_dir,
            step=step,
            batch=args.batch,
            out_path=out_path,
            dpi=args.dpi,
        )
        print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
