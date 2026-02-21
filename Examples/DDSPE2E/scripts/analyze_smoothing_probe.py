#!/usr/bin/env python3
"""
Analyze DDSPE2E probe-smoothing output.

Usage:
  python3 Examples/DDSPE2E/scripts/analyze_smoothing_probe.py --dir /tmp/ddsp_smoothing_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze raw vs FIR-smoothed control tensors")
    p.add_argument("--dir", required=True, help="Output dir from `DDSPE2E probe-smoothing`")
    p.add_argument("--output", default=None, help="Output PNG path (default: <dir>/smoothing_analysis.png)")
    return p.parse_args()


def read_series_csv(path: str, raw_key: str, fir_key: str) -> Tuple[List[float], List[float]]:
    raw: List[float] = []
    fir: List[float] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            raw.append(float(row[raw_key]))
            fir.append(float(row[fir_key]))
    return raw, fir


def mean_abs_diff(seq: List[float]) -> float:
    if len(seq) < 2:
        return 0.0
    return sum(abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)) / (len(seq) - 1)


def summarize(raw: List[float], fir: List[float]) -> Dict[str, float]:
    raw_tv = mean_abs_diff(raw)
    fir_tv = mean_abs_diff(fir)
    ratio = (fir_tv / raw_tv) if raw_tv > 0 else 0.0
    return {
        "raw_tv": raw_tv,
        "fir_tv": fir_tv,
        "fir_over_raw_tv": ratio,
    }


def main() -> None:
    args = parse_args()
    d = args.dir

    hg_raw, hg_fir = read_series_csv(
        os.path.join(d, "harmonic_gain_compare.csv"), "raw", "fir"
    )
    ng_raw, ng_fir = read_series_csv(
        os.path.join(d, "noise_gain_compare.csv"), "raw", "fir"
    )
    as_raw, as_fir = read_series_csv(
        os.path.join(d, "amp_sum_compare.csv"), "raw_sum", "fir_sum"
    )

    summary = {
        "harmonic_gain": summarize(hg_raw, hg_fir),
        "noise_gain": summarize(ng_raw, ng_fir),
        "amp_sum": summarize(as_raw, as_fir),
    }

    analysis_json = os.path.join(d, "analysis.json")
    if os.path.exists(analysis_json):
        with open(analysis_json, "r") as f:
            summary["swift_probe"] = json.load(f)

    print(json.dumps(summary, indent=2))

    out = args.output or os.path.join(d, "smoothing_analysis.png")
    x1 = list(range(len(hg_raw)))
    x2 = list(range(len(ng_raw)))
    x3 = list(range(len(as_raw)))

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    ax[0].plot(x1, hg_raw, label="raw")
    ax[0].plot(x1, hg_fir, label="fir")
    ax[0].set_title("Harmonic Gain vs Frame")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(x2, ng_raw, label="raw")
    ax[1].plot(x2, ng_fir, label="fir")
    ax[1].set_title("Noise Gain vs Frame")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    ax[2].plot(x3, as_raw, label="raw")
    ax[2].plot(x3, as_fir, label="fir")
    ax[2].set_title("Harmonic Amp Sum vs Frame")
    ax[2].set_xlabel("Frame")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"wrote plot: {out}")


if __name__ == "__main__":
    main()

