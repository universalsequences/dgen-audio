#!/usr/bin/env python3
"""Deterministically prepare a directory of snare one-shot WAV files."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import wave

SAMPLE_RATE = 44_100
DURATION_SECONDS = 0.75
OUTPUT_FRAMES = int(SAMPLE_RATE * DURATION_SECONDS)
ONSET_DB = -40.0
PRE_ROLL = 32
HIGHPASS_HZ = 30.0
PEAK_TARGET = 0.99


def read_wav(path: Path) -> tuple[list[float], dict]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        compression = wav.getcomptype()
        raw = wav.readframes(frames)
    if compression != "NONE" or width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported WAV encoding (compression={compression}, width={width})")
    if channels < 1 or rate < 1:
        raise ValueError("invalid channel count or sample rate")

    scale = float(1 << (8 * width - 1))
    samples: list[float] = []
    clipped_frames = 0
    clipped_run = max_clipped_run = 0
    offset = 0
    for _ in range(frames):
        values = []
        frame_clipped = False
        for _ in range(channels):
            chunk = raw[offset:offset + width]
            offset += width
            if width == 1:
                integer = chunk[0] - 128
                full_scale = integer in (-128, 127)
            elif width == 3:
                integer = int.from_bytes(chunk, "little", signed=True)
                full_scale = integer in (-(1 << 23), (1 << 23) - 1)
            else:
                integer = int.from_bytes(chunk, "little", signed=True)
                limit = 1 << (8 * width - 1)
                full_scale = integer in (-limit, limit - 1)
            frame_clipped = frame_clipped or full_scale
            values.append(integer / scale)
        samples.append(sum(values) / channels)
        if frame_clipped:
            clipped_frames += 1
            clipped_run += 1
            max_clipped_run = max(max_clipped_run, clipped_run)
        else:
            clipped_run = 0

    return samples, {
        "source_sample_rate": rate,
        "source_channels": channels,
        "source_frames": frames,
        "source_sample_width_bits": width * 8,
        "clipped_frame_count": clipped_frames,
        "max_clipped_run": max_clipped_run,
    }


def resample(samples: list[float], source_rate: int) -> list[float]:
    """Band-limited windowed-sinc resampling without external dependencies."""
    if source_rate == SAMPLE_RATE or not samples:
        return samples.copy()
    count = max(1, (len(samples) * SAMPLE_RATE + source_rate // 2) // source_rate)
    radius = 16
    cutoff = min(1.0, SAMPLE_RATE / source_rate)
    result = []
    for output_index in range(count):
        position = output_index * source_rate / SAMPLE_RATE
        center = math.floor(position)
        weighted_sum = weight_sum = 0.0
        for source_index in range(center - radius + 1, center + radius + 1):
            if not 0 <= source_index < len(samples):
                continue
            distance = position - source_index
            window_position = distance / radius
            if abs(window_position) >= 1.0:
                continue
            sinc_argument = cutoff * distance
            sinc = (1.0 if sinc_argument == 0.0 else
                    math.sin(math.pi * sinc_argument) / (math.pi * sinc_argument))
            # Blackman window suppresses downsampling aliases and ringing.
            window = (0.42 + 0.5 * math.cos(math.pi * window_position) +
                      0.08 * math.cos(2.0 * math.pi * window_position))
            weight = cutoff * sinc * window
            weighted_sum += samples[source_index] * weight
            weight_sum += weight
        result.append(weighted_sum / weight_sum if weight_sum else 0.0)
    return result


def first_onset(samples: list[float], peak: float) -> int:
    if peak <= 0.0:
        return 0
    threshold = peak * 10.0 ** (ONSET_DB / 20.0)
    return next((index for index, value in enumerate(samples) if abs(value) >= threshold), 0)


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def has_rms_rerise(samples: list[float], onset: int, peak: float) -> bool:
    """Detect a sustained >=6 dB RMS rise beginning after 300 ms."""
    window = int(0.020 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    start = onset + int(0.300 * SAMPLE_RATE)
    levels = [rms(samples[pos:pos + window])
              for pos in range(start, len(samples) - window + 1, hop)]
    if len(levels) < 4 or peak <= 0.0:
        return False
    meaningful = peak * 10.0 ** (-45.0 / 20.0)
    prior_min = levels[0]
    for index in range(1, len(levels) - 2):
        level = levels[index]
        baseline = max(prior_min, 1e-12)
        if (level >= 2.0 * baseline and level >= meaningful and
                min(levels[index:index + 3]) >= 1.8 * baseline):
            return True
        prior_min = min(prior_min, level)
    return False


def highpass(samples: list[float]) -> list[float]:
    """Apply a causal second-order Butterworth high-pass biquad."""
    omega = 2.0 * math.pi * HIGHPASS_HZ / SAMPLE_RATE
    cosine, sine = math.cos(omega), math.sin(omega)
    alpha = sine / math.sqrt(2.0)
    a0 = 1.0 + alpha
    b0 = ((1.0 + cosine) / 2.0) / a0
    b1 = (-(1.0 + cosine)) / a0
    b2 = b0
    a1 = (-2.0 * cosine) / a0
    a2 = (1.0 - alpha) / a0
    x1 = x2 = y1 = y2 = 0.0
    output = []
    for x0 in samples:
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output.append(y0)
        x2, x1, y2, y1 = x1, x0, y1, y0
    return output


def write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    integers = [max(-32768, min(32767, round(value * 32767.0))) for value in samples]
    payload = struct.pack(f"<{len(integers)}h", *integers)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(payload)


def prepare_file(source: Path, relative: Path, destination: Path) -> dict:
    entry = {"source": relative.as_posix(), "output": None}
    try:
        mono, metadata = read_wav(source)
    except (EOFError, ValueError, wave.Error) as error:
        entry.update({"decision": "rejected", "reason": "invalid_wav", "reasons": ["invalid_wav"],
                      "error": str(error)})
        return entry

    entry.update(metadata)
    original_peak = max((abs(value) for value in mono), default=0.0)
    source_rms = rms(mono)
    dc_offset = sum(mono) / len(mono) if mono else 0.0
    resampled = resample(mono, metadata["source_sample_rate"])
    resampled_peak = max((abs(value) for value in resampled), default=0.0)
    onset = first_onset(resampled, resampled_peak)
    trim_offset = max(0, onset - PRE_ROLL)

    reasons = []
    clipped_ratio = metadata["clipped_frame_count"] / max(1, metadata["source_frames"])
    if metadata["max_clipped_run"] >= 3 or clipped_ratio >= 0.001:
        reasons.append("clipped")
    if abs(dc_offset) > max(0.01, 0.1 * source_rms):
        reasons.append("dc_offset")
    if has_rms_rerise(resampled, onset, resampled_peak):
        reasons.append("rms_rerise_after_300ms")
    if original_peak == 0.0:
        reasons.append("silent")

    entry.update({
        "original_peak": original_peak,
        "dc_offset": dc_offset,
        "onset_sample": onset,
        "onset_offset_samples": trim_offset,
        "pre_roll_samples": onset - trim_offset,
    })
    if reasons:
        entry.update({"decision": "rejected", "reason": reasons[0], "reasons": reasons})
        return entry

    prepared = resampled[trim_offset:trim_offset + OUTPUT_FRAMES]
    prepared.extend([0.0] * (OUTPUT_FRAMES - len(prepared)))
    prepared = highpass(prepared)
    prepared_peak = max(abs(value) for value in prepared)
    normalization_gain = PEAK_TARGET / prepared_peak
    prepared = [value * normalization_gain for value in prepared]
    output_relative = relative.with_suffix(".wav")
    write_wav(destination / output_relative, prepared)
    entry.update({
        "decision": "accepted",
        "reason": None,
        "reasons": [],
        "output": output_relative.as_posix(),
        "normalization_gain": normalization_gain,
        "output_peak": PEAK_TARGET,
        "output_frames": OUTPUT_FRAMES,
    })
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing source WAVs (searched recursively)")
    parser.add_argument("output", type=Path, help="directory for prepared WAVs and manifest.json")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        parser.error(f"input is not a directory: {source}")
    if output == source:
        parser.error("input and output directories must differ")

    excluded = [output]
    files = []
    for path in source.rglob("*"):
        resolved = path.resolve()
        if path.is_file() and path.suffix.lower() == ".wav" and not any(
                resolved == item or item in resolved.parents for item in excluded):
            files.append(path)
    files.sort(key=lambda path: path.relative_to(source).as_posix())

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        entries = [prepare_file(path, path.relative_to(source), staging) for path in files]
        manifest = {
            "schema_version": 1,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "resampler": "32-tap-blackman-windowed-sinc",
                "duration_seconds": DURATION_SECONDS,
                "output_frames": OUTPUT_FRAMES,
                "onset_threshold_dbfs_relative_to_peak": ONSET_DB,
                "pre_roll_samples": PRE_ROLL,
                "peak_target": PEAK_TARGET,
                "highpass_hz": HIGHPASS_HZ,
                "highpass_type": "causal_butterworth_biquad",
                "dc_reject_threshold": "abs(mean) > max(0.01, 0.1 * RMS)",
                "clipping_reject_threshold": ">=3 consecutive full-scale frames or >=0.1% full-scale frames",
                "rms_rerise_threshold": ">=6 dB sustained for 3 frames after 300 ms",
            },
            "summary": {
                "total": len(entries),
                "accepted": sum(entry["decision"] == "accepted" for entry in entries),
                "rejected": sum(entry["decision"] == "rejected" for entry in entries),
            },
            "files": entries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared {manifest['summary']['accepted']} of {len(entries)} WAVs in {output}; "
          f"rejected {manifest['summary']['rejected']} (see manifest.json)")


if __name__ == "__main__":
    main()
