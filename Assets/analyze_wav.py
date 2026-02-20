#!/usr/bin/env python3
"""Analyze a WAV file and produce a compact summary for agentic debugging.

Usage:
    python3 analyze_wav.py <path.wav> [--json] [--segments N]

Output is compact text by default, designed for LLM consumption.
Use --json for machine-readable output.
Use --segments N to split the file into N time segments for temporal analysis (default: 4).
"""
import sys
import argparse
import json
import numpy as np
from collections import OrderedDict

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_wav(path):
    """Load WAV file, return (samples_float32, sample_rate).

    Tries soundfile first (more tolerant of header quirks), falls back to scipy.
    """
    if HAS_SOUNDFILE:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    if HAS_SCIPY:
        sr, audio = wavfile.read(path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 2.0:
            audio = audio / 32768.0
        return audio, sr

    raise ImportError("Need either soundfile or scipy installed: pip install soundfile")


def db(x):
    """Convert linear amplitude to dB, floor at -120."""
    return max(20 * np.log10(x + 1e-12), -120.0)


def analyze_basics(audio, sr):
    """Basic sanity checks and level analysis."""
    n = len(audio)
    duration = n / sr

    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))
    all_zero = bool(np.all(audio == 0))

    if has_nan or has_inf:
        clean = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        clean = audio

    peak = float(np.max(np.abs(clean)))
    rms = float(np.sqrt(np.mean(clean ** 2)))
    dc_offset = float(np.mean(clean))
    crest = peak / (rms + 1e-12)

    clip_pos = int(np.sum(clean >= 0.999))
    clip_neg = int(np.sum(clean <= -0.999))
    clipping = clip_pos + clip_neg

    return OrderedDict(
        samples=n,
        sample_rate=sr,
        duration_s=round(duration, 4),
        has_nan=has_nan,
        has_inf=has_inf,
        all_zero=all_zero,
        peak=round(peak, 6),
        peak_db=round(db(peak), 1),
        rms=round(rms, 6),
        rms_db=round(db(rms), 1),
        dc_offset=round(dc_offset, 6),
        crest_factor=round(crest, 1),
        clipping_samples=clipping,
    )


def analyze_spectrum(audio, sr, max_peaks=8):
    """FFT analysis: find dominant frequency peaks."""
    n = len(audio)
    if n == 0:
        return OrderedDict(peaks=[], noise_floor_db=-120)

    # Use full signal, windowed
    window = np.hanning(n)
    windowed = audio * window

    fft = np.fft.rfft(windowed)
    mag = np.abs(fft) * 2.0 / n  # Normalize
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Skip DC bin
    mag[0] = 0

    # Find peaks: local maxima above noise floor
    noise_floor = np.median(mag[1:]) * 2 + 1e-12
    noise_floor_db = round(db(noise_floor), 1)

    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > noise_floor:
            peaks.append((float(freqs[i]), float(mag[i])))

    # Sort by magnitude descending
    peaks.sort(key=lambda p: -p[1])
    peaks = peaks[:max_peaks]

    # Format peaks
    peak_list = []
    for freq, m in peaks:
        peak_list.append(OrderedDict(
            freq_hz=round(freq, 1),
            mag_db=round(db(m), 1),
        ))

    # Spectral centroid
    total_mag = np.sum(mag[1:])
    if total_mag > 1e-12:
        centroid = float(np.sum(freqs[1:] * mag[1:]) / total_mag)
    else:
        centroid = 0.0

    return OrderedDict(
        peaks=peak_list,
        spectral_centroid_hz=round(centroid, 1),
        noise_floor_db=noise_floor_db,
    )


def analyze_harmonics(peaks, tolerance_cents=50):
    """Detect harmonic structure from frequency peaks."""
    if not peaks:
        return OrderedDict(fundamental_hz=None, harmonics=[], is_tonal=False)

    f0 = peaks[0]["freq_hz"]
    if f0 < 1.0:
        return OrderedDict(fundamental_hz=None, harmonics=[], is_tonal=False)

    harmonics = []
    for p in peaks[1:]:
        ratio = p["freq_hz"] / f0
        nearest_int = round(ratio)
        if nearest_int >= 2:
            cents_off = abs(1200 * np.log2(ratio / nearest_int)) if nearest_int > 0 else 9999
            if cents_off < tolerance_cents:
                harmonics.append(OrderedDict(
                    harmonic=nearest_int,
                    freq_hz=p["freq_hz"],
                    mag_db=p["mag_db"],
                    cents_off=round(cents_off, 1),
                ))

    is_tonal = len(harmonics) >= 1 or (len(peaks) <= 2)

    return OrderedDict(
        fundamental_hz=f0,
        harmonics=harmonics,
        is_tonal=is_tonal,
    )


def analyze_temporal(audio, sr, n_segments=4):
    """Analyze how the signal changes over time."""
    n = len(audio)
    if n == 0:
        return OrderedDict(segments=[], zero_crossing_rate=0, is_stationary=True)

    seg_len = n // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else n
        seg = audio[start:end]

        rms = float(np.sqrt(np.mean(seg ** 2)))
        peak = float(np.max(np.abs(seg)))
        segments.append(OrderedDict(
            time_s=round(start / sr, 4),
            rms_db=round(db(rms), 1),
            peak_db=round(db(peak), 1),
        ))

    # Zero crossing rate
    zc = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = float(zc / n)

    # Stationarity: check if RMS varies more than 6dB across segments
    rms_vals = [s["rms_db"] for s in segments]
    rms_range = max(rms_vals) - min(rms_vals)
    is_stationary = rms_range < 6.0

    return OrderedDict(
        segments=segments,
        zero_crossing_rate=round(zcr, 4),
        is_stationary=is_stationary,
        rms_range_db=round(rms_range, 1),
    )


def classify_signal(basics, spectrum, harmonics, temporal):
    """One-line signal classification."""
    tags = []

    if basics["all_zero"]:
        return "SILENT (all zeros)"
    if basics["has_nan"]:
        tags.append("CORRUPTED(NaN)")
    if basics["has_inf"]:
        tags.append("CORRUPTED(Inf)")
    if basics["clipping_samples"] > 0:
        tags.append(f"CLIPPING({basics['clipping_samples']} samples)")

    if basics["rms_db"] < -60:
        tags.append("near-silent")
    elif basics["rms_db"] > -1:
        tags.append("very-loud")

    if abs(basics["dc_offset"]) > 0.01:
        tags.append(f"DC-offset({basics['dc_offset']:.4f})")

    if harmonics["is_tonal"]:
        n_harm = len(harmonics["harmonics"])
        if n_harm == 0:
            tags.append(f"pure-tone({harmonics['fundamental_hz']:.1f}Hz)")
        else:
            tags.append(f"tonal({harmonics['fundamental_hz']:.1f}Hz, {n_harm} harmonics)")
    else:
        if temporal["zero_crossing_rate"] > 0.3:
            tags.append("noise-like")
        else:
            tags.append("complex/inharmonic")

    if not temporal["is_stationary"]:
        tags.append("time-varying")
    else:
        tags.append("steady")

    return " | ".join(tags) if tags else "normal"


def format_compact(basics, spectrum, harmonics, temporal, classification):
    """Format as compact human/LLM-readable text."""
    lines = []

    # Line 1: File info
    lines.append(
        f"Duration: {basics['duration_s']:.4f}s | SR: {basics['sample_rate']} | "
        f"Samples: {basics['samples']}"
    )

    # Line 2: Levels
    flags = []
    if basics["has_nan"]:
        flags.append("HAS NaN!")
    if basics["has_inf"]:
        flags.append("HAS Inf!")
    if basics["all_zero"]:
        flags.append("ALL ZEROS!")
    if basics["clipping_samples"] > 0:
        flags.append(f"CLIPPING({basics['clipping_samples']})")
    flag_str = " | ".join(flags) if flags else "OK"

    lines.append(
        f"Level: RMS={basics['rms_db']:.1f}dB Peak={basics['peak_db']:.1f}dB "
        f"DC={basics['dc_offset']:.4f} Crest={basics['crest_factor']:.1f} | {flag_str}"
    )

    # Line 3: Spectral peaks
    if spectrum["peaks"]:
        peak_strs = [f"{p['freq_hz']:.1f}Hz({p['mag_db']:.1f}dB)" for p in spectrum["peaks"]]
        lines.append(f"Peaks: {', '.join(peak_strs)}")
        lines.append(f"Centroid: {spectrum['spectral_centroid_hz']:.1f}Hz | Noise floor: {spectrum['noise_floor_db']:.1f}dB")
    else:
        lines.append("Peaks: none detected")

    # Line 4: Harmonics
    if harmonics["fundamental_hz"]:
        if harmonics["harmonics"]:
            harm_strs = [f"H{h['harmonic']}={h['freq_hz']:.1f}Hz({h['mag_db']:.1f}dB)" for h in harmonics["harmonics"]]
            lines.append(f"Harmonics: F0={harmonics['fundamental_hz']:.1f}Hz | {', '.join(harm_strs)}")
        else:
            lines.append(f"Harmonics: F0={harmonics['fundamental_hz']:.1f}Hz (pure tone, no overtones)")

    # Line 5: Temporal
    seg_strs = [f"{s['time_s']:.2f}s:{s['rms_db']:.1f}dB" for s in temporal["segments"]]
    rms_range = temporal["rms_range_db"]
    stationarity = "Stationary" if temporal["is_stationary"] else f"Non-stationary(range={rms_range:.1f}dB)"
    lines.append(
        f"Temporal: ZCR={temporal['zero_crossing_rate']:.4f} | "
        f"{stationarity} | [{', '.join(seg_strs)}]"
    )

    # Line 6: Classification
    lines.append(f"Signal: {classification}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a WAV file for agentic debugging")
    parser.add_argument("path", help="Path to WAV file")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of compact text")
    parser.add_argument("--segments", type=int, default=4, help="Number of temporal segments (default: 4)")
    args = parser.parse_args()

    audio, sr = load_wav(args.path)

    basics = analyze_basics(audio, sr)
    spectrum = analyze_spectrum(audio, sr)
    harmonics = analyze_harmonics(spectrum["peaks"])
    temporal = analyze_temporal(audio, sr, n_segments=args.segments)
    classification = classify_signal(basics, spectrum, harmonics, temporal)

    if args.json:
        result = OrderedDict(
            basics=basics,
            spectrum=spectrum,
            harmonics=harmonics,
            temporal=temporal,
            classification=classification,
        )

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return obj.item()
                return super().default(obj)

        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        print(format_compact(basics, spectrum, harmonics, temporal, classification))


if __name__ == "__main__":
    main()
