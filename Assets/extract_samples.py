#!/usr/bin/env python3
"""Extract audio samples and pitch for neural synthesis training."""
import sys
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

def detect_pitch_autocorr(audio, sr, min_freq=50, max_freq=2000):
    """Simple autocorrelation-based pitch detection."""
    # Use first 0.1 seconds for pitch detection
    segment = audio[:int(sr * 0.1)]

    # Autocorrelation
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]

    # Find first peak after initial decay
    min_lag = int(sr / max_freq)
    max_lag = int(sr / min_freq)

    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return 440.0  # Default to A4

    peak_idx = np.argmax(search_region) + min_lag
    freq = sr / peak_idx
    return float(freq)

def extract_samples(wav_path, target_sr=8820, max_duration=1.0):
    """Extract downsampled audio samples."""
    sr, audio = wavfile.read(wav_path)

    # Convert to mono float normalized to [-1, 1]
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Normalize based on dtype
    if audio.max() > 1.0:
        audio = audio / 32768.0

    # Detect pitch before any processing
    pitch = detect_pitch_autocorr(audio, sr)

    # Trim to max duration
    max_samples = int(sr * max_duration)
    audio = audio[:max_samples]

    # Resample to target rate
    num_samples = int(len(audio) * target_sr / sr)
    resampled = resample(audio, num_samples)

    # Normalize to [0, 1] range for amplitude (take abs, scale)
    # Actually keep as-is for waveform matching
    peak = np.abs(resampled).max()
    if peak > 0:
        resampled = resampled / peak

    return {
        "original_sample_rate": int(sr),
        "target_sample_rate": int(target_sr),
        "num_samples": len(resampled),
        "duration_seconds": len(resampled) / target_sr,
        "detected_pitch_hz": round(pitch, 2),
        "samples": resampled.tolist()
    }

if __name__ == "__main__":
    wav_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "samples.json"

    result = extract_samples(wav_path)

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"Extracted {result['num_samples']} samples at {result['target_sample_rate']} Hz")
    print(f"Duration: {result['duration_seconds']:.3f}s")
    print(f"Detected pitch: {result['detected_pitch_hz']} Hz")
