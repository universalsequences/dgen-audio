#!/usr/bin/env python3
"""Extract amplitude envelope from wav file for DGen neural synthesis test."""
import sys
import json
import numpy as np
from scipy.io import wavfile

def extract_envelope(wav_path, num_points=64, hop_size=512):
    """Extract RMS amplitude envelope."""
    sr, audio = wavfile.read(wav_path)

    # Convert to mono float
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / 32768.0

    # Compute RMS in windows
    envelope = []
    for i in range(0, len(audio) - hop_size, hop_size):
        window = audio[i:i + hop_size]
        rms = np.sqrt(np.mean(window ** 2))
        envelope.append(float(rms))

    # Resample to num_points
    indices = np.linspace(0, len(envelope) - 1, num_points)
    resampled = np.interp(indices, range(len(envelope)), envelope)

    # Normalize to 0-1
    resampled = resampled / (resampled.max() + 1e-8)

    return {
        "sample_rate": sr,
        "num_points": num_points,
        "duration_seconds": len(audio) / sr,
        "envelope": resampled.tolist()
    }

if __name__ == "__main__":
    wav_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "envelope.json"

    result = extract_envelope(wav_path)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Extracted {result['num_points']} point envelope from {result['duration_seconds']:.2f}s audio")
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Envelope range: {min(result['envelope']):.3f} - {max(result['envelope']):.3f}")
