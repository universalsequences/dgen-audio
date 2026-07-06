#!/usr/bin/env python3
"""Independent MR-STFT comparison helper for SynthID outputs."""

import argparse
import struct

import numpy as np


def read_wav(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    offset = 12
    fmt = None
    data_chunk = None
    while offset + 8 <= len(blob):
        chunk_id = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        end = start + size
        if chunk_id == b"fmt ":
            audio_format, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", blob, start)
            fmt = audio_format, channels, rate, bits
        elif chunk_id == b"data":
            data_chunk = blob[start:end]
        offset = end + (size & 1)

    if fmt is None or data_chunk is None:
        raise ValueError(f"{path} is missing fmt or data chunk")

    audio_format, channels, rate, bits = fmt
    if audio_format == 1 and bits == 16:
        data = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 1 and bits == 24:
        raw = np.frombuffer(data_chunk, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        data = values.astype(np.float32) / 8388608.0
    elif audio_format == 3 and bits == 32:
        data = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format={audio_format} bits={bits}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def mrstft(a, b, windows=(256, 512, 1024, 2048)):
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    total = 0.0
    for win in windows:
        hop = max(1, win // 4)
        window = np.hanning(win).astype(np.float32)
        count = 0
        value = 0.0
        for start in range(0, max(1, n - win + 1), hop):
            aa = np.fft.rfft(a[start:start + win] * window)
            bb = np.fft.rfft(b[start:start + win] * window)
            value += np.mean(np.abs(np.log(np.abs(aa) + 1e-7) - np.log(np.abs(bb) + 1e-7)))
            count += 1
        total += value / max(count, 1)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--learned", required=True)
    args = parser.parse_args()
    target, sr1 = read_wav(args.target)
    learned, sr2 = read_wav(args.learned)
    if sr1 != sr2:
        raise SystemExit(f"sample-rate mismatch: {sr1} vs {sr2}")
    print(f"mrstft={mrstft(target, learned):.6f}")


if __name__ == "__main__":
    main()
