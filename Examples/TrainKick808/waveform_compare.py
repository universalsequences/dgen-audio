#!/usr/bin/env python3
import argparse
import cmath
import math
import struct
import zlib
from pathlib import Path


def read_wav(path):
    data = Path(path).read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    offset = 12
    fmt = None
    pcm = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        if chunk_id == b"fmt ":
            audio_format, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", data, start
            )
            fmt = (audio_format, channels, sample_rate, bits)
        elif chunk_id == b"data":
            pcm = data[start:end]
        offset = end + (size & 1)

    if fmt is None or pcm is None:
        raise ValueError(f"{path}: missing fmt or data chunk")

    audio_format, channels, sample_rate, bits = fmt
    samples = []

    if audio_format == 1 and bits == 16:
        total = len(pcm) // 2
        values = struct.unpack_from("<" + "h" * total, pcm, 0)
        samples = [v / 32768.0 for v in values]
    elif audio_format == 1 and bits == 24:
        total = len(pcm) // 3
        for i in range(total):
            b0, b1, b2 = pcm[i * 3 : i * 3 + 3]
            raw = b0 | (b1 << 8) | (b2 << 16)
            if raw & 0x800000:
                raw |= ~0xFFFFFF
            samples.append(raw / 8388608.0)
    elif audio_format == 3 and bits == 32:
        total = len(pcm) // 4
        samples = list(struct.unpack_from("<" + "f" * total, pcm, 0))
    else:
        raise ValueError(f"{path}: unsupported WAV format={audio_format} bits={bits}")

    if channels > 1:
        mono = []
        for i in range(0, len(samples), channels):
            mono.append(sum(samples[i : i + channels]) / channels)
        samples = mono

    return samples, sample_rate


def png_write(path, width, height, rgba):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(rgba[start : start + stride])

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    out.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
    out.extend(chunk(b"IEND", b""))
    Path(path).write_bytes(out)


def set_px(img, width, height, x, y, color):
    if 0 <= x < width and 0 <= y < height:
        i = (y * width + x) * 4
        img[i : i + 4] = bytes(color)


def rect(img, width, height, x, y, w, h, color):
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + w))
    y1 = min(height, int(y + h))
    for yy in range(y0, y1):
        base = yy * width * 4
        row = bytes(color) * max(0, x1 - x0)
        img[base + x0 * 4 : base + x1 * 4] = row


def line(img, width, height, x0, y0, x1, y1, color, thickness=1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    radius = max(0, thickness // 2)
    while True:
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                set_px(img, width, height, xx, yy, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


FONT = {
    "a": ["01110", "00001", "01111", "10001", "01111"],
    "b": ["10000", "10000", "11110", "10001", "11110"],
    "d": ["00001", "00001", "01111", "10001", "01111"],
    "e": ["01110", "10001", "11111", "10000", "01111"],
    "g": ["01111", "10001", "01111", "00001", "11110"],
    "l": ["11000", "01000", "01000", "01000", "11100"],
    "n": ["11110", "10001", "10001", "10001", "10001"],
    "r": ["10110", "11001", "10000", "10000", "10000"],
    "t": ["01000", "11100", "01000", "01000", "00111"],
}


def text(img, width, height, x, y, label, color, scale=3):
    cursor = x
    for ch in label.lower():
        glyph = FONT.get(ch)
        if glyph is None:
            cursor += 3 * scale
            continue
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    rect(img, width, height, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += 6 * scale


def lane_waveform(img, width, height, samples, x, y, w, h, color):
    peak = max((abs(s) for s in samples), default=1.0)
    scale = 0.42 * h / max(peak, 1e-9)
    mid = y + h // 2
    prev_x = x
    prev_y = mid
    for px in range(w):
        start = int(px * len(samples) / w)
        end = max(start + 1, int((px + 1) * len(samples) / w))
        bucket = samples[start:end]
        if not bucket:
            value = 0.0
        else:
            pos = max(bucket)
            neg = min(bucket)
            value = pos if abs(pos) >= abs(neg) else neg
        yy = int(round(mid - value * scale))
        xx = x + px
        if px:
            line(img, width, height, prev_x, prev_y, xx, yy, color, thickness=3)
        prev_x, prev_y = xx, yy


def render(learned, target, out, width, height):
    bg = (45, 45, 45, 255)
    img = bytearray(bg * (width * height))

    for gx in range(0, width, max(35, width // 18)):
        rect(img, width, height, gx, 0, 2, height, (38, 38, 38, 255))
    for gy in [70, 248, 423]:
        rect(img, width, height, 0, gy, width, 4, (27, 27, 27, 255))

    left = 107
    clip_w = min(width - left - 97, 445)
    lane_h = 174
    top_y = 75
    bottom_y = 250
    learned_col = (211, 159, 32, 255)
    target_col = (255, 164, 31, 255)
    grid_col = (166, 125, 28, 130)
    wave_col = (25, 25, 25, 255)

    for lane_y, col in [(top_y, learned_col), (bottom_y, target_col)]:
        rect(img, width, height, left, lane_y, clip_w, lane_h, col)
        rect(img, width, height, left, lane_y, clip_w, 31, tuple(min(255, c + 5) for c in col[:3]) + (255,))
        for i in range(0, clip_w + 1, max(35, clip_w // 12)):
            rect(img, width, height, left + i, lane_y + 32, 2, lane_h - 32, grid_col)
        fade_x = left + clip_w - 23
        for fx in range(23):
            alpha = fx / 22
            fade_h = int((1 - math.cos(alpha * math.pi)) * 0.5 * lane_h)
            rect(img, width, height, fade_x + fx, lane_y + lane_h - fade_h, 1, fade_h, (183, 135, 29, 255))

    text(img, width, height, left + 6, top_y + 7, "learned", wave_col, scale=3)
    text(img, width, height, left + 6, bottom_y + 7, "target", wave_col, scale=3)
    lane_waveform(img, width, height, learned, left + 3, top_y + 34, clip_w - 6, lane_h - 55, wave_col)
    lane_waveform(img, width, height, target, left + 3, bottom_y + 34, clip_w - 6, lane_h - 55, wave_col)

    png_write(out, width, height, img)


def normalized(samples):
    peak = max((abs(s) for s in samples), default=1.0)
    if peak <= 1e-12:
        return samples[:]
    return [s / peak for s in samples]


def mse(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum((a[i] - b[i]) ** 2 for i in range(n)) / n


def zero_crossings(samples):
    if not samples:
        return 0
    count = 0
    prev = samples[0]
    for value in samples[1:]:
        if (prev < 0 <= value) or (prev > 0 >= value):
            count += 1
        if value != 0:
            prev = value
    return count


def next_power_of_two(n):
    value = 1
    while value < n:
        value *= 2
    return value


def band_powers(samples, sample_rate, bands):
    if not samples:
        return {name: 0.0 for name, _, _ in bands}

    n_fft = max(256, next_power_of_two(len(samples)))
    windowed = []
    for i in range(n_fft):
        sample = samples[i] if i < len(samples) else 0.0
        if i < len(samples) and len(samples) > 1:
            win = 0.5 - 0.5 * math.cos(2 * math.pi * i / (len(samples) - 1))
        else:
            win = 0.0
        windowed.append(sample * win)

    powers = {name: 0.0 for name, _, _ in bands}
    centroid_num = 0.0
    centroid_den = 0.0
    for k in range(n_fft // 2 + 1):
        freq = k * sample_rate / n_fft
        acc = 0j
        for n, sample in enumerate(windowed):
            if sample:
                acc += sample * cmath.exp(-2j * math.pi * k * n / n_fft)
        power = (acc.real * acc.real + acc.imag * acc.imag) / n_fft
        centroid_num += freq * power
        centroid_den += power
        for name, lo, hi in bands:
            if lo <= freq < hi:
                powers[name] += power

    powers["centroid_hz"] = centroid_num / centroid_den if centroid_den > 1e-18 else 0.0
    return powers


def db_ratio(a, b):
    return 10.0 * math.log10(max(a, 1e-18) / max(b, 1e-18))


def time_slice_metrics(learned, target, sample_rate):
    bands = [
        ("low_0_1000", 0.0, 1000.0),
        ("presence_1000_4000", 1000.0, 4000.0),
        ("air_4000_12000", 4000.0, 12000.0),
        ("hf_2000_16000", 2000.0, 16000.0),
    ]
    slices = [
        ("0_5ms", 0.000, 0.005),
        ("5_10ms", 0.005, 0.010),
        ("10_20ms", 0.010, 0.020),
        ("20_30ms", 0.020, 0.030),
    ]
    out = {}
    for label, start_s, end_s in slices:
        start = int(sample_rate * start_s)
        end = min(len(learned), len(target), int(sample_rate * end_s))
        if end <= start:
            continue
        lb = band_powers(learned[start:end], sample_rate, bands)
        tb = band_powers(target[start:end], sample_rate, bands)
        learned_hf_ratio = db_ratio(lb["hf_2000_16000"], lb["low_0_1000"])
        target_hf_ratio = db_ratio(tb["hf_2000_16000"], tb["low_0_1000"])
        out[f"slice_{label}_air_4_12_delta_db"] = db_ratio(
            lb["air_4000_12000"], tb["air_4000_12000"]
        )
        out[f"slice_{label}_hf_2_16_delta_db"] = db_ratio(
            lb["hf_2000_16000"], tb["hf_2000_16000"]
        )
        out[f"slice_{label}_hf_to_low_delta_db"] = learned_hf_ratio - target_hf_ratio
    return out


def dft_magnitudes(samples, n_fft):
    mags = []
    for k in range(n_fft // 2 + 1):
        acc = 0j
        for n, sample in enumerate(samples):
            if sample:
                acc += sample * cmath.exp(-2j * math.pi * k * n / n_fft)
        mags.append(math.sqrt(acc.real * acc.real + acc.imag * acc.imag) / n_fft)
    return mags


def stft_db(samples, sample_rate, duration_ms=60, n_fft=256, hop=32):
    count = min(len(samples), int(sample_rate * duration_ms / 1000))
    frames = []
    for start in range(0, max(1, count - n_fft + 1), hop):
        chunk = []
        for i in range(n_fft):
            sample = samples[start + i] if start + i < count else 0.0
            win = 0.5 - 0.5 * math.cos(2 * math.pi * i / max(1, n_fft - 1))
            chunk.append(sample * win)
        mags = dft_magnitudes(chunk, n_fft)
        frames.append([20.0 * math.log10(max(m, 1e-8)) for m in mags])
    if not frames:
        frames = [[-160.0] * (n_fft // 2 + 1)]
    return frames


def heat_color(value, lo=-95.0, hi=-20.0):
    x = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    if x < 0.5:
        t = x * 2.0
        return (int(25 + t * 110), int(25 + t * 65), int(35 + t * 20), 255)
    t = (x - 0.5) * 2.0
    return (int(135 + t * 120), int(90 + t * 120), int(55 - t * 35), 255)


def delta_color(value, max_abs=24.0):
    x = max(-1.0, min(1.0, value / max_abs))
    if x < 0:
        t = -x
        return (int(35 + t * 35), int(55 + t * 80), int(75 + t * 180), 255)
    t = x
    return (int(65 + t * 190), int(55 + t * 40), int(45 + t * 15), 255)


def draw_spectrogram_panel(img, width, height, data, x, y, w, h, sample_rate, mode):
    frame_count = len(data)
    bin_count = len(data[0]) if data else 1
    max_freq = min(16_000.0, sample_rate / 2)
    max_bin = max(1, min(bin_count - 1, int(max_freq * (bin_count - 1) / (sample_rate / 2))))
    for px in range(w):
        frame = min(frame_count - 1, int(px * frame_count / max(1, w)))
        for py in range(h):
            # Log-ish frequency mapping gives the attack bands more visible space.
            norm = 1.0 - py / max(1, h - 1)
            freq = 40.0 * ((max_freq / 40.0) ** norm)
            b = max(0, min(max_bin, int(freq * (bin_count - 1) / (sample_rate / 2))))
            value = data[frame][b]
            color = delta_color(value) if mode == "delta" else heat_color(value)
            set_px(img, width, height, x + px, y + py, color)


def render_spectrogram(learned, target, sample_rate, out, width=760, height=560):
    learned = normalized(learned)
    target = normalized(target)
    learned_stft = stft_db(learned, sample_rate)
    target_stft = stft_db(target, sample_rate)
    delta_stft = []
    for lf, tf in zip(learned_stft, target_stft):
        delta_stft.append([lv - tv for lv, tv in zip(lf, tf)])

    img = bytearray((28, 28, 28, 255) * (width * height))
    left = 76
    top = 52
    panel_w = width - left - 36
    panel_h = 132
    gap = 42
    wave_col = (220, 220, 220, 255)

    rows = [
        ("learned", learned_stft, "mag"),
        ("target", target_stft, "mag"),
        ("delta", delta_stft, "delta"),
    ]
    for idx, (label, data, mode) in enumerate(rows):
        y = top + idx * (panel_h + gap)
        text(img, width, height, 8, y + 5, label, wave_col, scale=2)
        rect(img, width, height, left - 1, y - 1, panel_w + 2, panel_h + 2, (15, 15, 15, 255))
        draw_spectrogram_panel(img, width, height, data, left, y, panel_w, panel_h, sample_rate, mode)
        for frac in [0.25, 0.5, 0.75]:
            xx = left + int(panel_w * frac)
            rect(img, width, height, xx, y, 1, panel_h, (255, 255, 255, 35))
        for frac in [0.25, 0.5, 0.75]:
            yy = y + int(panel_h * frac)
            rect(img, width, height, left, yy, panel_w, 1, (255, 255, 255, 28))

    png_write(out, width, height, img)


def metrics(learned, target, sample_rate):
    n = min(len(learned), len(target))
    learned = normalized(learned[:n])
    target = normalized(target[:n])
    transient_n = min(n, int(sample_rate * 0.030))
    bands = [
        ("sub_0_200", 0.0, 200.0),
        ("body_200_1000", 200.0, 1000.0),
        ("presence_1000_4000", 1000.0, 4000.0),
        ("air_4000_12000", 4000.0, 12000.0),
        ("hf_2000_16000", 2000.0, 16000.0),
    ]
    learned_bands = band_powers(learned[:transient_n], sample_rate, bands)
    target_bands = band_powers(target[:transient_n], sample_rate, bands)
    stats = {
        "norm_mse": mse(learned, target),
        "transient_norm_mse": mse(learned[:transient_n], target[:transient_n]),
        "learned_zero_crossings_30ms": zero_crossings(learned[:transient_n]),
        "target_zero_crossings_30ms": zero_crossings(target[:transient_n]),
        "learned_peak": max((abs(s) for s in learned), default=0.0),
        "target_peak": max((abs(s) for s in target), default=0.0),
        "learned_centroid_30ms_hz": learned_bands["centroid_hz"],
        "target_centroid_30ms_hz": target_bands["centroid_hz"],
        "centroid_delta_30ms_hz": learned_bands["centroid_hz"] - target_bands["centroid_hz"],
    }
    for name, _, _ in bands:
        stats[f"{name}_db_delta_30ms"] = db_ratio(learned_bands[name], target_bands[name])
    stats.update(time_slice_metrics(learned[:transient_n], target[:transient_n], sample_rate))
    stats["selection_score"] = selection_score(stats)
    return {
        **stats
    }


def deficit(value):
    return max(0.0, -value)


def excess(value):
    return max(0.0, value)


def selection_score(stats):
    """Lower-is-better score biased toward the listening gaps found in this example."""
    score = 0.0
    score += stats["norm_mse"] * 120.0
    score += stats["transient_norm_mse"] * 80.0
    score += abs(stats["centroid_delta_30ms_hz"]) / 90.0
    score += abs(stats["sub_0_200_db_delta_30ms"]) / 4.0
    score += abs(stats["body_200_1000_db_delta_30ms"]) / 10.0
    score += excess(stats["presence_1000_4000_db_delta_30ms"]) / 12.0
    score += deficit(stats["air_4000_12000_db_delta_30ms"]) / 3.0
    score += deficit(stats["hf_2000_16000_db_delta_30ms"]) / 4.0

    for label in ["5_10ms", "10_20ms", "20_30ms"]:
        score += deficit(stats.get(f"slice_{label}_air_4_12_delta_db", 0.0)) / 4.0
        score += deficit(stats.get(f"slice_{label}_hf_2_16_delta_db", 0.0)) / 5.0
        score += abs(stats.get(f"slice_{label}_hf_to_low_delta_db", 0.0)) / 10.0

    # A small penalty for missing the target's early oscillation density.
    score += abs(
        stats["learned_zero_crossings_30ms"] - stats["target_zero_crossings_30ms"]
    ) / 8.0
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned", default="/tmp/dgen-train-kick808/learned.wav")
    parser.add_argument("--target", default="/tmp/dgen-train-kick808/target.wav")
    parser.add_argument("--out", default="/tmp/dgen-train-kick808/compare.png")
    parser.add_argument("--spectrogram-out")
    parser.add_argument("--width", type=int, default=650)
    parser.add_argument("--height", type=int, default=478)
    args = parser.parse_args()

    learned, learned_sr = read_wav(args.learned)
    target, target_sr = read_wav(args.target)
    n = min(len(learned), len(target))
    render(learned[:n], target[:n], args.out, args.width, args.height)
    print(args.out)
    if args.spectrogram_out:
        render_spectrogram(learned[:n], target[:n], min(learned_sr, target_sr), args.spectrogram_out)
        print(args.spectrogram_out)
    stats = metrics(learned[:n], target[:n], min(learned_sr, target_sr))
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
