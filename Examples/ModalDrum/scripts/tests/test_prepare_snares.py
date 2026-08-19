#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

SAMPLE_RATE = 48_000
SCRIPT = Path(__file__).resolve().parents[1] / "prepare_snares.sh"


def write_wav(path, samples, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for sample in samples:
        integer = max(-32768, min(32767, round(sample * 32767)))
        values.extend([integer] * channels)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack(f"<{len(values)}h", *values))


def burst(length, onset=200):
    result = [0.0] * length
    for index in range(onset, length):
        age = (index - onset) / SAMPLE_RATE
        result[index] = 0.7 * math.exp(-age / 0.12) * math.sin(2 * math.pi * 800 * age)
    return result


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class PrepareSnaresTests(unittest.TestCase):
    def test_pipeline_filters_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "prepared"
            length = SAMPLE_RATE * 3 // 4
            write_wav(source / "valid_stereo.wav", burst(length), channels=2)

            loop = burst(length)
            restart = int(0.45 * SAMPLE_RATE)
            for index in range(restart, length):
                age = (index - restart) / SAMPLE_RATE
                loop[index] += 0.6 * math.exp(-age / 0.15) * math.sin(2 * math.pi * 600 * age)
            write_wav(source / "loop.wav", loop)
            write_wav(source / "dc.wav", [sample + 0.15 for sample in burst(length)])
            write_wav(source / "clipped.wav", [1.0 if 200 <= i < 500 else 0.0 for i in range(length)])

            subprocess.run([str(SCRIPT), str(source), str(output)], check=True)
            first_hash = tree_hash(output)
            manifest = json.loads((output / "manifest.json").read_text())
            entries = {entry["source"]: entry for entry in manifest["files"]}

            self.assertEqual(manifest["summary"], {"accepted": 1, "rejected": 3, "total": 4})
            self.assertEqual(entries["valid_stereo.wav"]["decision"], "accepted")
            self.assertIn("rms_rerise_after_300ms", entries["loop.wav"]["reasons"])
            self.assertIn("dc_offset", entries["dc.wav"]["reasons"])
            self.assertIn("clipped", entries["clipped.wav"]["reasons"])
            self.assertEqual(entries["valid_stereo.wav"]["pre_roll_samples"], 32)
            self.assertGreater(entries["valid_stereo.wav"]["original_peak"], 0.6)

            with wave.open(str(output / "valid_stereo.wav"), "rb") as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()),
                                 (44_100, 1, 33_075))
                samples = struct.unpack(f"<{wav.getnframes()}h", wav.readframes(wav.getnframes()))
            self.assertAlmostEqual(max(abs(value) for value in samples) / 32767.0, 0.99, places=4)
            first_audible = next(i for i, value in enumerate(samples) if abs(value) > 100)
            self.assertGreaterEqual(first_audible, 30)
            self.assertLessEqual(first_audible, 34)
            self.assertEqual(manifest["config"]["highpass_hz"], 30.0)

            subprocess.run([str(SCRIPT), str(source), str(output)], check=True)
            self.assertEqual(tree_hash(output), first_hash)

    def test_highpass_rejects_dc(self):
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            import prepare_snares
            filtered = prepare_snares.highpass([1.0] * 44_100)
        finally:
            sys.path.pop(0)
        self.assertLess(abs(filtered[-1]), 1e-6)
        self.assertGreater(abs(filtered[0]), 0.9)


if __name__ == "__main__":
    unittest.main()
