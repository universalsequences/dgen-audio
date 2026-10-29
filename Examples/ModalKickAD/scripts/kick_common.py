"""Shared helpers for the latent-kick research: compile/load VILLAIN Kick (original or
stem-tap copy) through eseq's tools/audition, float WAV IO."""
import json, math, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
KICK = MAD.stage("kick")
RESEARCH = MAD.ROOT
REPO = MAD.audition(pinned=True)
os.environ.setdefault("DGEN_ROOT", "/tmp")
from audition import Instrument  # noqa: E402

SR = 48000.0
C4 = 261.6255653  # eseq transpose 0; VILLAIN Kick note = 2*pitch/440 = 1.18921 at C4
ORIG = os.path.join(REPO, "content/instruments/Drums/VILLAIN Kick")
STEMS = os.path.join(KICK, "villain_stems")
TAPS = ["out", "modal", "bank_all", "body_sig", "source", "texture", "limited"]
DATA = os.path.join(KICK, "data")

def load(stems=True, max_frames=128):
    return Instrument(STEMS if stems else ORIG, sample_rate=SR, max_frames=max_frames)

def render(inst, params, vel=1.0, seconds=1.0, pitch=C4, **kw):
    y, _ = inst.render(seconds, pitch=pitch, vel=vel, params=params, **kw)
    if y.ndim == 1:
        return {"out": y}
    return {n: np.ascontiguousarray(y[:, i]) for i, n in enumerate(TAPS[: y.shape[1]])}

def write_wav_f32(path, y, sr=SR):
    y = np.asarray(y, dtype="<f4")
    data = y.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, int(sr), int(sr) * 4, 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)) + data)

def read_wav_f32(path):
    b = open(path, "rb").read()
    i = b.index(b"data")
    n = struct.unpack("<I", b[i + 4:i + 8])[0]
    return np.frombuffer(b[i + 8:i + 8 + n], dtype="<f4")

def relrms(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2) / max(np.mean(b ** 2), 1e-30)))
