"""Shared helpers: compile/load VILLAIN Snare (stem-tap copy) through eseq's pinned
DGenLisp compiler via eseq's tools/audition, float WAV IO, numpy head re-synthesis."""
import json, math, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
RESEARCH = MAD.ROOT
REPO = MAD.audition(pinned=True)
os.environ.setdefault("DGEN_ROOT", "/tmp")  # only used as cwd when no compiler is given
from audition import Instrument  # noqa: E402

SR = 48000.0
C4 = 261.6255653  # what eseq sends at transpose 0 (audio/params.rs); tuning = 2*C4/440
ORIG = os.path.join(REPO, "content/instruments/Drums/VILLAIN Snare")
STEMS = os.path.join(RESEARCH, "villain_stems")
NOISE_KNOBS = ["rattle", "buzz", "crack", "room", "extra", "strokes", "hiss", "dust"]

def load(stems=True, max_frames=128):
    return Instrument(STEMS if stems else ORIG, sample_rate=SR, max_frames=max_frames)

def render(inst, params, vel=1.0, seconds=1.0, pitch=C4):
    """Fresh state, one trigger at sample 0, gate held (Release inert).
    Returns dict of float32 channels: out (+ head/source/strands for the stem copy)."""
    y, _ = inst.render(seconds, pitch=pitch, vel=vel, params=params)
    if y.ndim == 1:
        return {"out": y}
    names = ["out", "head", "source", "strands"][: y.shape[1]]
    return {n: np.ascontiguousarray(y[:, i]) for i, n in enumerate(names)}

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

def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def head_resynth(env, n, vel=1.0, pitch=C4, knobs=None):
    """numpy float64 re-synthesis of head_g from block-rate values (villain_table)."""
    k = {"tune": 1.0, "decay": 1.0, "bend": 1.0}
    k.update(knobs or {})
    sr = SR
    age = np.arange(n) / sr
    T = np.maximum(0, age - env["c_pre"])
    tuning = 2 * min(max(pitch / 440, 0.25), 4) * k["tune"]
    inv_dec = 1 / k["decay"]
    vel_e = max(vel, 1e-4)
    vpow = lambda e: (vel > 0) * math.exp(e * math.log(vel_e))
    gd_sagg = np.exp(0.115129255 * env["c_gd_sag"] * np.clip((T - env["c_gd_sag_ms"]) / max(0.001, env["c_gd_ms"] - env["c_gd_sag_ms"]), 0, 1))
    GD = (1 - env["c_gd_depth"]) + env["c_gd_depth"] * gd_sagg * (1 - smooth01((T - env["c_gd_ms"]) / env["c_gd_fall"]))
    GH = 1 - env["c_gh_depth"] * smooth01((T - env["c_gh_ms"]) / env["c_gh_fall"])
    acc = np.zeros(n)
    for i in range(32):
        f, g = env[f"b{i}_f"], env[f"b{i}_gf"]
        t = np.maximum(0, T - env[f"b{i}_t0"])
        sh = max(-0.9 * f, k["bend"] * g)
        tau = env[f"b{i}_tau"]
        se = np.exp(-t / tau)
        hz = tuning * (f + sh * se)
        m = (env[f"b{i}_amp"] * np.exp(-t * env[f"b{i}_idec"] * inv_dec) * (1 - np.exp(-t * env[f"b{i}_irise"]))
             * np.clip((0.48 * sr - hz) / (0.08 * sr), 0, 1)
             * np.sin(2 * np.pi * tuning * (f * t + sh * tau * (1 - se)) + env[f"b{i}_ph"]))
        acc += m * (1 - env[f"b{i}_mh"] * (1 - GH)) * (1 - env[f"b{i}_md"] * (1 - GD))
    head_v = env["c_head"] * vpow(env["c_hv"]) * (1 - np.exp(-T / env["c_hr"]))
    return head_v * acc
