#!/usr/bin/env python3
"""Exact numpy model of the realtime Latent Snare voice (one hit at n=0).

This is the forward model the E1/E2 trainer must implement (in DGenLazy) so
trained weights transfer: same decoder, same squashes, same closed-form bank,
same 16 RBJ band-passes, same envelopes, same drive. `python3
reference_render.py` checks it against the compiled dylib.

    y = render_ref(params, z=[..4], vel=1.0, tune=0, decay=1, level=0.7, seconds=1.0, sr=48000,
                   white=None)   # white=None reproduces dgen's xorshift32 seed-1 stream
"""
import json, math, os, sys
import numpy as np

import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e3proto")
LAY = json.load(open(os.path.join(MAD.code("e3proto"), "decoder_layout.json")))  # frozen E3 prototype contract (K=24)


def load_weights(d):
    return {k: np.array(json.load(open(os.path.join(d, v["file"])))["data"], np.float64).reshape(v["shape"])
            for k, v in LAY["weights"].items()}


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def decode(W, z, vel):
    """-> dict segment name -> squashed physical values (numpy arrays)."""
    x = np.array([[*z, min(max(vel, 0.0), 1.0)]])
    h = np.tanh(x @ W["l1.W"] + W["l1.b"])
    h = np.tanh(h @ W["l2.W"] + W["l2.b"])
    u = (h @ W["l3.W"] + W["l3.b"])[0]
    out = {}
    for s in LAY["segments"]:
        v = u[s["offset"]: s["offset"] + s["width"]]
        if s["squash"] == "log":
            out[s["name"]] = np.exp(math.log(s["lo"]) + (math.log(s["hi"]) - math.log(s["lo"])) * sigmoid(v))
        elif s["squash"] == "lin":
            out[s["name"]] = s["lo"] + (s["hi"] - s["lo"]) * sigmoid(v)
        else:
            out[s["name"]] = math.pi * np.tanh(v)
    return out


def xorshift_white(n):
    """dgen (noise) from a fresh voice (state 0 -> seed 1), mapped to
    zero-mean unit variance exactly as dsp.lisp does."""
    s = 1
    out = np.empty(n)
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        out[i] = np.float32(s / 4294967296.0)
    return 3.4641016 * (out - 0.5)


def band_coeffs(sr):
    fc = np.array(LAY["noise_bands"]["fc"]); Q = LAY["noise_bands"]["Q"]
    w0 = 2 * np.pi * fc / sr
    al = np.sin(w0) / (2 * Q)
    return al / (1 + al), -2 * np.cos(w0) / (1 + al), (1 - al) / (1 + al)


def bandpass_bank(x, sr):
    """[n] white -> [n, 16] unit-RMS band signals N_b = norm_b * BP(BP(x)).
    Each stage DF1, zero initial state: y[n] = b0 (x[n] - x[n-2]) - a1 y[n-1] - a2 y[n-2]."""
    b0, a1, a2 = band_coeffs(sr)
    norm = np.array(LAY["noise_bands"]["norm"][str(int(sr))])
    y = np.repeat(np.asarray(x, np.float64)[:, None], len(b0), axis=1)
    for _ in range(LAY["noise_bands"]["stages"]):
        out = np.zeros_like(y); x1 = x2 = y1 = y2 = np.zeros(len(b0))
        for n in range(len(y)):
            v = b0 * (y[n] - x2) - a1 * y1 - a2 * y2
            out[n] = v; x2, x1, y2, y1 = x1, y[n], y1, v
        y = out
    return y * norm


def smooth01(x):
    c = np.clip(x, 0, 1)
    return c * c * (3 - 2 * c)


def render_ref(W, z, vel=1.0, tune=0.0, decay=1.0, level=0.7, seconds=1.0, sr=48000.0, white=None, bands=None,
               parts=False):
    """One hit at n=0 of a fresh voice. Mirrors dsp.lisp term by term."""
    n = int(seconds * sr)
    age = np.arange(n) / sr
    p = decode(W, z, vel)
    g = {k: float(v[0]) for k, v in p.items() if len(v) == 1}
    f = p["mode_freq"] * math.exp(tune * math.log(2) / 12)
    sh = f * p["mode_glide"]
    nyq = np.clip((0.48 * sr - (f + np.maximum(sh, 0))) / (0.08 * sr), 0, 1)
    amp = p["mode_amp"] * nyq
    T = np.maximum(0, age - g["pre"])[:, None]
    t = np.maximum(0, T - p["mode_t0"])
    env = amp * np.exp(-t / (p["mode_decay"] * decay)) * (1 - np.exp(-t / p["mode_rise"]))
    tau = p["mode_gtau"]
    cyc = f * t + sh * tau * (1 - np.exp(-t / tau)) + p["mode_phase"] / (2 * math.pi)
    head = np.sum(env * np.sin(2 * np.pi * cyc), axis=1)
    E = np.sqrt(np.sum(env * env, axis=1) + 1e-10)
    if bands is None:
        bands = bandpass_bank(xorshift_white(n) if white is None else white, sr)
    nt = np.maximum(0, age - g["noise_onset"])[:, None]
    att = 1 - np.exp(-nt / g["noise_rise"])
    A, L = p["band_amp"], p["band_tail"]
    benv = att * (A * (1 - L) * np.exp(-nt / (p["band_d1"] * decay)) + A * L * np.exp(-nt / (p["band_d2"] * decay))) \
        + p["band_couple"] * E[:, None]
    noise = np.sum(benv * bands[:n], axis=1)
    dry = head + noise
    driven = dry + g["drive_amt"] * (np.tanh(dry * g["drive"]) * g["drive_norm"] - dry)
    f1 = 1 - smooth01((age - (g["len"] - g["fade"])) / g["fade"])
    fade = f1 * (1 - g["fade_curve"]) + f1 ** 4 * g["fade_curve"]
    y = driven * fade * g["gain"] * level
    return (y, head, noise) if parts else y


if __name__ == "__main__":
    MAD.audition()
    from audition import Instrument
    D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Latent Snare")
    inst = Instrument(D); W = load_weights(D)
    rng = np.random.default_rng(3)
    for trial in range(4):
        z = list(rng.uniform(-2, 2, 4)); vel = float(rng.uniform(0.3, 1)); tune = float(rng.uniform(-5, 5))
        dec = float(rng.uniform(0.5, 1.5))
        y, _ = inst.render(1.0, vel=vel, params={"z1": z[0], "z2": z[1], "z3": z[2], "z4": z[3],
                                               "tune": tune, "decay": dec, "level": 0.7})
        r = render_ref(W, z, vel, tune, dec, 0.7)
        err = np.sqrt(np.mean((y - r) ** 2) / np.mean(r ** 2))
        e10 = np.sqrt(np.mean((y[:4800] - r[:4800]) ** 2) / np.mean(r[:4800] ** 2))
        print(f"z={np.round(z, 2)} vel={vel:.2f} tune={tune:+.2f} decay={dec:.2f}: rel rms err {err:.2e} (first 100 ms {e10:.2e})")
