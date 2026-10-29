#!/usr/bin/env python3
"""numpy twin of the E2 Latent Snare voice (scripts/e3proto/gen_dsp.py, layout version e2).

decode(W, z, vel) -> natural params; render_natural(p, ...) -> one hit at n=0.
The Swift trainer (dgen Examples/ModalSnareAD, LatentModel.swift) implements the
same math; both read decoder_layout.json.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2")
R = MAD.ROOT
LAY = json.load(open(os.path.join(HERE, "decoder_layout.json")))
SEG = {s["name"]: s for s in LAY["segments"]}
SR = 48000.0


def env_setup():
    """Compile with this checkout's DGenLisp (scripts/make_dgenlisp_tool.sh) through eseq's tools/audition."""
    MAD.audition()
    os.environ.setdefault("DGEN_ROOT", "/tmp")


def load_weights(d):
    return {k: np.array(json.load(open(os.path.join(d, v["file"])))["data"], np.float64).reshape(v["shape"])
            for k, v in LAY["weights"].items()}


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def squash(name, u):
    s = SEG[name]
    if s["squash"] == "log":
        return np.exp(math.log(s["lo"]) + (math.log(s["hi"]) - math.log(s["lo"])) * sigmoid(u))
    if s["squash"] == "lin":
        return s["lo"] + (s["hi"] - s["lo"]) * sigmoid(u)
    return math.pi * np.tanh(u)


def unsquash(name, x, eps=1e-3):
    """natural -> sigmoid position p in [eps, 1-eps] (pi: (x/pi+1)/2)."""
    s = SEG[name]; x = np.asarray(x, np.float64)
    if s["squash"] == "log":
        p = (np.log(np.maximum(x, 1e-30)) - math.log(s["lo"])) / (math.log(s["hi"]) - math.log(s["lo"]))
    elif s["squash"] == "lin":
        p = (x - s["lo"]) / (s["hi"] - s["lo"])
    else:
        p = 0.5 * (np.arctanh(np.clip(x / math.pi, -1 + 1e-6, 1 - 1e-6)) * 0 + x / math.pi + 1)
    return np.clip(p, eps, 1 - eps)


def decode_u(W, z, vel):
    x = np.array([[*z, min(max(vel, 0.0), 1.0)]])
    h = np.tanh(x @ W["l1.W"] + W["l1.b"])
    h = np.tanh(h @ W["l2.W"] + W["l2.b"])
    return (h @ W["l3.W"] + W["l3.b"])[0]


def decode(W, z, vel):
    u = decode_u(W, z, vel)
    return {s["name"]: squash(s["name"], u[s["offset"]: s["offset"] + s["width"]]) for s in LAY["segments"]}


def xorshift_white(n):
    s = 1; out = np.empty(n)
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        out[i] = np.float32(s / 4294967296.0)
    return 3.4641016 * (out - 0.5)


def bandpass_bank(x, sr=SR):
    nb = LAY["noise_bands"]; fc = np.array(nb["fc"]); Q = nb["Q"]
    w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q)
    b0, a1, a2 = al / (1 + al), -2 * np.cos(w0) / (1 + al), (1 - al) / (1 + al)
    y = np.repeat(np.asarray(x, np.float64)[:, None], len(fc), axis=1)
    for _ in range(nb["stages"]):
        out = np.zeros_like(y); x1 = x2 = y1 = y2 = np.zeros(len(fc))
        for n in range(len(y)):
            v = b0 * (y[n] - x2) - a1 * y1 - a2 * y2
            out[n] = v; x2, x1, y2, y1 = x1, y[n], y1, v
        y = out
    return y * np.array(nb["norm"][str(int(sr))])


def smooth01(x):
    c = np.clip(x, 0, 1)
    return c * c * (3 - 2 * c)


def render_natural(p, tune=0.0, decay=1.0, level=0.7, seconds=1.0, sr=SR, bands=None, parts=False):
    n = int(seconds * sr); age = np.arange(n) / sr
    g = {k: float(np.ravel(v)[0]) for k, v in p.items() if np.size(v) == 1}
    f = np.asarray(p["mode_freq"]) * math.exp(tune * math.log(2) / 12)
    sh = f * np.asarray(p["mode_glide"])
    nyq = np.clip((0.48 * sr - (f + np.maximum(sh, 0))) / (0.08 * sr), 0, 1)
    amp = np.asarray(p["mode_amp"]) * nyq
    T = np.maximum(0, age - g["pre"])[:, None]
    GH = 1 - g["gh_depth"] * smooth01((T - g["gh_time"]) / g["gh_fall"])
    GD = 1 - g["gd_depth"] * smooth01((T - g["gd_time"]) / g["gd_fall"])
    t = np.maximum(0, T - np.asarray(p["mode_t0"]))
    env = amp * GH * GD * np.exp(-t / (np.asarray(p["mode_decay"]) * decay)) * (1 - np.exp(-t / np.asarray(p["mode_rise"])))
    tau = np.asarray(p["mode_gtau"])
    cyc = f * t + sh * tau * (1 - np.exp(-t / tau)) + np.asarray(p["mode_phase"]) / (2 * math.pi)
    head = np.sum(env * np.sin(2 * np.pi * cyc), axis=1)
    E = np.sqrt(np.sum(env * env, axis=1) + 1e-10)
    if bands is None:
        bands = bandpass_bank(xorshift_white(n), sr)
    nt = np.maximum(0, age - g["noise_onset"])[:, None]
    att = 1 - np.exp(-nt / g["noise_rise"])
    A, L = np.asarray(p["band_amp"]), np.asarray(p["band_tail"])
    benv = att * (A * (1 - L) * np.exp(-nt / (np.asarray(p["band_d1"]) * decay))
                  + A * L * np.exp(-nt / (np.asarray(p["band_d2"]) * decay))) + np.asarray(p["band_couple"]) * E[:, None]
    noise = (1 - g["gate_noise"] * (1 - GD[:, 0])) * np.sum(benv * bands[:n], axis=1)
    dry = head + noise
    driven0 = dry + g["drive_amt"] * (np.tanh(dry * g["drive"]) * g["drive_norm"] - dry)
    u = driven0 / g["clip_ceil"]
    driven = driven0 + g["clip_mix"] * (g["clip_ceil"] * u / np.sqrt(np.sqrt(1 + u ** 4)) - driven0)
    f1 = 1 - smooth01((age - (g["len"] - g["fade"])) / g["fade"])
    fade = f1 * (1 - g["fade_curve"]) + f1 ** 4 * g["fade_curve"]
    y = driven * fade * g["gain"] * level
    return (y, head, noise) if parts else y


def render(W, z, vel=1.0, **kw):
    return render_natural(decode(W, z, vel), **kw)


if __name__ == "__main__":
    # parity: compiled instrument (this checkout's DGenLisp) vs this twin, random/trained weights
    env_setup()
    from audition import Instrument
    D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Latent Snare")
    inst = Instrument(D); W = load_weights(D)
    rng = np.random.default_rng(3)
    for trial in range(4):
        z = list(rng.uniform(-2, 2, 4)); vel = float(rng.uniform(0.3, 1)); tune = float(rng.uniform(-5, 5))
        dec = float(rng.uniform(0.5, 1.5))
        y, _ = inst.render(1.0, vel=vel, params={"z1": z[0], "z2": z[1], "z3": z[2], "z4": z[3],
                                               "tune": tune, "decay": dec, "level": 0.7})
        r = render(W, z, vel, tune=tune, decay=dec, level=0.7)
        err = np.sqrt(np.mean((y - r) ** 2) / np.mean(r ** 2))
        print(f"z={np.round(z, 2)} vel={vel:.2f} tune={tune:+.2f} decay={dec:.2f}: rel rms err {err:.2e}")
