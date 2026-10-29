"""E7 helpers: E2 trained decoder (trainer space) + numpy twin with an E5 W2 wire layer."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e7"); R = MAD.ROOT
MAD.use("dataset", "e1", "e5", "e2")
import template as tp          # e2 twin (decode, squash, layout)
import latent_np as ln
from e5lib import wires, buzz_bp, E, MODES, head_modes, out as v_out, stem, strand_out_target, out_r0, villain_wire_params
from metric import score as mr_score, band_score
SR = 48000
RUN = os.path.join(R, "e2/runs/outbox_d4")
W, Z, NAMES = ln.load_run(RUN)
MAN = {it["id"]: it for it in json.load(open(os.path.join(R, "data/villain/manifest.json")))["items"]}
CAL = json.load(open(os.path.join(R, "e2/calib.json")))
NAME2K = {MAN[f"snare_{k:02d}_v100"]["name"]: k for k in range(32)}
BANDS = ln.e1_bank()
SEG = tp.SEG
WIRE = ["wire_thr", "wire_level", "wire_lo", "wire_hi", "wire_k"]
WFIX = dict(drive=1.2, hz=620.0, t60=1.0)
DET2 = [0.951, 1.062]

def rid_of(name, vel=1.0): return f"snare_{NAME2K[name]:02d}_v{int(round(vel * 100)):03d}"
def code(name): return Z[NAMES.index(name)]
def decode(z, vel, Wt=None): return tp.decode(Wt or W, z, vel)

def smooth01(x): c = np.clip(x, 0, 1); return c * c * (3 - 2 * c)

def parts(p, n=SR, couple=True, bands=BANDS, decay=1.0):
    """template.render_natural split: returns head (gated, pre-drive), noise, chain fn."""
    age = np.arange(n) / SR
    g = {k: float(np.ravel(v)[0]) for k, v in p.items() if np.size(v) == 1}
    f = np.asarray(p["mode_freq"]); sh = f * np.asarray(p["mode_glide"])
    nyq = np.clip((0.48 * SR - (f + np.maximum(sh, 0))) / (0.08 * SR), 0, 1)
    amp = np.asarray(p["mode_amp"]) * nyq
    T = np.maximum(0, age - g["pre"])[:, None]
    GH = 1 - g["gh_depth"] * smooth01((T - g["gh_time"]) / g["gh_fall"])
    GD = 1 - g["gd_depth"] * smooth01((T - g["gd_time"]) / g["gd_fall"])
    t = np.maximum(0, T - np.asarray(p["mode_t0"]))
    env = amp * GH * GD * np.exp(-t / (np.asarray(p["mode_decay"]) * decay)) * (1 - np.exp(-t / np.asarray(p["mode_rise"])))
    tau = np.asarray(p["mode_gtau"])
    cyc = f * t + sh * tau * (1 - np.exp(-t / tau)) + np.asarray(p["mode_phase"]) / (2 * math.pi)
    head = np.sum(env * np.sin(2 * np.pi * cyc), axis=1)
    Eh = np.sqrt(np.sum(env * env, axis=1) + 1e-10)
    nt = np.maximum(0, age - g["noise_onset"])[:, None]
    att = 1 - np.exp(-nt / g["noise_rise"])
    A, L = np.asarray(p["band_amp"]), np.asarray(p["band_tail"])
    C = np.asarray(p["band_couple"]) if couple else 0.0
    benv = att * (A * (1 - L) * np.exp(-nt / (np.asarray(p["band_d1"]) * decay))
                  + A * L * np.exp(-nt / (np.asarray(p["band_d2"]) * decay))) + C * Eh[:, None]
    noise = (1 - g["gate_noise"] * (1 - GD[:, 0])) * np.sum(benv * bands[:n], axis=1)
    f1 = 1 - smooth01((age - (g["len"] - g["fade"])) / g["fade"])
    fade = f1 * (1 - g["fade_curve"]) + f1 ** 4 * g["fade_curve"]
    def chain(dry, level=1.0):
        d0 = dry + g["drive_amt"] * (np.tanh(dry * g["drive"]) * g["drive_norm"] - dry)
        u = d0 / g["clip_ceil"]
        d = d0 + g["clip_mix"] * (g["clip_ceil"] * u / np.sqrt(np.sqrt(1 + u ** 4)) - d0)
        return d * fade * g["gain"] * level
    lin = fade * g["gain"]  # linearised chain gain (drive/clip ignored)
    return head, noise, chain, lin

def wire_raw(head, thr, level, drive=WFIX["drive"], hz=WFIX["hz"], t60=WFIX["t60"]):
    """E5 W2 layer pre band-pass (dsp: wire-raw), level = decoded wire_level."""
    return wires(head, nstr=2, det=DET2, zs=drive, gap=thr * drive, wire_hz=hz, t60_ms=t60, level=60.0 * level).astype(np.float64)

def wire_out(head, k):
    return buzz_bp(wire_raw(head, k["wire_thr"], k["wire_level"]), k["wire_lo"], k["wire_hi"], k["wire_k"]).astype(np.float64)

def render(p, knobs=None, couple=True, level=1.0, n=SR):
    head, noise, chain, _ = parts(p, n, couple)
    dry = head + noise
    if knobs is not None: dry = dry + wire_out(head, knobs)
    return chain(dry, level)

def scores(y, rid):
    t = v_out(rid); c = CAL[rid]
    mr, bd = mr_score(y, t), band_score(y, t)
    return dict(mr=mr, band=bd, mr_norm=(mr - c["mr_floor"]) / (c["mr_nearest"] - c["mr_floor"]),
                band_norm=(bd - c["band_floor"]) / (c["band_nearest"] - c["band_floor"]))

def pos(name, v):
    """natural -> sigmoid position; u = logit."""
    s = SEG[name]; return (math.log(v) - math.log(s["lo"])) / (math.log(s["hi"]) - math.log(s["lo"]))

def noise_basis(p, n=SR, bands=BANDS, decay=1.0):
    """noise = Nb @ m + Nc: Nb[:, b] = band b's own-envelope noise (band_amp scaled by m_b),
    Nc = the band_couple (head-envelope) part."""
    age = np.arange(n) / SR
    g = {k: float(np.ravel(v)[0]) for k, v in p.items() if np.size(v) == 1}
    q = dict(p); q["band_amp"] = np.zeros(16)
    _, nc, _, _ = parts(q, n, True, bands, decay)
    T = np.maximum(0, age - g["pre"])
    GD = 1 - g["gd_depth"] * smooth01((T - g["gd_time"]) / g["gd_fall"])
    nt = np.maximum(0, age - g["noise_onset"])[:, None]
    att = 1 - np.exp(-nt / g["noise_rise"])
    A, L = np.asarray(p["band_amp"]), np.asarray(p["band_tail"])
    benv = att * (A * (1 - L) * np.exp(-nt / (np.asarray(p["band_d1"]) * decay)) + A * L * np.exp(-nt / (np.asarray(p["band_d2"]) * decay)))
    return (1 - g["gate_noise"] * (1 - GD))[:, None] * benv * bands[:n], nc
