"""E8 helpers: compile/render instruments, codes, metrics."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e8"); R = MAD.ROOT
MAD.audition()
os.chdir(HERE)  # build/, ref/, renders/ and the result JSONs live in $MODAL_AD_ROOT/e8
MAD.use("dataset"); MAD.use("e1")
from audition import Instrument
SR = 48000
INST = os.path.join(R, "e6", "stage", "Latent Snare")  # the A/B Latent Snare (scripts/e6/build_instrument.py)
CODES = json.load(open(os.path.join(INST, "codes_std.json")))["codes"]

def zp(code, pre="z"):
    return {f"{pre}{i+1}": float(code[i]) for i in range(4)}

def render(inst, seconds=1.0, vel=1.0, params=None, **kw):
    p = {"tune": 0.0, "decay": 1.0, "level": 1.0}; p.update(params or {})
    y, _ = inst.render(seconds, pitch=261.6256, vel=vel, params=p, **kw)
    return y.astype(np.float64)

def rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30)))

# ---- float64 twin of the E7 decoder + closed-form head (from the installed dsp text) ----
import re
def _load(name):
    d = json.load(open(os.path.join(INST, name))); return np.array(d["data"], np.float64).reshape(d["shape"])
_W = [(_load("w1.json"), _load("b1.json")), (_load("w2.json"), _load("b2.json")), (_load("w3.json"), _load("b3.json"))]
_SQ = {}
for line in open(os.path.join(INST, "dsp.lisp")):
    if not line.startswith("(def raw-"): continue
    name = line.split()[1][4:]
    i, j = map(int, re.search(r"\[0:1,(\d+):(\d+)\]", line).groups())
    if "(tanh" in line: _SQ[name] = ("pi", i, j, 0, 0)
    else:
        A, B = map(float, re.search(r"\(\+ (-?[\d.e-]+) \(\* ([\d.e-]+) \(sigmoid", line).groups())
        _SQ[name] = ("exp" if line.split()[2] == "(exp" else "lin", i, j, A, B)

def decode64(z, vel=1.0):
    x = np.array(list(z) + [vel])[None]
    h = np.tanh(x @ _W[0][0] + _W[0][1]); h = np.tanh(h @ _W[1][0] + _W[1][1]); u = (h @ _W[2][0] + _W[2][1])[0]
    p = {}
    for n, (k, i, j, A, B) in _SQ.items():
        s = 1 / (1 + np.exp(-u[i:j]))
        v = np.pi * np.tanh(u[i:j]) if k == "pi" else (np.exp(A + B * s) if k == "exp" else A + B * s)
        p[n] = v if j - i > 1 else float(v[0])
    return p

def smooth(x): c = np.clip(x, 0, 1); return c * c * (3 - 2 * c)

def head64(z, vel=1.0, n=SR, tune=0.0, decay=1.0, ratio=1.0):
    p = decode64(z, vel); age = np.arange(n) / SR
    f = p["mode_freq"] * ratio * np.exp(tune * 0.057762265046662105); sh = f * p["mode_glide"]
    nyq = np.clip((0.48 * SR - (f + np.maximum(sh, 0))) / (0.08 * SR), 0, 1)
    T = np.maximum(0, age - p["pre"])[:, None]
    GH = 1 - p["gh_depth"] * smooth((T - p["gh_time"]) / p["gh_fall"])
    GD = 1 - p["gd_depth"] * smooth((T - p["gd_time"]) / p["gd_fall"])
    t = np.maximum(0, T - p["mode_t0"])
    env = p["mode_amp"] * nyq * GH * GD * np.exp(-t / (p["mode_decay"] * decay)) * (1 - np.exp(-t / p["mode_rise"]))
    cyc = f * t + sh * p["mode_gtau"] * (1 - np.exp(-t / p["mode_gtau"])) + p["mode_phase"] / (2 * np.pi)
    return np.sum(env * np.sin(2 * np.pi * cyc), axis=1)
