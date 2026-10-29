"""Load the Swift trainer's exports (weights.json + codes.json) for numpy use."""
import json, os
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT

def arr(v): return np.asarray(v["data"], np.float64).reshape(v["shape"])

def load_run(run):
    w = json.load(open(os.path.join(run, "weights.json")))
    W = {f"{k}.W": arr(w[k]["W"]) for k in ("l1", "l2", "l3")}
    W.update({f"{k}.b": arr(w[k]["b"]) for k in ("l1", "l2", "l3")})
    c = json.load(open(os.path.join(run, "codes.json")))
    Z = np.array([c["codes"][n] for n in c["order"]])
    return W, Z, c["order"]

def e1_bank():
    bj = json.load(open(os.path.join(R, "e1/noise_bands/bands.json")))
    return np.fromfile(os.path.join(R, "e1/noise_bands/bands.f32"), np.float32).reshape(len(bj["bands"]), bj["samples"]).T.astype(np.float64)

def decode_np(W, z, vel):
    x = np.array([[*z, min(max(vel, 0.0), 1.0)]])
    h = np.tanh(x @ W["l1.W"] + W["l1.b"]); h = np.tanh(h @ W["l2.W"] + W["l2.b"])
    return (h @ W["l3.W"] + W["l3.b"])[0]
