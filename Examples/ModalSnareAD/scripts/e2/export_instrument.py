#!/usr/bin/env python3
"""Trainer run -> realtime instrument.
Codes are re-expressed exactly as z' = (z - mu) V / s (PCA-rotated, unit variance over the
32 snares; z1 = largest axis), with W1z' = diag(s) V^T W1z and b1' = b1 + mu W1z, so the
decoder output is unchanged. Writes w1..b3.json (export_weights.py format) + dsp.lisp
(gen_dsp.py, LATENT_D=D) + codes_std.json into DEST.
    python3 export_instrument.py RUN DEST"""
import json, os, subprocess, sys, shutil
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT; MAD.use("e2")
import latent_np as ln
run, dest = sys.argv[1], sys.argv[2]
W, Z, names = ln.load_run(run); D = Z.shape[1]
mu = Z.mean(0); U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
sc = s / np.sqrt(len(Z)); V = Vt.T
Zs = (Z - mu) @ V / sc
# the instrument clips z to its knob range [-3, 3]: shrink so every snare code sits within
# +-2.5 (the prior over codes is then N(0, prior_std^2) per knob)
kfac = max(1.0, float(np.abs(Zs).max()) / 2.5)
Zs = Zs / kfac; sc = sc * kfac
W1 = W["l1.W"]; W1z, w1v = W1[:D], W1[D:]
W1z_s = (np.diag(sc) @ V.T) @ W1z
b1_s = W["l1.b"] + mu @ W1z
Wn = dict(W); Wn["l1.W"] = np.vstack([W1z_s, w1v]); Wn["l1.b"] = b1_s
# exactness check
for i in (0, 7, 19):
    a = ln.decode_np(W, Z[i], 0.7); b = ln.decode_np(Wn, Zs[i], 0.7)
    assert np.max(np.abs(a - b)) < 1e-6 * max(1, np.abs(a).max()), np.max(np.abs(a - b))
os.makedirs(dest, exist_ok=True)
src = os.path.join(MAD.code("e3proto"), "Latent Snare")  # instrument.json + ui.lisp templates
for f in ("instrument.json", "ui.lisp"):
    if not os.path.exists(os.path.join(dest, f)): shutil.copy(os.path.join(src, f), dest)
env = dict(os.environ, LATENT_D=str(D))
subprocess.run([sys.executable, os.path.join(MAD.code("e3proto"), "gen_dsp.py"), "--out", dest,
                "--layout", os.path.join(dest, "decoder_layout.json")], check=True, env=env)
files = {"l1.W": "w1.json", "l1.b": "b1.json", "l2.W": "w2.json", "l2.b": "b2.json", "l3.W": "w3.json", "l3.b": "b3.json"}
for k, f in files.items():
    a = Wn[k]; json.dump({"shape": list(a.shape), "data": [round(float(x), 8) for x in a.reshape(-1)]}, open(os.path.join(dest, f), "w"))
codes = {n: [round(float(x), 5) for x in Zs[i]] for i, n in enumerate(names)}
json.dump({"D": D, "codes": codes, "order": names, "knob_range": [-3, 3],
           "transform": "z_knob = (z_raw - mu) V / s (PCA over the 32 codes, then shrunk so max |z| = 2.5)",
           "prior_std": 1.0 / kfac,
           "mu": mu.tolist(), "V": V.tolist(), "s": sc.tolist(),
           "velocity": "decoder input after z1..zD = eseq note velocity in [0,1]; trained at 0.4/0.7/1.0",
           "source_run": os.path.relpath(os.path.abspath(run), MAD.ROOT)}, open(os.path.join(dest, "codes_std.json"), "w"), indent=1)
print(f"D={D}: codes std range [{Zs.min():.2f}, {Zs.max():.2f}], PCA s {np.round(sc, 3)} -> {dest}")
