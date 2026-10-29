#!/usr/bin/env python3
"""Latent A/B blend (straight z line through the warp) vs VILLAIN Kick's own blend renders
(manifest blend_*), band metric; reference: VILLAIN blend vs its nearer endpoint kick."""
import json, os, sys
import numpy as np
from kick_common import DATA, read_wav_f32
from kick_metric import band_score
import kick_np as KN, kick_layout as L
from train_np import load, forward
run = sys.argv[1]
W, Z, ids = load(run); warp = np.array(json.load(open(os.path.join(run, "warp.json")))["table"])
man = json.load(open(os.path.join(DATA, "villain", "manifest.json")))["items"]
bands = KN.noise_bands(48000)
rend = lambda z: KN.render(L.decode_u(forward(W, z[None])[2][0]), bands=bands)["out"]
res = []
for it in man:
    if it["kind"] != "blend": continue
    a, b, t = it["params"]["kick_a"], it["params"]["kick_b"], it["params"]["blend"]
    tgt = read_wav_f32(os.path.join(DATA, "villain", it["file"]))
    ya = read_wav_f32(os.path.join(DATA, "villain", f"kick_{a:02d}_v100.wav")); yb = read_wav_f32(os.path.join(DATA, "villain", f"kick_{b:02d}_v100.wav"))
    near = min(band_score(ya, tgt), band_score(yb, tgt))
    tau = np.interp(t, np.linspace(0, 1, 11), warp[a, b])
    lw = band_score(rend((1 - tau) * Z[a] + tau * Z[b]), tgt)
    ls = band_score(rend((1 - t) * Z[a] + t * Z[b]), tgt)
    res.append((lw, ls, near))
r = np.array(res)
print("latent blend vs VILLAIN blend: warped median %.3f dB, straight %.3f; VILLAIN blend vs nearer endpoint %.3f; norm (warped/near) median %.2f, %d/%d below 1"
      % (np.median(r[:, 0]), np.median(r[:, 1]), np.median(r[:, 2]), np.median(r[:, 0] / r[:, 2]), int(np.sum(r[:, 0] < r[:, 2])), len(r)))
json.dump({"rows": r.tolist()}, open(os.path.join(run, "blend_eval.json"), "w"))
