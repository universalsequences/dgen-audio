#!/usr/bin/env python3
"""Inputs for the dgen audio stages: texture+grit targets (VILLAIN twin, v1.0, exact noise)
-> data/tex_targets/<id>.f32, and the param-space anchor (column weights, base, teacher positions)
-> runs/<run>/anchor.json"""
import json, os, sys
import numpy as np
from kick_common import DATA
from kick_table import BlockRate
import villain_kick_np as V
from train_np import col_weights
run = sys.argv[1]
T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
ids = [f"kick_{k:02d}_v100" for k in range(33)]
d = os.path.join(DATA, "tex_targets"); os.makedirs(d, exist_ok=True)
br = BlockRate(); W = V.white_noise(48000)
for k, rid in enumerate(ids):
    r = V.render_hit(br.eval(kick_a=k, kick_b=k), white=W)
    (r["texture"] + r["_parts"]["grit"]).astype("<f4").tofile(os.path.join(d, rid + ".f32"))
U = np.array([T[i]["u"] for i in ids]); S = 1 / (1 + np.exp(-U))
w = col_weights([T[i]["P"] for i in ids])
base = float(np.mean((S - S.mean(0)) ** 2 * w ** 2))
json.dump({"ids": ids, "w": w.tolist(), "base": base, "S": S.reshape(-1).tolist()}, open(os.path.join(run, "anchor.json"), "w"))
print("ok")
