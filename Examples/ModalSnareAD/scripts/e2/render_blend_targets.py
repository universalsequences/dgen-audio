"""VILLAIN renders for 96 blend-line training rows (non-held-out pairs, v1.0): 48 at t=0.5,
24 at 0.25, 24 at 0.75 -> blend_targets/<row id>.wav + blend_targets/rows.json"""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("dataset")
from villain_common import load, write_wav_f32, C4
tb = json.load(open(os.path.join(HERE, "targets_blend.json")))
rng = np.random.default_rng(20261028)
pairs = sorted({(v["a"], v["b"]) for v in tb.values()}); rng.shuffle(pairs)
rows = [f"bl_{a:02d}_{b:02d}_t050_v100" for a, b in pairs[:48]] + \
       [f"bl_{a:02d}_{b:02d}_t025_v100" for a, b in pairs[48:72]] + [f"bl_{a:02d}_{b:02d}_t075_v100" for a, b in pairs[72:96]]
od = os.path.join(HERE, "blend_targets"); os.makedirs(od, exist_ok=True)
V = load(stems=False)
for r in rows:
    t = tb[r]; y, _ = V.render(1.0, pitch=C4, vel=1.0, params={"snare_a": t["a"], "snare_b": t["b"], "blend": t["t"]})
    write_wav_f32(os.path.join(od, r + ".wav"), y if y.ndim == 1 else y[:, 0])
json.dump(rows, open(os.path.join(od, "rows.json"), "w"))
print(len(rows), "blend targets")
