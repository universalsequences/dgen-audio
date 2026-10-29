"""Metric floor from noise realisation alone: render each fit snare (v100) a second time with the
voice memory carried over (all xorshift noise states advanced, head identical), score vs the
dataset render. Writes noise_floor.json and e1/noise_floor/<id>.hit2.wav."""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1"); MAD.use("e1"); MAD.use("dataset")
from villain_common import load, read_wav_f32, write_wav_f32, C4
from metric import score
inst = load(stems=False)
os.makedirs(os.path.join(HERE, "noise_floor"), exist_ok=True)
rows = []
for i in range(32):
    rid = f"snare_{i:02d}_v100"; p = {"snare_a": i, "snare_b": i}
    y1, mem = inst.render(1.0, pitch=C4, vel=1.0, params=p)
    y2, mem = inst.render(1.0, pitch=C4, vel=1.0, params=p, mem=mem)
    y3, _ = inst.render(1.0, pitch=C4, vel=1.0, params=p, mem=mem)
    ref = read_wav_f32(os.path.join(HERE, "..", "data/villain", rid + ".wav"))
    y1 = y1 if y1.ndim == 1 else y1[:, 0]; y2 = y2 if y2.ndim == 1 else y2[:, 0]; y3 = y3 if y3.ndim == 1 else y3[:, 0]
    write_wav_f32(os.path.join(HERE, "noise_floor", rid + ".hit2.wav"), y2)
    r = {"id": rid, "hit1_vs_ref": score(y1, ref), "hit2_vs_ref": score(y2, ref), "hit3_vs_ref": score(y3, ref)}
    rows.append(r); print(rid, {k: round(v, 4) for k, v in r.items() if k != "id"})
print("mean hit2", np.mean([r["hit2_vs_ref"] for r in rows]), "mean hit3", np.mean([r["hit3_vs_ref"] for r in rows]))
json.dump(rows, open(os.path.join(HERE, "noise_floor.json"), "w"), indent=1)
