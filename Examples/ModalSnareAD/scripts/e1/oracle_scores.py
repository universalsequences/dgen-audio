"""Oracle/ceiling diagnostics with the selection metric, per fit snare (v100):
 - ho+no:  out_headonly + out_noiseonly vs out   (linear superposition of VILLAIN's own parts)
 - clean:  out_clean vs out                      (SP sampler + tanh drive removed)
 - fit:    e1 fit render vs out (from a run dir)"""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1")
MAD.use("e1"); MAD.use("dataset")
from metric import score
from villain_common import read_wav_f32
V = os.path.join(HERE, "..", "data/villain")
run = sys.argv[1] if len(sys.argv) > 1 else None
rows = []
for i in range(32):
    rid = f"snare_{i:02d}_v100"
    out = read_wav_f32(f"{V}/{rid}.wav"); ho = read_wav_f32(f"{V}/stems/{rid}.out_headonly.wav")
    no = read_wav_f32(f"{V}/stems/{rid}.out_noiseonly.wav"); cl = read_wav_f32(f"{V}/stems/{rid}.out_clean.wav")
    r = {"id": rid, "ho+no": score(ho + no, out), "clean": score(cl, out), "headonly": score(ho, out)}
    if run: r["fit"] = score(read_wav_f32(os.path.join(run, rid + ".fit.wav")), out)
    rows.append(r); print(rid, {k: round(v, 3) for k, v in r.items() if k != "id"})
for k in rows[0]:
    if k != "id": print("mean", k, np.mean([r[k] for r in rows]))
json.dump(rows, open(os.path.join(HERE, "oracle_scores.json"), "w"), indent=1)
