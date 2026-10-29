#!/usr/bin/env python3
"""Reconstruction of a trained run: decoder(code_k) through kick_np (mode=full, the
instrument) vs VILLAIN Kick renders, at v1.0/0.7/0.4, and vs the teacher row.
Normalised = band / nearest-other-kick band (data/ablations.json; 0 = VILLAIN itself).
    python3 eval_run.py runs/pre_d4 [--write-wavs]   (from $MODAL_AD_ROOT/kick)"""
import json, os, sys
import numpy as np
from kick_common import DATA, read_wav_f32, write_wav_f32
from kick_metric import band_score, mr, relrms
import kick_np as KN, kick_layout as L
from train_np import load, forward

run = sys.argv[1]
W, Z, ids = load(run)
_, _, U = forward(W, Z)
man = {it["id"]: it for it in json.load(open(os.path.join(DATA, "villain", "manifest.json")))["items"]}
T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
abl = json.load(open(os.path.join(DATA, "ablations.json")))["nearest_other_kick"]
bands = KN.noise_bands(48000)
rows = []
wdir = os.path.join(run, "wav")
if "--write-wavs" in sys.argv:
    os.makedirs(wdir, exist_ok=True)
for k, rid in enumerate(ids):
    P = L.decode_u(U[k]); Pt = L.decode_u(np.array(T[rid]["u"]))
    for v in ("100", "070", "040"):
        vid = rid.replace("v100", "v" + v)
        it = man[vid]
        tgt = read_wav_f32(os.path.join(DATA, "villain", it["file"]))
        tm = read_wav_f32(os.path.join(DATA, "villain", it["stems"]["modal"]))
        y = KN.render(P, vel=it["vel"], bands=bands)
        yt = KN.render(Pt, vel=it["vel"], bands=bands)
        b = band_score(y["out"], tgt)
        rows.append({"id": vid, "band": b, "mr": mr(y["out"], tgt), "norm": b / abl["band_per_kick"][k],
                     "teacher_band": band_score(yt["out"], tgt), "vs_teacher_band": band_score(y["out"], yt["out"]),
                     "modal_rel": relrms(y["modal"], tm), "peak": float(np.abs(y["out"]).max()),
                     "peak_villain": float(np.abs(tgt).max())})
        if "--write-wavs" in sys.argv and v == "100":
            write_wav_f32(os.path.join(wdir, f"{rid}.latent.wav"), y["out"])
            write_wav_f32(os.path.join(wdir, f"{rid}.villain.wav"), tgt)
summ = {}
for v in ("100", "070", "040"):
    rs = [r for r in rows if r["id"].endswith(v)]
    summ[v] = {key: float(np.median([r[key] for r in rs])) for key in ("band", "mr", "norm", "teacher_band", "vs_teacher_band", "modal_rel")}
    summ[v]["norm_max"] = float(np.max([r["norm"] for r in rs]))
    summ[v]["band_max"] = float(np.max([r["band"] for r in rs]))
    summ[v]["peak_ratio_med"] = float(np.median([r["peak"] / r["peak_villain"] for r in rs]))
    print(f"v{v}: " + " ".join(f"{a}={b:.4f}" for a, b in summ[v].items()))
worst = sorted(rows, key=lambda r: -r["norm"])[:5]
print("worst:", [(r["id"], round(r["band"], 2), round(r["norm"], 2), round(r["modal_rel"], 3)) for r in worst])
json.dump({"summary": summ, "rows": rows}, open(os.path.join(run, "eval.json"), "w"), indent=1)
