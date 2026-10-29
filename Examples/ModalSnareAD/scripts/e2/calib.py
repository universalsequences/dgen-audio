"""Calibration controls for every dataset render (E1's floor/nearest, extended to all 156):
floor = VILLAIN re-hit (voice memory carried over: head identical, noise advanced);
nearest = min over the 32 fit snares at the same velocity (excluding self) for fits,
over the 32 v100 fits for blends / pc points. -> calib.json, floor/<id>.hit2.wav"""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("e1"); MAD.use("dataset")
from metric import score, band_score
from villain_common import load, read_wav_f32, write_wav_f32, C4
man = json.load(open(os.path.join(R, "data/villain/manifest.json")))["items"]
V = os.path.join(R, "data/villain"); os.makedirs(os.path.join(HERE, "floor"), exist_ok=True)
inst = load(stems=False)
T = {it["id"]: read_wav_f32(os.path.join(V, it["file"])) for it in man}
out = {}
for it in man:
    rid = it["id"]; p = dict(it["params"]); vel = it["vel"]
    fp = os.path.join(HERE, "floor", rid + ".hit2.wav")
    if not os.path.exists(fp):
        y1, mem = inst.render(1.0, pitch=C4, vel=vel, params=p)
        y2, _ = inst.render(1.0, pitch=C4, vel=vel, params=p, mem=mem)
        y1 = y1 if y1.ndim == 1 else y1[:, 0]; y2 = y2 if y2.ndim == 1 else y2[:, 0]
        assert np.max(np.abs(y1 - T[rid])) < 1e-4, (rid, np.max(np.abs(y1 - T[rid])))
        write_wav_f32(fp, y2)
    h2 = read_wav_f32(fp); t = T[rid]
    vt = f"v{int(round(vel * 100)):03d}" if it["kind"] == "fit" else "v100"
    cands = [f"snare_{i:02d}_{vt}" for i in range(32) if f"snare_{i:02d}_{vt}" != rid]
    mr = {c: score(T[c], t) for c in cands}; bd = {c: band_score(T[c], t) for c in cands}
    out[rid] = {"kind": it["kind"], "vel": vel, "mr_floor": score(h2, t), "band_floor": band_score(h2, t),
                "mr_nearest": min(mr.values()), "mr_nearest_id": min(mr, key=mr.get),
                "band_nearest": min(bd.values()), "band_nearest_id": min(bd, key=bd.get),
                "mr_silence": score(np.zeros_like(t), t), "band_silence": band_score(np.zeros_like(t), t)}
    if it["kind"] == "blend":
        a, b = it["params"]["snare_a"], it["params"]["snare_b"]
        for tag, s in (("a", a), ("b", b)):
            out[rid][f"mr_end_{tag}"] = score(T[f"snare_{s:02d}_v100"], t); out[rid][f"band_end_{tag}"] = band_score(T[f"snare_{s:02d}_v100"], t)
    print(rid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[rid].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "calib.json"), "w"), indent=1)
