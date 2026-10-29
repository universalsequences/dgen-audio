#!/usr/bin/env python3
"""Template (kick_np, teacher rows) vs VILLAIN Kick renders: modal stem exact; full output on
waveform / band / MR, for mode=full (instrument) and mode=trainer (no IIR stages).
Also checks the decoder-space round trip (clamp_to_layout)."""
import json, os, sys
import numpy as np
from kick_common import DATA, read_wav_f32
from kick_metric import band_score, mr, relrms
import kick_np as KN
import kick_layout as L

man = {it["id"]: it for it in json.load(open(os.path.join(DATA, "villain", "manifest.json")))["items"]}
T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
abl = json.load(open(os.path.join(DATA, "ablations.json")))
near_b = abl["nearest_other_kick"]["band_per_kick"]; near_m = abl["nearest_other_kick"]["mr_per_kick"]
bands = KN.noise_bands(48000)
sel = sys.argv[1:]
rows = []
for k in range(33):
    for v in ("100", "070", "040"):
        rid = f"kick_{k:02d}_v{v}"
        if sel and rid not in sel and f"kick_{k:02d}" not in sel:
            continue
        it = man[rid]
        P = L.decode_u(np.array(T[f"kick_{k:02d}_v100"]["u"]))
        tgt = read_wav_f32(os.path.join(DATA, "villain", it["file"]))
        tm = read_wav_f32(os.path.join(DATA, "villain", it["stems"]["modal"]))
        r = KN.render(P, vel=it["vel"], bands=bands)
        rt = KN.render(P, vel=it["vel"], bands=bands, mode="trainer")
        row = {"id": rid, "modal": relrms(r["modal"], tm), "out_rel": relrms(r["out"], tgt),
               "out_band": band_score(r["out"], tgt), "out_mr": mr(r["out"], tgt),
               "tr_rel": relrms(rt["out"], tgt), "tr_band": band_score(rt["out"], tgt), "tr_mr": mr(rt["out"], tgt),
               "near_band": near_b[k], "near_mr": near_m[k]}
        rows.append(row)
        print(" ".join(f"{a}={b:.4f}" if isinstance(b, float) else str(b) for a, b in row.items()), flush=True)
if len(rows) > 1:
    for v in ("100", "070", "040"):
        rs = [r for r in rows if r["id"].endswith(v)]
        if not rs: continue
        print(f"v{v}: " + " ".join(f"{key} med {np.median([r[key] for r in rs]):.4f} max {np.max([r[key] for r in rs]):.4f}"
                                  for key in ("modal", "out_rel", "out_band", "out_mr", "tr_rel", "tr_band", "tr_mr")))
    json.dump(rows, open(os.path.join(DATA, "parity_template.json"), "w"), indent=1)
