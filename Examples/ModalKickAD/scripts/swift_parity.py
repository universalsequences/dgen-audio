#!/usr/bin/env python3
"""Swift trainer renderer (ModalKickAD parity, writes data/swift_parity/<id>.<tap>.f32) vs the
numpy twin kick_np (mode=trainer) and VILLAIN stems.  usage: swift_parity.py id1 id2 ..."""
import json, os, sys
import numpy as np
from kick_common import DATA, read_wav_f32
from kick_metric import relrms, band_score
import kick_np as KN, kick_layout as L

T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
man = {it["id"]: it for it in json.load(open(os.path.join(DATA, "villain", "manifest.json")))["items"]}
bands = KN.noise_bands(48000)
D = os.path.join(DATA, "swift_parity")
rows = []
for rid in sys.argv[1:]:
    it = man[rid]
    key = f"kick_{rid.split('_')[1]}_v100" if rid.startswith(("kick_", "sweep_")) else rid
    P = L.decode_u(np.array(T[key]["u"]))
    knobs = {k: v for k, v in it["params"].items() if k in KN.KNOB_DEF}
    yt = KN.render(P, knobs=knobs, vel=it["vel"], pitch=it["pitch_hz"], bands=bands, mode="trainer", taps=True)
    sw = {tap: np.fromfile(os.path.join(D, f"{rid}.{tap}.f32"), "<f4").astype(float) for tap in ("out", "modal", "noise")}
    row = {"id": rid, "out": relrms(sw["out"], yt["out"]), "modal": relrms(sw["modal"], yt["modal"]),
           "noise": relrms(sw["noise"], yt["noise"])}
    if "modal" in it.get("stems", {}):
        row["modal_vs_villain"] = relrms(sw["modal"], read_wav_f32(os.path.join(DATA, "villain", it["stems"]["modal"])))
    row["out_band_vs_villain"] = band_score(sw["out"], read_wav_f32(os.path.join(DATA, "villain", it["file"])))
    rows.append(row)
    print(" ".join(f"{a}={b:.2e}" if isinstance(b, float) else str(b) for a, b in row.items()))
for k in ("out", "modal", "noise", "modal_vs_villain"):
    v = [r[k] for r in rows if k in r]
    if v: print(f"{k:18s} median {np.median(v):.2e} max {np.max(v):.2e}")
json.dump(rows, open(os.path.join(DATA, "swift_parity.json"), "w"), indent=1)
