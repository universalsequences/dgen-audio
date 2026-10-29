#!/usr/bin/env python3
"""Parity of the numpy twin vs the VILLAIN Kick stems for a set of renders."""
import json, os, sys, time
import numpy as np
from kick_common import DATA, read_wav_f32, relrms, C4
from kick_table import BlockRate
from villain_kick_np import render_hit, white_noise, KNOB_DEFAULTS

man = json.load(open(os.path.join(DATA, "villain", "manifest.json")))
br = BlockRate()
W = white_noise(48000)
ids = sys.argv[1:] or [it["id"] for it in man["items"]]
rows = []
for it in man["items"]:
    if it["id"] not in ids:
        continue
    p = it["params"]
    table_p = {k: v for k, v in p.items() if k not in KNOB_DEFAULTS or k in ("drop_time", "tilt", "spread", "beat")}
    knobs = {k: v for k, v in p.items() if k in KNOB_DEFAULTS}
    e = br.eval(**p)
    t0 = time.time()
    r = render_hit(e, vel=it["vel"], pitch=it["pitch_hz"], knobs=knobs, white=W)
    dt = time.time() - t0
    errs = {}
    for tap in ("modal", "bank_all", "body_sig", "source", "texture", "limited"):
        errs[tap] = relrms(r[tap], read_wav_f32(os.path.join(DATA, "villain", it["stems"][tap])))
    errs["out"] = relrms(r["out"], read_wav_f32(os.path.join(DATA, "villain", it["file"])))
    rows.append((it["id"], errs))
    print(it["id"], " ".join(f"{k}={v:.2e}" for k, v in errs.items()), f"({dt:.2f}s)", flush=True)
if len(rows) > 1:
    for tap in rows[0][1]:
        v = np.array([r[1][tap] for r in rows])
        print(f"{tap:9s} median {np.median(v):.2e} p90 {np.percentile(v, 90):.2e} max {v.max():.2e} ({rows[int(v.argmax())][0]})")
