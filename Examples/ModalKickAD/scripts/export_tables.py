#!/usr/bin/env python3
"""Fixed signals the dgen trainer reads (exactly what the instrument computes on a fresh voice):
  kick/data/tables/bands.f32  time-major [48000, 18] unit-RMS band noise (seed-1 xorshift white)
  kick/data/tables/hum.f32    [48000] VILLAIN hum noise (dgen 12 Hz LP of the same white)
  kick/data/tables/tables.json shapes + doc"""
import json, os
import numpy as np
from kick_common import DATA
import kick_np as KN, kick_layout as L, villain_kick_np as V
d = os.path.join(DATA, "tables"); os.makedirs(d, exist_ok=True)
n = 48000
u = V.white_noise(n)
KN.noise_bands(n, u).astype("<f4").tofile(os.path.join(d, "bands.f32"))
KN.hum_noise(n, u).astype("<f4").tofile(os.path.join(d, "hum.f32"))
json.dump({"frames": n, "bands": L.NB, "fc": L.BAND_FC, "bands_layout": "time-major [frames, bands] float32 LE",
           "hum_layout": "[frames] float32 LE", "doc": __doc__}, open(os.path.join(d, "tables.json"), "w"), indent=1)
print("ok", d)
