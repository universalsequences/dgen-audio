#!/usr/bin/env python3
"""Presets for the E6 Latent Snare: 32 single snares (A = B, blend 0) + N A/B blends at blend 0.5 picked by an
ear proxy over all 496 walks (walks_after.json): no degeneracy flag on the walk's middle, distinct endpoints
(chord band > median), and the most NOVEL warped midpoint (largest band distance to every corpus snare), each
snare used once. usage: write_presets.py RUN WARP.json WALKS.json DEST_PRESETS [N]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from warp import tau_of
run, wp, walks, dest = sys.argv[1:5]; N = int(sys.argv[5]) if len(sys.argv) > 5 else 10
W, Z, names = C.load_run(run); T = np.array(json.load(open(wp))["table"])
rows = json.load(open(walks))["rows"]
corpus = [C.band_env(C.render(W, Z[i])[1]) for i in range(32)]
chord_med = np.median([r["chord"]["band"] for r in rows])
cands = []
for r in rows:
    if r["chord"]["band"] < chord_med: continue
    if any(0.3 <= t <= 0.7 for t, _ in r["flags"]): continue
    a, b = r["a"], r["b"]; tau = tau_of(T[a, b], 0.5)
    p, y, h, n = C.render(W, (1 - tau) * Z[a] + tau * Z[b])
    pk = float(np.abs(y).max())
    if not 0.25 <= pk <= 1.3: continue
    e = C.band_env(y); nov = min(C.band_from_env(e, c) for c in corpus)
    cands.append((nov, a, b, pk))
cands.sort(reverse=True); pick = []
for cap in (1, 2):  # each snare once, then (if short of N) at most twice
    used = {}
    for a_, b_, *_ in pick: used[a_] = used.get(a_, 0) + 1; used[b_] = used.get(b_, 0) + 1
    for nov, a, b, pk in cands:
        if len(pick) == N: break
        if any((a, b) == (x, y) for x, y, *_ in pick) or used.get(a, 0) >= cap or used.get(b, 0) >= cap: continue
        pick.append((a, b, nov, pk)); used[a] = used.get(a, 0) + 1; used[b] = used.get(b, 0) + 1
pre = []
for i, n in enumerate(names):
    pre.append({"id": f"latent-snare-{n.lower()}", "name": f"Latent {n}", "base_note_offset": 0,
                "params": {"snare_a": i, "snare_b": i, "blend": 0, "even": 1, "z1": 0, "z2": 0, "z3": 0, "z4": 0, "tune": 0, "decay": 1, "level": 0.8}})
for a, b, nov, pk in pick:
    pre.append({"id": f"latent-snare-{names[a].lower()}-x-{names[b].lower()}", "name": f"Latent {names[a]} x {names[b]}", "base_note_offset": 0,
                "params": {"snare_a": a, "snare_b": b, "blend": 0.5, "even": 1, "z1": 0, "z2": 0, "z3": 0, "z4": 0, "tune": 0, "decay": 1, "level": 0.8}})
    print(f"{names[a]} x {names[b]}: midpoint novelty {nov:.2f} dB (nearest corpus snare), peak {pk:.2f}")
doc = {"version": 1, "engine_name": "Research/Latent Snare", "source_file": "instruments/Research/Latent Snare/dsp.lisp", "presets": pre}
json.dump(doc, open(dest, "w"), indent=2)
print("wrote", dest, len(pre), "presets; novelty of corpus nearest-neighbours for scale:",
      round(float(np.median([min(C.band_from_env(corpus[i], corpus[j]) for j in range(32) if j != i) for i in range(32)])), 2))
