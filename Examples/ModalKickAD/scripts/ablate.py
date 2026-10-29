#!/usr/bin/env python3
"""Which VILLAIN Kick stages matter? Twin ablations on the 33 kicks at v1.0 (and nearest-other control)."""
import json, os
import numpy as np
from kick_common import DATA
from kick_table import BlockRate
import villain_kick_np as V
from kick_metric import band_score, mr, relrms

br = BlockRate(); W = V.white_noise(48000)
envs = [br.eval(kick_a=k, kick_b=k) for k in range(33)]
outs = [V.render_hit(e, white=W)["out"] for e in envs]

def zero(e, keys, val=0.0):
    e = dict(e)
    for k in keys: e[k] = val
    return e
S = range(24)
AB = [f"x{s}{j}_amp" for s in "AB" for j in range(5)]
ABL = {
    "no_transients": lambda e, k: (zero(e, AB), k),
    "no_growl": lambda e, k: (e, {**k, "growl": 0}),
    "no_burst": lambda e, k: (e, {**k, "tick": 0}) if False else (zero(e, ["c_bu_amp"]), k),
    "no_texture": lambda e, k: (e, {**k, **{n: 0 for n in ("air", "room", "click", "floor", "crackle", "hum")}}),
    "no_grit": lambda e, k: (e, {**k, "grit": 0}),
    "no_shaper": lambda e, k: (zero(zero(e, [f"b{i}_MSM" for i in S] + [f"x{s}{j}_sm" for s in "AB" for j in range(5)] + ["c_bu_sm"]), ["c_bk_amt"]), k),
    "no_bark": lambda e, k: (zero(e, ["c_bk_amt"]), k),
    "no_bodyHP": lambda e, k: (zero(e, ["c_bhp_on"]), k),
    "no_clipper": lambda e, k: (e, {**k, "clip_amt": 0}),
    "no_dcblock": lambda e, k: (zero(e, ["c_dc_hz"]), k),
    "no_recLP": lambda e, k: (zero(e, ["c_rl_mix"]), k),
    "no_gates": lambda e, k: (zero(e, [f"b{i}_MG{g}" for i in S for g in "AB"] + [f"x{s}{j}_g{g}" for s in "AB" for j in range(5) for g in "ab"]), k),
    "no_outgate_fade": lambda e, k: (zero(zero(e, ["c_oA"], 1.0), ["c_len"], 100.0), k),
    "no_lowcut_fade": lambda e, k: (zero(e, [f"b{i}_MLC" for i in S]), k),
    "no_timewarp": lambda e, k: (zero(e, ["c_wk"]), k),
    "no_2nd_drop": lambda e, k: (zero(e, [f"b{i}_g2" for i in range(4)]), k),
    "no_drop": lambda e, k: (e, {**k, "bend": 0}),
    "no_onset_ramp": lambda e, k: (zero(zero(e, ["c_on_del"], -1.0), ["c_on_s"], 1e-6), k),
    "no_phase": lambda e, k: (zero(e, [f"b{i}_ph" for i in S]), k),
}
res = {}
for name, fn in ABL.items():
    r, b, m = [], [], []
    for e, o in zip(envs, outs):
        e2, k2 = fn(e, {})
        y = V.render_hit(e2, knobs=k2, white=W)["out"]
        r.append(relrms(y, o)); b.append(band_score(y, o)); m.append(mr(y, o))
    res[name] = {"rel_med": float(np.median(r)), "rel_max": float(np.max(r)), "band_med": float(np.median(b)),
                 "band_max": float(np.max(b)), "mr_med": float(np.median(m)), "worst": int(np.argmax(b))}
    print(f"{name:16s} rel med {np.median(r):.3f} max {np.max(r):.3f} | band dB med {np.median(b):.3f} max {np.max(b):.3f} (k{np.argmax(b)}) | MR med {np.median(m):.3f}", flush=True)
# control: nearest other kick
nb, nm = [], []
for i, o in enumerate(outs):
    bs = [band_score(outs[j], o) for j in range(33) if j != i]
    ms = [mr(outs[j], o) for j in range(33) if j != i]
    nb.append(min(bs)); nm.append(min(ms))
res["nearest_other_kick"] = {"band_med": float(np.median(nb)), "mr_med": float(np.median(nm)),
                             "band_per_kick": nb, "mr_per_kick": nm}
print(f"nearest other kick: band dB med {np.median(nb):.3f} min {np.min(nb):.3f} | MR med {np.median(nm):.3f}")
json.dump(res, open(os.path.join(DATA, "ablations.json"), "w"), indent=1)
