#!/usr/bin/env python3
"""K0: exact per-render block-rate table -> kick/data/kick_modes.json, + kick_stats.json.
Every value is what VILLAIN Kick computes at block rate for that render (kick_table.BlockRate);
villain_kick_np.render_hit(env, ...) resynthesises every stem from `env`."""
import json, os
import numpy as np
from kick_common import DATA, read_wav_f32, relrms
from kick_table import BlockRate
from villain_kick_np import render_hit, white_noise, KNOB_DEFAULTS, modal_bank

man = json.load(open(os.path.join(DATA, "villain", "manifest.json")))
br = BlockRate()
SLOT = ["f", "amp", "idec", "tau", "itau", "g", "g2", "tau2", "itau2", "irise", "t0", "ph", "MGA", "MGB", "MSM", "MLC"]
XS = ["f", "idec", "dl", "itau", "tau", "irise", "rd", "amp", "ph", "t0", "te", "wb", "wk", "wt", "wc", "ga", "gb", "sm"]

def needed_env(e):
    keep = {}
    for k, v in e.items():
        if k.startswith("c_") or k in ("bl",) or (k[0] in "bx" and "_" in k and not k.endswith(("_m", "_l"))):
            if isinstance(v, float):
                keep[k] = v
    return keep

renders = {}
for it in man["items"]:
    e = br.eval(**it["params"])
    slots = []
    for i in range(24):
        s = {k: e[f"b{i}_{k}"] for k in SLOT if f"b{i}_{k}" in e}
        s["group"] = "knock" if i >= 14 else "body"
        s["drop_law"] = ("hz(t)=fmul*(f + sh*exp(-tw/tau)" + (" + sh2*exp(-tw/tau2)" if "g2" in s else "") +
                         "), sh=max(-0.9f, bendE*g*f)")
        slots.append(s)
    trans = {side: [{k: e[f"x{side}{j}_{k}"] for k in XS} for j in range(5)] for side in "AB"}
    renders[it["id"]] = {"params": it["params"], "vel": it["vel"], "pitch_hz": it["pitch_hz"],
                         "slots": slots, "transients": trans, "env": needed_env(e)}

doc = ("Per-render VILLAIN Kick block-rate values (ModalKickAD/scripts/kick_table.py evaluates dsp.lisp's top-level defs; "
       "identical to the per-hit latched copy). Modal slot i: t=max(0,T-t0), tw=max(0,Tw-t0), "
       "T=mix(age, age*tuning, c_samp), fmul=mix(tuning,1,c_samp), Tw=T+c_wk*bendE*c_wtau*(1-exp(-T/c_wtau)), "
       "sh=max(-0.9f, bendE*g*f) (+ sh2 with g2/tau2 on slots 0-3), "
       "phase=2pi*fmul*(f*tw + sh*tau*(1-exp(-tw/tau)) + sh2*tau2*(1-exp(-tw/tau2))) + ph, "
       "hz=fmul*(f+sh*e^(-tw/tau)+sh2*e^(-tw/tau2)), "
       "g=amp*grpE*exp(-t*idec/decE)*(1-exp(-t*irise/attE))*nyq(hz)*(1-MLC(1-clip((mix(hz,fmul f,c_lcb)-c_lcf)/c_lcw)))*gates, "
       "grpE = bodyE (slots 0-13) or knockE (14-23). tuning = 2*pitch/440*tune (1.18921 at C4). "
       "Transients x{A,B}j: own sines with absolute drop dl (Hz), envelope rise/decay, weight (1-bl)/bl. "
       "env = everything villain_kick_np.render_hit needs.")
json.dump({"doc": doc, "tuning_at_C4": 2 * 261.6255653 / 440, "renders": renders},
          open(os.path.join(DATA, "kick_modes.json"), "w"))

# ---- stats on the 33 fits (v1.0)
W = white_noise(48000)
def E(x): return float(np.sum(np.asarray(x, float) ** 2))
stats = []
for k in range(33):
    rid = f"kick_{k:02d}_v100"
    it = next(i for i in man["items"] if i["id"] == rid)
    e = br.eval(**it["params"])
    r = render_hit(e, white=W)
    P = r["_parts"]
    out = read_wav_f32(os.path.join(DATA, "villain", it["file"]))
    nt = read_wav_f32(os.path.join(DATA, "villain", it["stems"]["out_notex"]))
    nc = read_wav_f32(os.path.join(DATA, "villain", it["stems"]["out_noclip"]))
    # per-slot energy share in the modal stem
    T = np.arange(48000) / 48000.0
    slotE = []
    for i in range(24):
        b, _, _ = modal_bank(e, T * 1.0, T, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.ones(48000), np.ones(48000), 0.0, slots=[i])
        slotE.append(E(b))
    st = {"kick": k, "name": it["name"], "peak": float(np.abs(out).max()),
          "E_modal/E_bank_all": E(r["modal"]) / E(r["bank_all"]),
          "E_growl/E_bank_all": E(P["growl"]) / E(r["bank_all"]),
          "E_burst/E_bank_all": E(P["burst"]) / E(r["bank_all"]),
          "rel_body_vs_bank": relrms(r["body_sig"], r["bank_all"]),
          "E_texture/E_source": E(r["texture"]) / E(r["source"]),
          "E_grit/E_source": E(P["grit"]) / E(r["source"]),
          "rel(out_notex, out)": relrms(nt, out), "rel(out_noclip, out)": relrms(nc, out),
          "c_samp": e["c_samp"], "c_wk": e["c_wk"], "len_s": e["c_len"],
          "f_sub": e["b0_f"], "g_sub": e["b0_g"], "tau_sub": e["b0_tau"], "g2_sub": e["b0_g2"], "tau2_sub": e["b0_tau2"],
          "n_slots_gt1pct": int(np.sum(np.array(slotE) / max(sum(slotE), 1e-30) > 0.01)),
          "slot_share": [x / max(sum(slotE), 1e-30) for x in slotE],
          "trans_amp": [e[f"xA{j}_amp"] for j in range(5)]}
    stats.append(st)
json.dump(stats, open(os.path.join(DATA, "kick_stats.json"), "w"), indent=1)
keys = [k for k in stats[0] if isinstance(stats[0][k], float)]
for key in keys:
    v = np.array([s[key] for s in stats])
    print(f"{key:22s} median {np.median(v):9.4g}  min {v.min():9.4g}  max {v.max():9.4g}")
v = np.array([s["n_slots_gt1pct"] for s in stats]); print("slots >1% energy: median", np.median(v), "range", v.min(), v.max())
sh = np.array([s["slot_share"] for s in stats]); print("mean slot share:", np.round(sh.mean(0), 3))
