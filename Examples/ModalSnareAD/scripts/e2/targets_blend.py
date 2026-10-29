#!/usr/bin/env python3
"""Blend-line teacher rows: VILLAIN's own A/B blend (snare_a=A, snare_b=B, blend=t) in the
template's natural space, for every unordered pair NOT used by a held-out dataset blend,
t in {0.25, 0.5, 0.75}, velocities 0.4/0.7/1.0.
Head + chain constants: exact block-rate values (scripts/villain_table.py, the evaluator
that produced villain_modes.json). Noise + output stage: the endpoints' E1 fits mixed with
VILLAIN's blend weight t (log-mix for positive log-range params, linear otherwise); the
noise level at v<1 mixes the endpoints' scales. -> targets_blend.json
    python3 targets_blend.py"""
import json, math, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("e2"); MAD.use("dataset")
import targets as T
from villain_table import BlockRate
br = BlockRate()
man = json.load(open(os.path.join(R, "data/villain/manifest.json")))["items"]
held = {tuple(sorted((m["params"]["snare_a"], m["params"]["snare_b"]))) for m in man if m["kind"] == "blend"}
LIN = {"noise_tail_frac", "noise_onset_s", "drive_amt"}
def mixfit(fa, fb, t):
    out = {}
    for k, va in fa.items():
        if k in ("id",): continue
        vb = fb[k]
        a, b = np.asarray(va, float), np.asarray(vb, float)
        if k in LIN or k.startswith("head_"): out[k] = ((1 - t) * a + t * b).tolist()
        else: out[k] = np.exp((1 - t) * np.log(np.maximum(a, 1e-12)) + t * np.log(np.maximum(b, 1e-12))).tolist()
    for k in ("noise_onset_s", "noise_rise_s", "drive", "drive_amt", "out_gain"):
        out[k] = float(np.ravel(out[k])[0])
    return out
def main():
    nsc = {}
    for s in range(32):
        for vt in ("040", "070"):
            nsc[(s, vt)] = T.noise_rms(f"snare_{s:02d}_v{vt}") / max(T.noise_rms(f"snare_{s:02d}_v100"), 1e-9)
        nsc[(s, "100")] = 1.0
    out = {}
    for a in range(32):
        for b in range(a + 1, 32):
            if (a, b) in held: continue
            for t in (0.25, 0.5, 0.75):
                env = br.eval(snare_a=a, snare_b=b, blend=t)
                slots = [{"f": env[f"b{i}_f"], "amp": env[f"b{i}_amp"], "dec0": env[f"b{i}_dec0"], "g0": env[f"b{i}_g0"],
                          "tau": env[f"b{i}_tau"], "rise0": env[f"b{i}_rise0"], "t0": env[f"b{i}_t0"], "ph": env[f"b{i}_ph"],
                          "mh": env[f"b{i}_mh"], "md": env[f"b{i}_md"]} for i in range(32)]
                chain = {k: v for k, v in env.items() if k.startswith("c_")}
                glob = {k: env[k] for k in ("dv", "d_amt", "d_norm", "len_s", "fade_s", "lvl")}
                r = {"slots": slots, "chain": chain, "global": glob}
                fit = mixfit(T.e1[f"snare_{a:02d}_v100"], T.e1[f"snare_{b:02d}_v100"], t)
                for vt, vel in T.VELS.items():
                    fit["_nscale"] = math.exp((1 - t) * math.log(nsc[(a, vt)]) + t * math.log(nsc[(b, vt)]))
                    tt = T.target_from(r, vel, fit)
                    out[f"bl_{a:02d}_{b:02d}_t{int(t*100):03d}_v{vt}"] = {"a": a, "b": b, "t": t, "vel": vel,
                        "natural": {k: np.asarray(v).tolist() for k, v in tt.items()}}
    json.dump(out, open(os.path.join(HERE, "targets_blend.json"), "w"))
    print(len(out), "blend rows;", len(held), "held-out pairs excluded")


if __name__ == "__main__":
    main()
