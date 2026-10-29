#!/usr/bin/env python3
"""E6 panel: ui_template.lisp + the generated codes block -> DEST/ui.lisp; DEST/data/family.json with each
snare's peak envelope rendered by the trained voice WITH the wire layer (e7lib twin), knob-space codes.
usage: build_panel.py RUN DEST [--asset-path instruments/Research/Latent Snare/data/family.json] [--axes 1,3,2]"""
import argparse, json, os, re, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e6"); R = MAD.ROOT
MAD.use("e6"); MAD.use("e4"); MAD.use("e7")
import common as C
from build_family import peak_envelope, ENV_SECONDS, BEGIN, END  # noqa: E402
import e7lib  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("dest")
ap.add_argument("--asset-path", default="instruments/Research/Latent Snare/data/family.json")
ap.add_argument("--axes", default="1,3,2")
a = ap.parse_args()
W, Z, names = C.load_run(a.run); mu, M, Zk = C.knob_transform(Z)
dims = [int(v) - 1 for v in a.axes.split(",")]; rest = [d for d in range(4) if d not in dims[:2]]
envs = []
for i in range(32):
    p = C.decode(W, Z[i], 1.0)
    knobs = {k: float(np.ravel(p[k])[0]) for k in e7lib.WIRE}
    y = e7lib.render(p, knobs, n=int((ENV_SECONDS + 0.01) * C.SR))
    envs.append([round(float(v), 4) for v in peak_envelope(np.asarray(y, float))])
fam = {"version": 1, "voice": "latent-snare", "axes": [f"z{d + 1}" for d in dims[:3]], "names": names,
       "scores": [[round(float(c[d]), 5) for d in dims[:3]] for c in Zk], "codes": [[round(float(v), 5) for v in c] for c in Zk],
       "dims": {"x": dims[0] + 1, "y": dims[1] + 1, "size": dims[2] + 1}, "lattice": {"members": envs, "pairs": {}}}
os.makedirs(os.path.join(a.dest, "data"), exist_ok=True)
with open(os.path.join(a.dest, "data", "family.json"), "w") as f:
    json.dump(fam, f, separators=(",", ":")); f.write("\n")
rows = "\n".join(f'    ("{n}" {" ".join(f"{v:.5f}" for v in c)})' for n, c in zip(names, Zk))
labels = " ".join(f'({i} "{n}")' for i, n in enumerate(names))
block = "\n".join([BEGIN, ";; Written by scripts/e6/build_panel.py; do not edit by hand.",
                   f'(def ls-family-file () "{a.asset_path}")',
                   f'(def ls-x-param () "z{dims[0] + 1}")', f'(def ls-y-param () "z{dims[1] + 1}")',
                   f"(def ls-x-dim () {dims[0] + 1})", f"(def ls-y-dim () {dims[1] + 1})", f"(def ls-size-dim () {dims[2] + 1})",
                   f"(def ls-knob-params () '({' '.join(chr(34) + 'z' + str(d + 1) + chr(34) for d in rest)}))",
                   f"(def ls-slot-labels () '({labels}))",
                   ";; (name z1 z2 z3 z4) in knob space, snare param order.", "(def ls-codes ()", "  '(", rows + "))", END])
text = open(os.path.join(MAD.code("e6"), "ui_template.lisp")).read()
text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
text = re.sub(r";; Snare param index -> name.*?\n\(def ls-slot-labels \(\)\n  \(map .*?\)\)\)\n", "", text, flags=re.S)
assert text.count("(def ls-slot-labels") == 1
open(os.path.join(a.dest, "ui.lisp"), "w").write(text)
print(f"{a.dest}: ui.lisp + data/family.json (axes {fam['axes']})")
