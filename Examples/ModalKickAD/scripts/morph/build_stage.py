#!/usr/bin/env python3
"""Stage + install Latent Kick Morph: dsp.lisp + weights (gen_morph.py), data/family.json (map
envelopes of the 33 kick codes rendered by the numpy twin, as ModalKickAD/scripts/build_instrument.py),
ui.lisp (ui.tmpl.lisp + the generated codes block), instrument.json, and the .presets beside it,
in $MODAL_AD_ROOT/kick/morph/Latent Kick Morph.

usage: build_stage.py [--hop 128] [--install DIR]   (DIR: an eseq instrument library folder, e.g.
<eseq user-instruments>/Research from `eseq paths`)"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("kick/morph"); KICK = MAD.stage("kick"); R = MAD.ROOT
MAD.use("kick")
import kick_layout as L, kick_np as KN  # noqa: E402
sys.path.append(MAD.code("e4"))
from build_family import peak_envelope, BEGIN, END  # noqa: E402
from train_np import load, forward  # noqa: E402

NAME = "Latent Kick Morph"
STAGE = os.path.join(HERE, NAME)
ASSET = f"instruments/Research/{NAME}/data/family.json"
ap = argparse.ArgumentParser(); ap.add_argument("--hop", type=int, default=128); ap.add_argument("--install", metavar="DIR")
a = ap.parse_args()

subprocess.run([sys.executable, os.path.join(MAD.code("kick/morph"), "gen_morph.py"), STAGE, "--hop", str(a.hop)], check=True)
W, Z, ids = load(os.path.join(KICK, "runs", "final_d4"))
names = json.load(open(os.path.join(STAGE, "codes_std.json")))["names"]
assert len(names) == len(Z)

# map data: same envelopes / layout as the Latent Kick panel (x = z1, y = z2, size = z3)
fam = os.path.join(STAGE, "data", "family.json")
os.makedirs(os.path.dirname(fam), exist_ok=True)
bands = KN.noise_bands(48000)
rend = lambda z: KN.render(L.decode_u(forward(W, np.asarray(z)[None])[2][0]), bands=bands)["out"]
envs = [[round(float(v), 4) for v in peak_envelope(np.asarray(rend(Z[i]), float))] for i in range(len(Z))]
dims = [0, 1, 2]
famj = {"version": 1, "voice": "latent-kick-morph", "axes": [f"z{d + 1}" for d in dims], "names": names,
        "scores": [[round(float(c[d]), 5) for d in dims] for c in Z], "codes": [[round(float(v), 5) for v in c] for c in Z],
        "dims": {"x": 1, "y": 2, "size": 3}, "lattice": {"members": envs, "pairs": {}}}
with open(fam, "w") as f:
    json.dump(famj, f, separators=(",", ":")); f.write("\n")

rows = "\n".join(f'    ("{n}" {" ".join(f"{v:.5f}" for v in c)})' for n, c in zip(names, Z))
labels = " ".join(f'({i} "{n}")' for i, n in enumerate(names))
block = "\n".join([BEGIN, ";; Written by dgen Examples/ModalKickAD/scripts/morph/build_stage.py; do not edit by hand.",
                   f'(def lkm-family-file () "{ASSET}")',
                   "(def lkm-x-dim () 1)", "(def lkm-y-dim () 2)", "(def lkm-size-dim () 3)",
                   f"(def lkm-slot-labels () '({labels}))",
                   ";; (name z1 z2 z3 z4) in knob space, kick param order.", "(def lkm-codes ()", "  '(", rows + "))", END])
ui = open(os.path.join(MAD.code("kick/morph"), "ui.tmpl.lisp")).read().replace("@@CODES@@", block)
open(os.path.join(STAGE, "ui.lisp"), "w").write(ui)
open(os.path.join(STAGE, "instrument.json"), "w").write('{"version":1,"run_mode":"instrument"}\n')
print("staged", STAGE)
if a.install:
    DEST = os.path.join(a.install, NAME)
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    shutil.copytree(STAGE, DEST)
    pres = os.path.join(HERE, f"{NAME}.presets")
    if os.path.exists(pres):
        shutil.copy(pres, os.path.dirname(DEST))
    print("installed", DEST)
