#!/usr/bin/env python3
"""Build the installable Latent Kick instrument from a knob-space run (knobspace.py + warp.py):
dsp.lisp (gen_dsp with the A/B/blend/even front end, codes + warp compiled in), w1..b3.json,
decoder_layout.json, ui.lisp (scripts/ui_template.lisp + generated codes block), data/family.json
(peak envelopes for the map), instrument.json, and the .presets next to the folder.

    python3 build_instrument.py RUN DEST_DIR [--asset-path instruments/Research/Latent Kick/data/family.json]
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
KICK = MAD.stage("kick"); R = MAD.ROOT
import gen_dsp as G, kick_layout as L, kick_np as KN  # noqa: E402  (before e4: it imports another gen_dsp)
sys.path.append(MAD.code("e4"))
from build_family import peak_envelope, ENV_SECONDS, BEGIN, END  # noqa: E402
from train_np import load, forward  # noqa: E402
from kick_common import REPO  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("dest")
ap.add_argument("--asset-path", default="instruments/Research/Latent Kick/data/family.json")
ap.add_argument("--axes", default="1,2,3")
ap.add_argument("--blends", type=int, default=10)
a = ap.parse_args()
W, Z, ids = load(a.run)
fam = json.load(open(os.path.join(REPO, "content/instruments/Drums/VILLAIN Kick/data/family.json")))
names = [f"K{n}" for n in fam["kicks"]]
warp = json.load(open(os.path.join(a.run, "warp.json")))["table"]
os.makedirs(os.path.join(a.dest, "data"), exist_ok=True)

# dsp + weights + layout
G.BLEND = {"codes": Z.tolist(), "warp": warp, "names": names}
norms = G.norms_table()
open(os.path.join(a.dest, "dsp.lisp"), "w").write(G.gen_dsp(norms))
G.write_weights({"l1.W": W["W1"], "l1.b": W["b1"][None], "l2.W": W["W2"], "l2.b": W["b2"][None],
                 "l3.W": W["W3"], "l3.b": W["b3"][None]}, a.dest)
json.dump(L.layout_json({str(int(sr)): v for sr, v in norms.items()}), open(os.path.join(a.dest, "decoder_layout.json"), "w"), indent=1)
json.dump({"version": 1, "run_mode": "instrument"}, open(os.path.join(a.dest, "instrument.json"), "w"))
json.dump({"names": names, "codes": Z.tolist(), "run": os.path.relpath(os.path.abspath(a.run), MAD.ROOT)}, open(os.path.join(a.dest, "codes_std.json"), "w"))

# family map data + ui
bands = KN.noise_bands(48000)
rend = lambda z: KN.render(L.decode_u(forward(W, np.asarray(z)[None])[2][0]), bands=bands)["out"]
envs = [[round(float(v), 4) for v in peak_envelope(np.asarray(rend(Z[i]), float))] for i in range(len(Z))]
dims = [int(v) - 1 for v in a.axes.split(",")]
rest = [d for d in range(Z.shape[1]) if d not in dims[:2]]
famj = {"version": 1, "voice": "latent-kick", "axes": [f"z{d + 1}" for d in dims[:3]], "names": names,
        "scores": [[round(float(c[d]), 5) for d in dims[:3]] for c in Z], "codes": [[round(float(v), 5) for v in c] for c in Z],
        "dims": {"x": dims[0] + 1, "y": dims[1] + 1, "size": dims[2] + 1}, "lattice": {"members": envs, "pairs": {}}}
with open(os.path.join(a.dest, "data", "family.json"), "w") as f:
    json.dump(famj, f, separators=(",", ":")); f.write("\n")
rows = "\n".join(f'    ("{n}" {" ".join(f"{v:.5f}" for v in c)})' for n, c in zip(names, Z))
labels = " ".join(f'({i} "{n}")' for i, n in enumerate(names))
block = "\n".join([BEGIN, ";; Written by ModalKickAD/scripts/build_instrument.py; do not edit by hand.",
                   f'(def lk-family-file () "{a.asset_path}")',
                   f'(def lk-x-param () "z{dims[0] + 1}")', f'(def lk-y-param () "z{dims[1] + 1}")',
                   f"(def lk-x-dim () {dims[0] + 1})", f"(def lk-y-dim () {dims[1] + 1})", f"(def lk-size-dim () {dims[2] + 1})",
                   f"(def lk-knob-params () '({' '.join(chr(34) + 'z' + str(d + 1) + chr(34) for d in rest)}))",
                   f"(def lk-slot-labels () '({labels}))",
                   ";; (name z1 z2 z3 z4) in knob space, kick param order.", "(def lk-codes ()", "  '(", rows + "))", END])
text = open(os.path.join(MAD.code("kick"), "ui_template.lisp")).read()
text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
text = re.sub(r";; Kick param index -> name.*?\n\(def lk-slot-labels \(\)\n  \(map .*?\)\)\)\n", "", text, flags=re.S)
assert text.count("(def lk-slot-labels") == 1
open(os.path.join(a.dest, "ui.lisp"), "w").write(text)

# presets: 33 singles + the most novel even midpoints (each kick at most once)
from kick_metric import band_env_db  # noqa: E402
def dist(A, B):
    lo = B.max() - 70
    return float(np.abs(np.maximum(A, lo) - np.maximum(B, lo)).mean())
corpus = [band_env_db(rend(Z[i])) for i in range(len(Z))]
rng = np.random.default_rng(3)
cands = []
for i in range(len(Z)):
    for j in range(i + 1, len(Z)):
        if dist(corpus[i], corpus[j]) < 1.0:
            continue
        cands.append((i, j))
sel = [cands[k] for k in rng.choice(len(cands), min(160, len(cands)), replace=False)]
scored = []
for i, j in sel:
    tau = float(np.interp(0.5, np.linspace(0, 1, 11), warp[i][j]))
    y = rend((1 - tau) * Z[i] + tau * Z[j])
    pk = float(np.abs(y).max())
    if not 0.3 <= pk <= 1.2:
        continue
    e = band_env_db(y)
    scored.append((min(dist(e, c) for c in corpus), i, j))
scored.sort(reverse=True)
used, picks = set(), []
for nov, i, j in scored:
    if i in used or j in used:
        continue
    picks.append((i, j, nov)); used |= {i, j}
    if len(picks) == a.blends:
        break
base = {"blend": 0, "even": 1, "z1": 0, "z2": 0, "z3": 0, "z4": 0}
pre = [{"id": f"latent-kick-{n[1:]}", "name": f"Latent Kick {n[1:]}", "base_note_offset": 0,
        "params": {"kick_a": i, "kick_b": i, **base}} for i, n in enumerate(names)]
for i, j, nov in picks:
    pre.append({"id": f"latent-kick-{names[i][1:]}-x-{names[j][1:]}", "name": f"Latent Kick {names[i][1:]} x {names[j][1:]}",
                "base_note_offset": 0, "params": {"kick_a": i, "kick_b": j, **base, "blend": 0.5}})
    print(f"{names[i]} x {names[j]}: midpoint novelty {nov:.2f} dB")
dest_name = os.path.basename(a.dest.rstrip("/"))
doc = {"version": 1, "engine_name": f"Research/{dest_name}", "source_file": f"instruments/Research/{dest_name}/dsp.lisp", "presets": pre}
json.dump(doc, open(os.path.join(os.path.dirname(a.dest.rstrip("/")), dest_name + ".presets"), "w"), indent=2)
print(f"built {a.dest}: {len(pre)} presets, map axes {famj['axes']}")
