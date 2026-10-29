#!/usr/bin/env python3
"""Compiled Latent Kick (this checkout's DGenLisp, C backend) vs the numpy twin kick_np (mode=full).
(a) decoder pinned to a teacher row (W3 = 0, b3 = u): instrument vs twin vs VILLAIN, at 3
    velocities and through the transform-knob sweeps; (b) random weights at random z
    (checks the matmul decoder path).  Writes kick/data/instrument_parity.json"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
os.environ["ESEQ_DGENLISP_TOOL"] = MAD.dgenlisp()  # the Latent Kick needs this checkout's compiler
from kick_common import DATA, KICK, read_wav_f32, C4, Instrument
from kick_metric import band_score, mr, relrms
import kick_layout as L, kick_np as KN
import gen_dsp as G

WORK = os.path.join(KICK, "variants", "parity")
os.makedirs(WORK, exist_ok=True)
os.environ["LATENT_TAP"] = "1"
G.TAP = True
norms = G.norms_table()
open(os.path.join(WORK, "dsp.lisp"), "w").write(G.gen_dsp(norms))
T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
MAN = json.load(open(os.path.join(DATA, "villain", "manifest.json")))
man = {it["id"]: it for it in MAN["items"]}
bands = KN.noise_bands(48000)

def build(weights):
    G.write_weights(weights, WORK)
    return Instrument(WORK, sample_rate=48000.0, max_frames=128)

def inst_render(inst, z=(0, 0, 0, 0), vel=1.0, pitch=C4, knobs=None):
    p = {f"z{i + 1}": float(v) for i, v in enumerate(z)}
    p.update(knobs or {})
    y, _ = inst.render(1.0, pitch=pitch, vel=vel, params=p)
    return {"out": y[:, 0], "modal": y[:, 2], "noise": y[:, 3]}

res = {"teacher": [], "random": []}
D, H, P = L.D_IN, L.H, L.P
kicks = [int(a) for a in sys.argv[1:]] or [0, 3, 5, 11, 16, 21, 26, 29]
for k in kicks:
    u = np.array(T[f"kick_{k:02d}_v100"]["u"])
    W = {"l1.W": np.zeros((D, H)), "l1.b": np.zeros((1, H)), "l2.W": np.zeros((H, H)), "l2.b": np.zeros((1, H)),
         "l3.W": np.zeros((H, P)), "l3.b": u[None, :]}
    inst = build(W)
    Pd = L.decode_u(u)
    cases = [("v100", 1.0, None, C4), ("v070", 0.7, None, C4), ("v040", 0.4, None, C4)]
    if k in (0, 11, 29):
        for knob, vals in MAN["sweeps"].items():
            if knob in ("level", "drive", "tune", "drop_time", "tilt", "spread", "beat", "decay", "bend", "attack", "body", "knock", "tick", "shape", "growl"):
                for v in vals:
                    cases.append((f"sweep_{k:02d}_{knob}_{v:g}", 1.0, {knob: v}, C4))
        for pn in ("C3", "G4", "C5"):
            cases.append((f"sweep_{k:02d}_pitch_{pn}", 1.0, None, man[f"sweep_{k:02d}_pitch_{pn}"]["pitch_hz"]))
    for tag, vel, kn, pitch in cases:
        yi = inst_render(inst, vel=vel, pitch=pitch, knobs=kn)
        yt = KN.render(Pd, knobs=kn, vel=vel, pitch=pitch, bands=bands, taps=True)
        vid = f"kick_{k:02d}_{tag}" if tag.startswith("v") else tag
        tgt = read_wav_f32(os.path.join(DATA, "villain", man[vid]["file"]))
        row = {"id": vid, "inst_vs_twin_out": relrms(yi["out"], yt["out"]), "inst_vs_twin_modal": relrms(yi["modal"], yt["modal"]),
               "inst_vs_twin_noise": relrms(yi["noise"], yt["noise"]),
               "inst_vs_villain_band": band_score(yi["out"], tgt), "twin_vs_villain_band": band_score(yt["out"], tgt),
               "inst_vs_villain_rel": relrms(yi["out"], tgt), "twin_vs_villain_rel": relrms(yt["out"], tgt)}
        if tag.startswith("v"):
            tm = read_wav_f32(os.path.join(DATA, "villain", man[vid]["stems"]["modal"]))
            row["inst_vs_villain_modal"] = relrms(yi["modal"], tm)
        res["teacher"].append(row)
        print(" ".join(f"{a}={b:.2e}" if isinstance(b, float) else str(b) for a, b in row.items()), flush=True)

# (b) random weights
Wr = G.random_weights(11)
inst = build(Wr)
rng = np.random.default_rng(5)
for j in range(4):
    z = rng.normal(0, 1, D)
    h1 = np.tanh(z[None] @ Wr["l1.W"] + Wr["l1.b"]); h2 = np.tanh(h1 @ Wr["l2.W"] + Wr["l2.b"])
    u = (h2 @ Wr["l3.W"] + Wr["l3.b"])[0]
    yi = inst_render(inst, z=z)
    yt = KN.render(L.decode_u(u), bands=bands, taps=True)
    row = {"z": [float(x) for x in z], "inst_vs_twin_out": relrms(yi["out"], yt["out"]),
           "inst_vs_twin_modal": relrms(yi["modal"], yt["modal"]), "peak": float(np.abs(yi["out"]).max())}
    res["random"].append(row)
    print("random", row, flush=True)
json.dump(res, open(os.path.join(DATA, "instrument_parity.json"), "w"), indent=1)
t = res["teacher"]
for key in ("inst_vs_twin_out", "inst_vs_twin_modal", "inst_vs_twin_noise", "inst_vs_villain_band", "twin_vs_villain_band", "inst_vs_villain_modal"):
    v = [r[key] for r in t if key in r]
    print(f"{key:22s} median {np.median(v):.2e} max {np.max(v):.2e}")
