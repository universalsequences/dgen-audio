#!/usr/bin/env python3
"""K0: render the VILLAIN Kick dataset (48 kHz mono float32 WAV, 1.0 s, C4, gate held,
fresh voice state per render, trigger at sample 0) + stems + knob sweeps + manifest.
    python3 render_kick_dataset.py"""
import json, os
import numpy as np
from kick_common import load, render, write_wav_f32, TAPS, DATA, C4, SR, REPO

OUT = os.path.join(DATA, "villain")
STEMDIR = os.path.join(OUT, "stems")
os.makedirs(STEMDIR, exist_ok=True)
FAM = json.load(open(os.path.join(REPO, "content/instruments/Drums/VILLAIN Kick/data/family.json")))
NAMES = [f"Kick {k}" for k in FAM["kicks"]]
TEX_KNOBS = ["air", "room", "click", "floor", "crackle", "hum", "grit"]
SWEEP_KICKS = [0, 11, 29]
SWEEPS = {  # transform knob -> 3 non-default values inside the param range
    "tune": [0.5, 0.8, 1.6], "drop_time": [0.25, 0.5, 3.0], "tilt": [-1.5, -0.5, 1.5],
    "spread": [0.5, 0.8, 1.5], "beat": [0.0, 0.5, 3.0], "decay": [0.3, 0.6, 2.0],
    "bend": [0.0, 0.5, 2.0], "attack": [0.3, 0.6, 2.5], "body": [0.0, 0.5, 1.8],
    "knock": [0.0, 0.5, 2.5], "tick": [0.0, 0.5, 2.5], "shape": [0.0, 0.5, 2.5],
    "growl": [0.0, 0.5, 2.5], "level": [0.5, 1.5, 2.0], "drive": [2.0, 3.0, 5.0],
}
PITCHES = {"C3": C4 / 2, "G4": C4 * 2 ** (7 / 12), "C5": C4 * 2}

inst = load(stems=True)
rng = np.random.default_rng(20261028)
items = []

def add(rid, kind, params, vel=1.0, pitch=C4, stems=True, extra=None):
    r = render(inst, params, vel=vel, pitch=pitch)
    write_wav_f32(os.path.join(OUT, rid + ".wav"), r["out"])
    item = {"id": rid, "kind": kind, "file": rid + ".wav", "vel": vel, "pitch_hz": pitch, "params": params}
    if stems:
        item["stems"] = {}
        for ch in TAPS[1:]:
            write_wav_f32(os.path.join(STEMDIR, f"{rid}.{ch}.wav"), r[ch])
            item["stems"][ch] = f"stems/{rid}.{ch}.wav"
    if extra:
        item.update(extra)
    items.append(item)
    return r

base = lambda k: {"kick_a": k, "kick_b": k, "blend": 0.0, "drive": 1.0}
# 1. the 33 presets at 3 velocities (+ ablations at v1.0)
for k in range(33):
    for vel in (0.4, 0.7, 1.0):
        rid = f"kick_{k:02d}_v{int(vel*100):03d}"
        add(rid, "fit", base(k), vel=vel, extra={"name": NAMES[k]})
        if vel == 1.0:
            nt = render(inst, {**base(k), **{n: 0.0 for n in TEX_KNOBS}})["out"]
            write_wav_f32(os.path.join(STEMDIR, f"{rid}.out_notex.wav"), nt)
            cl = render(inst, {**base(k), "clip_amt": 0.0})["out"]
            write_wav_f32(os.path.join(STEMDIR, f"{rid}.out_noclip.wav"), cl)
            items[-1]["stems"]["out_notex"] = f"stems/{rid}.out_notex.wav"
            items[-1]["stems"]["out_noclip"] = f"stems/{rid}.out_noclip.wav"
# 2. transform-knob sweeps on 3 kicks (v1.0)
for k in SWEEP_KICKS:
    for knob, vals in SWEEPS.items():
        for v in vals:
            add(f"sweep_{k:02d}_{knob}_{v:g}", "sweep", {**base(k), knob: v}, extra={"knob": knob, "value": v})
    for pn, hz in PITCHES.items():
        add(f"sweep_{k:02d}_pitch_{pn}", "sweep", base(k), pitch=hz, extra={"knob": "pitch", "value": hz})
# 3. random A/B blends (held-out style) and PCA macro points
for j in range(24):
    a, b = rng.choice(33, 2, replace=False)
    add(f"blend_{j:02d}", "blend", {"kick_a": int(a), "kick_b": int(b), "drive": 1.0,
                                     "blend": round(float(rng.uniform(0.15, 0.85)), 4)})
for j in range(12):
    k = int(rng.integers(33))
    pcs = [round(float(x), 4) for x in rng.uniform(-2, 2, 3)]
    add(f"pc_{j:02d}", "pc", {**base(k), "pc1": pcs[0], "pc2": pcs[1], "pc3": pcs[2]})

# determinism
det = {}
p = base(7)
y1 = render(inst, p)["out"]; y2 = render(inst, p)["out"]
det["same_params_fresh_state_bit_identical"] = bool(np.array_equal(y1, y2))
y3 = render(load(stems=True, max_frames=512), p)["out"]
det["block128_vs_block512_maxabs"] = float(np.abs(y1 - y3).max())
y4, mem = inst.render(1.0, pitch=C4, params=p)
y5, _ = inst.render(1.0, pitch=C4, params=p, mem=mem)
det["second_hit_same_voice_maxabs_diff_out"] = float(np.abs(y4[:, 0] - y5[:, 0]).max())
det["second_hit_same_voice_maxabs_diff_modal"] = float(np.abs(y4[:, 1] - y5[:, 1]).max())
det["second_hit_note"] = ("second hit plays in the other voice slot (A) and crossfades B out over 5 ms; "
                          "the white stream continues (one shared (noise) cell), so texture differs")
y6 = render(load(stems=False), p)["out"]
det["stem_copy_out_bit_identical_to_shipped"] = bool(np.array_equal(y1, y6))

manifest = {"sample_rate": SR, "seconds": 1.0, "pitch_hz_default": C4, "format": "WAV float32 mono",
            "gate": "held for the whole render (hold/release gates are time-based, not gate-based)",
            "state": "fresh voice memory per render; trigger at sample 0 (plays in voice slot B); "
                     "ONE (noise) xorshift32 cell, state 0 -> seed 1",
            "instrument": "content/instruments/Drums/VILLAIN Kick (stem-tap copy kick/villain_stems, output bit-identical)",
            "compiler": inst.compiler, "compiler_sha256": inst.compiler_sha256,
            "stems_doc": {"modal": "onset_ramp * (24 slots + 10 transient sines), pre-shaper",
                          "bank_all": "modal + growl + burst", "body_sig": "after body shaper, bark, 18 Hz body HP",
                          "source": "body + grit + pre-chain texture", "texture": "air+room+click+aux (noise layers)",
                          "limited": "post recording chain + drive split, pre level/fade/out gate",
                          "out_notex": "full output with air,room,click,floor,crackle,hum,grit = 0",
                          "out_noclip": "full output with clip_amt = 0"},
            "sweeps": SWEEPS, "sweep_kicks": SWEEP_KICKS, "pitches": PITCHES,
            "determinism": det, "names": NAMES, "items": items}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(len(items), "renders;", det)
