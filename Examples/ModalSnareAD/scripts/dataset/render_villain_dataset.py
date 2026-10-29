#!/usr/bin/env python3
"""Render the VILLAIN Snare dataset (48 kHz mono float32 WAV, 1.0 s, C4, gate held,
fresh voice state per render) + manifest.json + villain_modes.json.
Writes $MODAL_AD_ROOT/data/villain/ and $MODAL_AD_ROOT/data/villain_modes.json; needs the stem-tap
copy from make_stems_copy.py first."""
import json, os
import numpy as np
from villain_common import (load, render, write_wav_f32, head_resynth, NOISE_KNOBS,
                            RESEARCH, REPO, C4, SR)
from villain_table import BlockRate

OUT = os.path.join(RESEARCH, "data", "villain")
STEMDIR = os.path.join(OUT, "stems")
os.makedirs(STEMDIR, exist_ok=True)
NAMES = json.load(open(os.path.join(REPO, "content/instruments/Drums/VILLAIN Snare/data/family.json")))["names"]
PRESETS = {p["params"]["snare_a"]: p for p in json.load(open(os.path.join(REPO, "content/instruments/Drums/VILLAIN Snare.presets")))["presets"]}

br = BlockRate()
inst = load(stems=True)
rng = np.random.default_rng(20261027)
items = []

def add(rid, kind, params, vel=1.0, stems=False, knob_stems=False):
    r = render(inst, params, vel=vel)
    write_wav_f32(os.path.join(OUT, rid + ".wav"), r["out"])
    item = {"id": rid, "kind": kind, "file": rid + ".wav", "vel": vel, "params": params}
    if stems:
        for ch in ("head", "source", "strands"):
            write_wav_f32(os.path.join(STEMDIR, f"{rid}.{ch}.wav"), r[ch])
        item["stems"] = {ch: f"stems/{rid}.{ch}.wav" for ch in ("head", "source", "strands")}
    if knob_stems:
        ho = render(inst, {**params, **{k: 0.0 for k in NOISE_KNOBS}}, vel=vel)["out"]
        no = render(inst, {**params, "body": 0.0, "ring": 0.0}, vel=vel)["out"]
        write_wav_f32(os.path.join(STEMDIR, f"{rid}.out_headonly.wav"), ho)
        write_wav_f32(os.path.join(STEMDIR, f"{rid}.out_noiseonly.wav"), no)
        item["stems"]["out_headonly"] = f"stems/{rid}.out_headonly.wav"
        item["stems"]["out_noiseonly"] = f"stems/{rid}.out_noiseonly.wav"
        cl = render(inst, {**params, "grit": 0.0, "drive": 0.5}, vel=vel)["out"]
        write_wav_f32(os.path.join(STEMDIR, f"{rid}.out_clean.wav"), cl)
        item["stems"]["out_clean"] = f"stems/{rid}.out_clean.wav"
    items.append(item)
    return r

# 1. the 32 fits at 3 velocities
for k in range(32):
    for vel in (0.4, 0.7, 1.0):
        add(f"snare_{k:02d}_v{int(vel*100):03d}", "fit", {"snare_a": k, "snare_b": k, "blend": 0.0},
            vel=vel, stems=True, knob_stems=(vel == 1.0))
        items[-1]["name"] = NAMES[k]
        items[-1]["preset_release_ms"] = PRESETS[k]["params"].get("release")
# 2. random A/B blends
for j in range(40):
    a, b = rng.choice(32, 2, replace=False)
    add(f"blend_{j:02d}", "blend", {"snare_a": int(a), "snare_b": int(b), "blend": round(float(rng.uniform(0.15, 0.85)), 4)},
        stems=True)
# 3. random PCA macro points (on a random base snare)
for j in range(20):
    k = int(rng.integers(32))
    pcs = [round(float(x), 4) for x in rng.uniform(-2, 2, 3)]
    add(f"pc_{j:02d}", "pc", {"snare_a": k, "snare_b": k, "blend": 0.0, "pc1": pcs[0], "pc2": pcs[1], "pc3": pcs[2]},
        stems=True)

# determinism checks
det = {}
p = {"snare_a": 7, "snare_b": 7}
y1 = render(inst, p)["out"]; y2 = render(inst, p)["out"]
det["same_params_fresh_state_bit_identical"] = bool(np.array_equal(y1, y2))
inst512 = load(stems=True, max_frames=512)
y3 = render(inst512, p)["out"]
det["block128_vs_block512_maxabs"] = float(np.abs(y1 - y3).max())
y4, mem = inst.render(1.0, pitch=C4, params=p)
y5, _ = inst.render(1.0, pitch=C4, params=p, mem=mem)  # second hit, voice state carried
det["second_hit_same_voice_maxabs_diff"] = float(np.abs(y4[:, 0] - y5[:, 0]).max())
det["second_hit_head_stem_maxabs_diff"] = float(np.abs(y4[:, 1] - y5[:, 1]).max())

# modal tables for every render (exact block-rate values as the DSP computes them)
SLOT_KEYS = ["f", "amp", "dec0", "g0", "tau", "rise0", "t0", "ph", "mh", "md", "rg0"]
def modal(params):
    e = br.eval(**params)
    slots = [{k: e[f"b{i}_{k}"] for k in SLOT_KEYS} | {"presence_wB": e[f"pw_b{i}"]} for i in range(32)]
    chain = {n: v for n, v in e.items() if n.startswith("c_") and not n.endswith(("_m", "_x"))}
    glob = {n: e[n] for n in ("bank_level", "bank_norm", "dv", "d_amt", "d_norm", "spr", "bits_e", "len_s", "fade_s", "lvl")}
    return {"slots": slots, "global": glob, "chain": chain}

modes = {"doc": "Per-render VILLAIN Snare block-rate values evaluated from dsp.lisp by scripts/villain_table.py "
                "(validated: numpy re-synthesis of head_g matches the DSP head stem to ~1e-5 rel). "
                "Per slot i the head mode is m_i(t) = amp*exp(-t/(dec0*decay))*(1-exp(-t/rise0))*nyq_fade(hz)*"
                "sin(2pi*tuning*(f*t + sh*tau*(1-exp(-t/tau))) + ph), t = max(0, T - t0), T = age - c_pre, "
                "sh = max(-0.9 f, bend*g0*f), tuning = 2*pitch/440*tune (1.18921 at C4), gated by "
                "(1-mh(1-GH))(1-md(1-GD)), summed and scaled by head_v = c_head*vel^c_hv*(1-exp(-T/c_hr)). "
                "amp already includes bank_norm and body/ring knob (=1). f in Hz at tuning 1; times in s.",
         "tuning_at_C4": 2 * C4 / 440, "renders": {}}
for it in items:
    modes["renders"][it["id"]] = modal({k: v for k, v in it["params"].items()})

manifest = {"sample_rate": SR, "seconds": 1.0, "pitch_hz": C4, "format": "WAV float32 mono",
            "gate": "held for the whole render (Release inert; preset release recorded for reference)",
            "state": "fresh voice memory per render; trigger at sample 0; all (noise) xorshift32 states start at 0->1",
            "instrument": "content/instruments/Drums/VILLAIN Snare (stem-tap copy $MODAL_AD_ROOT/villain_stems from make_stems_copy.py, output bit-identical)",
            "compiler": inst.compiler, "compiler_sha256": inst.compiler_sha256,
            "stems_doc": {"head": "head_g pre-chain (modal bank incl. gates & head_v)", "source": "pre-recording-chain sum (drum+strokes+hiss+crackle)",
                          "strands": "snare-wire strand signal after rattle gain, pre buzz band-pass",
                          "out_headonly": "full output with rattle,buzz,crack,room,extra,strokes,hiss,dust = 0",
                          "out_noiseonly": "full output with body=ring=0 (no head => no strands either)",
                          "out_clean": "full output with grit=0 (SP sampler at 48k/24 bit = transparent) and drive=0.5 (tanh drive off); clipper + record filters + limiters remain"},
            "determinism": det, "items": items}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
json.dump(modes, open(os.path.join(RESEARCH, "data", "villain_modes.json"), "w"))
print(len(items), "renders;", det)
