#!/usr/bin/env python3
"""Per-render stats for the VILLAIN dataset -> data/villain/stats.json + a table on stdout."""
import json, os
import numpy as np
from villain_common import read_wav_f32, RESEARCH, SR

D = os.path.join(RESEARCH, "data", "villain")
man = json.load(open(os.path.join(D, "manifest.json")))
modes = json.load(open(os.path.join(RESEARCH, "data", "villain_modes.json")))["renders"]

def env_db(y, win=96):
    p = np.convolve(y.astype(np.float64) ** 2, np.ones(win) / win, mode="same")
    return 10 * np.log10(p + 1e-20)

def decay_time(y, drop):
    e = env_db(y); i0 = int(np.argmax(e)); thr = e[i0] - drop
    above = np.nonzero(e[i0:] > thr)[0]
    last = i0 + above[-1]
    return None if last >= len(y) - 200 else (last - i0) / SR

def centroid(y, t0=0.0, t1=1.0):
    seg = y[int(t0 * SR):int(t1 * SR)].astype(np.float64)
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float((S * f).sum() / S.sum())

E = lambda y: float((y.astype(np.float64) ** 2).sum())
rows = {}
for it in man["items"]:
    y = read_wav_f32(os.path.join(D, it["file"]))
    r = {"peak": float(np.abs(y).max()), "t_peak_ms": 1000 * float(np.argmax(np.abs(y))) / SR,
         "t30_ms": None, "t60_ms": None, "centroid_hz": centroid(y), "centroid_0_50ms_hz": centroid(y, 0, 0.05),
         "rms_dbfs": float(10 * np.log10(E(y) / len(y)))}
    t30, t60 = decay_time(y, 30), decay_time(y, 60)
    r["t30_ms"] = None if t30 is None else 1000 * t30
    r["t60_ms"] = None if t60 is None else 1000 * t60
    st = it.get("stems", {})
    if "head" in st:
        h = read_wav_f32(os.path.join(D, st["head"])); s = read_wav_f32(os.path.join(D, st["source"]))
        r["prechain_head_energy_frac"] = E(h) / E(s)
        r["head_t60_ms"] = None if decay_time(h, 60) is None else 1000 * decay_time(h, 60)
        r["head_centroid_hz"] = centroid(h)
    if "out_headonly" in st:
        ho = read_wav_f32(os.path.join(D, st["out_headonly"])); no = read_wav_f32(os.path.join(D, st["out_noiseonly"]))
        r["out_headonly_energy_frac"] = E(ho) / E(y)
        r["out_noiseonly_energy_frac"] = E(no) / E(y)
        cl = read_wav_f32(os.path.join(D, st["out_clean"]))
        r["sp_drive_diff_energy_frac"] = E(y - cl) / E(y)
        r["out_residual_energy_frac"] = E(y - ho - no) / E(y)  # strands + head strokes + nonlinear cross terms
    sl = modes[it["id"]]["slots"]
    amps = np.array([abs(s["amp"]) for s in sl])
    # mode energy ~ amp^2 * dec/2 (ignoring rise)
    en = amps ** 2 * np.array([s["dec0"] for s in sl]) / 2
    r["n_modes_energy_gt_1pct"] = int((en > 0.01 * en.sum()).sum())
    top = int(np.argmax(en)); r["dominant_slot"] = top
    r["dominant_hz_heard"] = sl[top]["f"] * 2 * 261.6255653 / 440
    r["dominant_dec_ms"] = 1000 * sl[top]["dec0"]
    rows[it["id"]] = r
json.dump(rows, open(os.path.join(D, "stats.json"), "w"), indent=1)

def fmt(v, p=0):
    return "  -  " if v is None else f"{v:.{p}f}"
print("id          name      peak  t30ms t60ms  cent  c50ms  headE  hoE  noE  resE  nModes domHz domDec")
for k in range(32):
    rid = f"snare_{k:02d}_v100"; r = rows[rid]
    name = next(i["name"] for i in man["items"] if i["id"] == rid)
    print(f"{rid} {name:9s} {r['peak']:.2f} {fmt(r['t30_ms']):>5} {fmt(r['t60_ms']):>5} {r['centroid_hz']:5.0f} {r['centroid_0_50ms_hz']:5.0f}"
          f"  {r['prechain_head_energy_frac']:.2f}  {r['out_headonly_energy_frac']:.2f} {r['out_noiseonly_energy_frac']:.2f} {r['out_residual_energy_frac']:.2f}"
          f"  {r['n_modes_energy_gt_1pct']:3d}  {r['dominant_hz_heard']:5.0f} {r['dominant_dec_ms']:5.0f}")
for key in ("peak", "t30_ms", "t60_ms", "centroid_hz", "out_headonly_energy_frac", "out_noiseonly_energy_frac", "out_residual_energy_frac", "prechain_head_energy_frac", "sp_drive_diff_energy_frac"):
    v = [rows[f"snare_{k:02d}_v100"][key] for k in range(32)]
    vv = np.array([x for x in v if x is not None])
    print(f"{key:28s} median {np.median(vv):8.3f}  min {vv.min():8.3f}  max {vv.max():8.3f}  (n={len(vv)}/32)")
for vel in ("040", "070"):
    rp = [rows[f"snare_{k:02d}_v{vel}"]["peak"] / rows[f"snare_{k:02d}_v100"]["peak"] for k in range(32)]
    rc = [rows[f"snare_{k:02d}_v{vel}"]["centroid_hz"] / rows[f"snare_{k:02d}_v100"]["centroid_hz"] for k in range(32)]
    print(f"vel {vel}: peak ratio median {np.median(rp):.3f} [{min(rp):.3f},{max(rp):.3f}]  centroid ratio median {np.median(rc):.3f} [{min(rc):.3f},{max(rc):.3f}]")
