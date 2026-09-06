#!/usr/bin/env python3
"""Diagnose a clap fit: metric noise floor (same patch, other noise seed),
residual grid (time x frequency cells of the 1024-window log-mag error), and
envelope/burst comparison. Usage: clap_residual.py output/clap_vN"""
import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
import compare, fit_clap as fc, render_reference as ref

out = sys.argv[1]
rep = json.load(open(os.path.join(out, "recovered_params.json")))
p, sr, frames = rep["params"], rep["sampleRate"], rep["frames"]
target, _ = compare.read_wav(os.path.join(out, "target.wav"))
learned, _ = compare.read_wav(os.path.join(out, "learned.wav"))
obj = fc.Objective(target, sr, compare.DEFAULT_HIGHPASS_HZ)

# noise floor: same params, noise stream shifted by a large offset
orig = ref.dgen_noise
def shifted(n):
    return orig(n + 7919)[7919:]
ref.dgen_noise = shifted
fc.noise_bp.cache_clear(); fc.noise_bp_lp.cache_clear()
alt = fc.render(p, frames, sr)
pk = float(np.max(np.abs(alt)));  alt = alt * np.float32(0.9 / pk) if pk > 0.9 else alt
ref.dgen_noise = orig
fc.noise_bp.cache_clear(); fc.noise_bp_lp.cache_clear()
hp = lambda x: compare.capture_highpass(x, sr, compare.DEFAULT_HIGHPASS_HZ)
d_learned = obj.distance(hp(learned)); d_alt = obj.distance(hp(alt))
d_self = compare.mrstft(hp(learned), hp(alt))
print(f"learned vs target {d_learned:.4f} | other-seed vs target {d_alt:.4f} | learned vs other-seed (metric floor) {d_self:.4f}")
print(f"excess over floor: {d_learned - d_self:.4f}  baseline {rep['baseline']:.4f}")

# residual grid, 1024 window
w = 1024; hop = w // 4
ft = obj.features(hp(target), w); fl = obj.features(hp(learned), w)
err = np.abs(ft - fl)
f = np.fft.rfftfreq(w, 1 / sr); tms = (np.arange(err.shape[0]) * hop + w / 2) / sr * 1000
tb = [0, 15, 30, 50, 80, 120, 200, 300, 450]; fb = [0, 300, 600, 1000, 1600, 2500, 4000, 7000, 12000, 24000]
tot = err.sum()
print("residual share (%) rows=time ms, cols=freq Hz; (mean signed error dB: +=learned louder)")
print("            " + " ".join(f"{a:>5d}-{b:<5d}" for a, b in zip(fb[:-1], fb[1:])))
for a, b in zip(tb[:-1], tb[1:]):
    tm = (tms >= a) & (tms < b)
    row = []
    for c, d in zip(fb[:-1], fb[1:]):
        fm = (f >= c) & (f < d)
        cell = err[np.ix_(tm, fm)]
        signed = (fl - ft)[np.ix_(tm, fm)].mean() * 8.686 if cell.size else 0
        row.append(f"{100 * cell.sum() / tot:4.1f}({signed:+4.0f})")
    print(f"{a:4d}-{b:<4d} ms " + " ".join(f"{r:>11s}" for r in row))

# envelope comparison (1 ms rms), first 60 ms + tail decay
def env(x):
    n = int(0.001 * sr); k = len(x) // n
    e = np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1)); return 20 * np.log10(e / e.max() + 1e-9)
et, el = env(target), env(learned)
print("env dB  t:", np.round(et[:48]).astype(int).tolist())
print("env dB  l:", np.round(el[:48]).astype(int).tolist())
for a, b in [(40, 80), (80, 160), (160, 300)]:
    st = np.polyfit(np.arange(a, b) / 1000, et[a:b] / 8.686, 1)[0]; sl = np.polyfit(np.arange(a, b) / 1000, el[a:b] / 8.686, 1)[0]
    print(f"tail decay {a}-{b} ms: target {st:.1f}/s learned {sl:.1f}/s")
print("pinned:", rep["pinned"])
