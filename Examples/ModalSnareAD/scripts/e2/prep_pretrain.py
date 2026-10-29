"""targets.json (natural) -> pretrain.json for the Swift trainer:
s*  : sigmoid-position targets per render [96, P] (pi segments: (x/pi+1)/2, model side sigmoid(2u))
w   : per-output loss weight [96, P]: (c_seg)^2 * element weight, c_seg converts an
      s error to natural units (log segments: log-ratio; freq x10 so cents count)
z_init[D]: PCA of the weighted v100 s-vectors, unit variance per dim."""
import json, math, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); MAD.use("e2")
import template as tp
L = tp.LAY; P = L["P"]
BLEND = "--blend" in sys.argv   # + blend-line rows (targets_blend.json) -> pretrain_blend.json
tg = json.load(open(os.path.join(HERE, "targets.json")))
ids = [f"snare_{s:02d}_v{v}" for s in range(32) for v in ("040", "070", "100")]
for r in ids: tg[r].update(a=tg[r]["snare"], b=tg[r]["snare"], t=0.0, roww=1.0)
if BLEND:
    tb = json.load(open(os.path.join(HERE, "targets_blend.json")))
    for k, v in tb.items(): v.update(name=k, snare=v["a"], roww=float(os.environ.get("BLEND_ROWW", "0.3")))
    tg.update(tb); ids = ids + sorted(tb)
IMP = {"mode_freq": 30, "mode_amp": 1, "mode_decay": 1, "mode_rise": 0.5, "mode_t0": 0.2, "mode_glide": 2,
       "mode_gtau": 0.5, "band_amp": 1, "band_d1": 0.5, "band_tail": 1, "band_d2": 0.5, "band_couple": 0.5}
def cseg(s):
    imp = IMP.get(s["name"], 1.0)
    if s["squash"] == "log": return imp * math.log(s["hi"] / s["lo"])
    if s["squash"] == "pi": return imp * 2 * math.pi
    if s["name"] in ("pre", "noise_onset"): return 0.2 * 1000 * (s["hi"] - s["lo"])  # ms
    return imp * 2.0 if s["name"] in ("mode_glide", "drive_amt", "gh_depth", "gd_depth") else imp
S = np.zeros((len(ids), P)); Wt = np.zeros((len(ids), P))
for i, rid in enumerate(ids):
    nat = tg[rid]["natural"]; mw = np.array(nat["_mode_w"]); vel = tg[rid]["vel"]
    for s in L["segments"]:
        o, w = s["offset"], s["width"]; x = np.array(nat[s["name"]])
        S[i, o:o + w] = tp.unsquash(s["name"], x)
        ew = np.ones(w)
        if s["name"].startswith("mode_") and s["name"] != "mode_amp": ew = mw.copy()
        if s["name"] == "mode_phase": ew = mw * 0.3
        if s["name"] in ("band_d1", "band_tail", "band_d2"):
            A = np.array(nat["band_amp"]); ew = np.sqrt(A ** 2 / (A ** 2).sum()) + 0.05
        if s["name"] in ("gh_time", "gh_fall"): ew[:] = nat["gh_depth"][0] + 0.05
        if s["name"] in ("gd_time", "gd_fall"): ew[:] = nat["gd_depth"][0] + 0.05
        if s["name"] == "gate_noise": ew[:] = 0.1
        if s["name"].startswith("wire_"): ew[:] = 0.0   # E5 W2 knobs: distilled later
        if s["name"].startswith("band_") and vel < 1.0: ew = ew * 0.3   # guessed noise at v<1
        Wt[i, o:o + w] = cseg(s) ** 2 * ew * tg[rid]["roww"]
Wt /= Wt.sum() / Wt.size
# scale so the mean-snare predictor scores 1 (then z priors / anchors are relative to that)
nfit = 96
L0 = float((Wt[:nfit] * (S[:nfit] - S[:nfit].mean(0)) ** 2).mean()); Wt /= L0   # fit rows define the scale
print("weights normalised; mean-predictor loss 1 (raw", L0, ")")
v100 = [i for i, r in enumerate(ids[:96]) if r.endswith("100")]
X = S[v100] * np.sqrt(Wt[v100]); X = X - X.mean(0)
U, sv, Vt = np.linalg.svd(X, full_matrices=False)
print("PCA explained (first 10):", np.round((sv ** 2 / (sv ** 2).sum())[:10], 3))
zi = {}
for D in (2, 4, 6, 8, 12, 16):
    z = U[:, :D] * sv[:D]; z = z / z.std(0)
    zi[str(D)] = z.reshape(-1).tolist()
audio = list(range(nfit))
if BLEND:
    br = json.load(open(os.path.join(HERE, "blend_targets", "rows.json")))
    audio += [ids.index(r) for r in br]
out = {"ids": ids, "snare": [tg[r]["snare"] for r in ids], "vel": [tg[r]["vel"] for r in ids],
       "a": [tg[r]["a"] for r in ids], "b": [tg[r]["b"] for r in ids], "t": [tg[r]["t"] for r in ids],
       "audio_rows": audio, "nfit": nfit,
       "names": [tg[r]["name"] for r in ids], "P": P, "s": S.reshape(-1).tolist(), "w": Wt.reshape(-1).tolist(), "z_init": zi}
json.dump(out, open(os.path.join(HERE, "pretrain_blend.json" if BLEND else "pretrain.json"), "w"))
print("wrote pretrain.json", S.shape)
