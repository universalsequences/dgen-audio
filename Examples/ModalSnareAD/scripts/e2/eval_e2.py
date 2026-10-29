"""Evaluate a run dir of <id>.fit.wav / <id>.target.wav (latent-train or latent-fitz) with
E1's calibrated metrics: MR-STFT and band-envelope dB, normalised 0 = VILLAIN re-hit floor,
1 = nearest other snare (calib.json). Fit renders also get E1 per-snare fit scores (v100)
and velocity response (peak, centroid vs target). usage: eval_e2.py RUN -> RUN/eval.json"""
import glob, json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("e1"); MAD.use("dataset")
from metric import score, band_score
from villain_common import read_wav_f32
run = sys.argv[1]
cal = json.load(open(os.path.join(HERE, "calib.json")))
e1 = {r["id"]: r for r in json.load(open(os.path.join(R, "e1/fits/eval.json")))["rows"]}
def centroid(y, sr=48000):
    S = np.abs(np.fft.rfft(y[: int(0.3 * sr)])) ** 2; f = np.fft.rfftfreq(int(0.3 * sr), 1 / sr)
    return float((S * f).sum() / max(S.sum(), 1e-20))
rows = []
for fp in sorted(glob.glob(os.path.join(run, "*.fit.wav"))):
    rid = os.path.basename(fp)[:-8]
    y = read_wav_f32(fp).astype(np.float64)
    t = read_wav_f32(os.path.join(R, "data/villain", rid + ".wav")).astype(np.float64)
    c = cal[rid]
    r = {"id": rid, "kind": c["kind"], "vel": c["vel"], "mr": score(y, t), "band": band_score(y, t),
         "finite": bool(np.isfinite(y).all()), "peak": float(np.abs(y).max()), "peak_t": float(np.abs(t).max()),
         "cent": centroid(y), "cent_t": centroid(t)}
    r["mr_norm"] = (r["mr"] - c["mr_floor"]) / (c["mr_nearest"] - c["mr_floor"])
    r["band_norm"] = (r["band"] - c["band_floor"]) / (c["band_nearest"] - c["band_floor"])
    if rid in e1:
        r["e1_mr"] = e1[rid]["mr_fit"]; r["e1_band"] = e1[rid]["band_fit"]
        r["e1_mr_norm"] = e1[rid]["mr_norm"]; r["e1_band_norm"] = e1[rid]["band_norm"]
    rows.append(r)
def summ(sub, tag):
    if not sub: return {}
    k = ["mr", "band", "mr_norm", "band_norm"] + (["e1_mr", "e1_band", "e1_mr_norm", "e1_band_norm"] if "e1_mr" in sub[0] else [])
    out = {f"mean_{x}": float(np.mean([r[x] for r in sub])) for x in k}
    out.update({f"median_{x}": float(np.median([r[x] for r in sub])) for x in k})
    out["n"] = len(sub); out["all_finite"] = all(r["finite"] for r in sub)
    out["frac_norm_below_0.5"] = float(np.mean([(r["mr_norm"] < 0.5) and (r["band_norm"] < 0.5) for r in sub]))
    out["peak_ratio_median"] = float(np.median([r["peak"] / r["peak_t"] for r in sub]))
    out["cent_ratio_median"] = float(np.median([r["cent"] / r["cent_t"] for r in sub]))
    print(f"{tag:10s} n={len(sub):3d} MR {out['mean_mr']:.3f} band {out['mean_band']:.3f} dB | norm median MR {out['median_mr_norm']:.3f} band {out['median_band_norm']:.3f}"
          + (f" | E1 v100 norm median MR {out['median_e1_mr_norm']:.3f} band {out['median_e1_band_norm']:.3f}" if "median_e1_mr_norm" in out else "")
          + f" | <0.5 both {out['frac_norm_below_0.5']:.2f} | peak x{out['peak_ratio_median']:.2f} cent x{out['cent_ratio_median']:.2f}")
    return out
S = {}
for kind in ("fit", "blend", "pc"):
    sub = [r for r in rows if r["kind"] == kind]
    if kind == "fit":
        for v in (0.4, 0.7, 1.0): S[f"fit_v{v}"] = summ([r for r in sub if abs(r["vel"] - v) < 1e-6], f"fit v{v}")
        # velocity response: ratios to the snare's own v1.0 render, model vs VILLAIN
        vr = []
        for s in range(32):
            g = {r["vel"]: r for r in sub if r["id"].startswith(f"snare_{s:02d}_")}
            if 1.0 in g:
                for v in (0.4, 0.7):
                    if v in g: vr.append((v, g[v]["peak"] / g[1.0]["peak"], g[v]["peak_t"] / g[1.0]["peak_t"],
                                          g[v]["cent"] / g[1.0]["cent"], g[v]["cent_t"] / g[1.0]["cent_t"]))
        if vr:
            vr = np.array(vr)
            for v in (0.4, 0.7):
                m = vr[vr[:, 0] == v]
                print(f"  velocity {v}: peak/peak(v1) model {np.median(m[:,1]):.2f} villain {np.median(m[:,2]):.2f}; "
                      f"centroid ratio model {np.median(m[:,3]):.2f} villain {np.median(m[:,4]):.2f}; "
                      f"corr(peak ratio) {np.corrcoef(m[:,1], m[:,2])[0,1]:.2f}")
                S[f"velocity_{v}"] = {"peak_ratio_model": float(np.median(m[:, 1])), "peak_ratio_villain": float(np.median(m[:, 2])),
                                     "cent_ratio_model": float(np.median(m[:, 3])), "cent_ratio_villain": float(np.median(m[:, 4]))}
    S[kind] = summ(sub, kind)
json.dump({"rows": rows, "summary": S}, open(os.path.join(run, "eval.json"), "w"), indent=1)
