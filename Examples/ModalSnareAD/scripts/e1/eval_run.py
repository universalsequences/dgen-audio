"""Per-snare evaluation table for a fit run: MR-STFT score (training-selection metric) and the
realisation-robust band-envelope dB metric, each with controls: noise-realisation floor (VILLAIN hit 2),
nearest wrong snare, silence. usage: eval_run.py RUN_DIR -> RUN_DIR/eval.json + table on stdout"""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1"); MAD.use("e1"); MAD.use("dataset")
from metric import score, band_score
from villain_common import read_wav_f32
run = sys.argv[1]; V = os.path.join(HERE, "..", "data/villain")
ids = [f"snare_{i:02d}_v100" for i in range(32)]
T = {r: read_wav_f32(f"{V}/{r}.wav") for r in ids}
rows = []
for r in ids:
    t = T[r]; fit = read_wav_f32(os.path.join(run, r + ".fit.wav")); h2 = read_wav_f32(os.path.join(HERE, "noise_floor", r + ".hit2.wav"))
    others = [o for o in ids if o != r]
    wrong_mr = min(score(T[o], t) for o in others); wrong_b = min(band_score(T[o], t) for o in others)
    row = {"id": r, "mr_fit": score(fit, t), "mr_floor": score(h2, t), "mr_wrong": wrong_mr, "mr_silence": score(np.zeros_like(t), t),
           "band_fit": band_score(fit, t), "band_floor": band_score(h2, t), "band_wrong": wrong_b, "band_silence": band_score(np.zeros_like(t), t)}
    row["mr_norm"] = (row["mr_fit"] - row["mr_floor"]) / (row["mr_wrong"] - row["mr_floor"])
    row["band_norm"] = (row["band_fit"] - row["band_floor"]) / (row["band_wrong"] - row["band_floor"])
    rows.append(row)
    print(f"{r} MR fit {row['mr_fit']:.3f} floor {row['mr_floor']:.3f} wrong {row['mr_wrong']:.3f} norm {row['mr_norm']:+.2f} | band dB fit {row['band_fit']:.2f} floor {row['band_floor']:.2f} wrong {row['band_wrong']:.2f} sil {row['band_silence']:.1f} norm {row['band_norm']:+.2f}")
keys = [k for k in rows[0] if k != "id"]
mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}; med = {k: float(np.median([r[k] for r in rows])) for k in keys}
print("MEAN", {k: round(v, 3) for k, v in mean.items()}); print("MEDIAN", {k: round(v, 3) for k, v in med.items()})
json.dump({"rows": rows, "mean": mean, "median": med}, open(os.path.join(run, "eval.json"), "w"), indent=1)
