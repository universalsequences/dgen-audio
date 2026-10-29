"""Pretrain fidelity: decoded params vs teacher targets (natural units) and audio
(template render with E1's bank) vs out.wav, for every fit render. usage: eval_pre.py RUN"""
import json, math, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("e2"); MAD.use("e1"); MAD.use("dataset")
import template as tp, latent_np as ln
from metric import score, band_score
from villain_common import read_wav_f32
run = sys.argv[1]
W, Z, names = ln.load_run(run)
tg = json.load(open(os.path.join(HERE, "targets.json"))); bank = ln.e1_bank()
cal = json.load(open(os.path.join(HERE, "calib.json"))) if os.path.exists(os.path.join(HERE, "calib.json")) else {}
rows = []
for rid, t in tg.items():
    nat = {k: np.array(v) for k, v in t["natural"].items()}
    p = tp.decode(W, Z[t["snare"]], t["vel"])
    E = nat["mode_amp"] ** 2 * nat["mode_decay"]; w = E / E.sum()
    cents = 1200 * np.abs(np.log2(p["mode_freq"] / nat["mode_freq"]))
    ampe = np.abs(np.log(np.maximum(p["mode_amp"], 1e-6) / np.maximum(nat["mode_amp"], 1e-6)))
    dece = np.abs(np.log(p["mode_decay"] / nat["mode_decay"]))
    y = tp.render_natural(p, level=1.0, bands=bank)
    ref = read_wav_f32(os.path.join(R, "data/villain", rid + ".wav"))
    r = {"id": rid, "cents_w": float((cents * w).sum()), "cents_max_live": float(cents[w > 0.01].max()),
         "amp_logerr_w": float((ampe * w).sum()), "dec_logerr_w": float((dece * w).sum()),
         "mr": score(y, ref), "band": band_score(y, ref)}
    if rid in cal:
        c = cal[rid]
        r["mr_norm"] = (r["mr"] - c["mr_floor"]) / (c["mr_nearest"] - c["mr_floor"])
        r["band_norm"] = (r["band"] - c["band_floor"]) / (c["band_nearest"] - c["band_floor"])
    rows.append(r)
keys = [k for k in rows[0] if k != "id"]
for vt in ("040", "070", "100", ""):
    sub = [r for r in rows if r["id"].endswith(vt)]
    print(f"v{vt or 'all'}: " + " ".join(f"{k} {np.mean([r[k] for r in sub]):.4g}" for k in keys) + (f" | median mr_norm {np.median([r['mr_norm'] for r in sub]):.3f} band_norm {np.median([r['band_norm'] for r in sub]):.3f}" if 'mr_norm' in sub[0] else ""))
json.dump(rows, open(os.path.join(run, "eval_pre.json"), "w"), indent=1)
