#!/usr/bin/env python3
"""Built Latent Kick (A/B front end) vs the numpy twin of the same weights, and vs VILLAIN Kick:
every kick at blend 0 (v1.0/0.4), 6 warped midpoints, 3 knob settings.  usage: installed_parity.py RUN INST_DIR"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
os.environ["ESEQ_DGENLISP_TOOL"] = MAD.dgenlisp()  # the Latent Kick needs this checkout's compiler
from kick_common import DATA, read_wav_f32, C4, Instrument
from kick_metric import band_score, relrms
import kick_np as KN, kick_layout as L
from train_np import load, forward
run, idir = sys.argv[1], sys.argv[2]
W, Z, ids = load(run); warp = np.array(json.load(open(os.path.join(run, "warp.json")))["table"])
inst = Instrument(idir, sample_rate=48000.0, max_frames=128)
bands = KN.noise_bands(48000)
abl = json.load(open(os.path.join(DATA, "ablations.json")))["nearest_other_kick"]["band_per_kick"]
tw = lambda z, **kw: KN.render(L.decode_u(forward(W, np.asarray(z)[None])[2][0]), bands=bands, **kw)["out"]
rows = []
for k in range(33):
    for vel in (1.0, 0.4):
        y = inst.render(1.0, pitch=C4, vel=vel, params={"kick_a": k, "kick_b": k})[0]
        y = y[:, 0] if y.ndim > 1 else y
        tgt = read_wav_f32(os.path.join(DATA, "villain", f"kick_{k:02d}_v{int(vel * 100):03d}.wav"))
        b = band_score(y, tgt)
        rows.append({"id": f"k{k}_v{vel}", "inst_vs_twin": relrms(y, tw(Z[k], vel=vel)), "band_vs_villain": b, "norm": b / abl[k],
                     "peak": float(np.abs(y).max()), "peak_villain": float(np.abs(tgt).max())})
rng = np.random.default_rng(1)
for _ in range(6):
    a, b = rng.choice(33, 2, replace=False)
    tau = float(np.interp(0.5, np.linspace(0, 1, 11), warp[a, b]))
    y = inst.render(1.0, pitch=C4, vel=1.0, params={"kick_a": int(a), "kick_b": int(b), "blend": 0.5})[0]
    y = y[:, 0] if y.ndim > 1 else y
    rows.append({"id": f"blend_{a}_{b}", "inst_vs_twin": relrms(y, tw((1 - tau) * Z[a] + tau * Z[b]))})
for kn in ({"tune": 1.4}, {"decay": 0.5, "bend": 1.8}, {"spread": 1.3, "tilt": 0.8, "shape": 2.0}):
    y = inst.render(1.0, pitch=C4 * 0.75, vel=0.8, params={"kick_a": 11, "kick_b": 11, **kn})[0]
    y = y[:, 0] if y.ndim > 1 else y
    rows.append({"id": f"knobs_{kn}", "inst_vs_twin": relrms(y, tw(Z[11], vel=0.8, pitch=C4 * 0.75, knobs=kn))})
for r in rows: print(r)
fits = [r for r in rows if r["id"].endswith("v1.0")]
print("inst vs twin median %.2e max %.2e" % (np.median([r["inst_vs_twin"] for r in rows]), max(r["inst_vs_twin"] for r in rows)))
print("v1.0 band vs VILLAIN median %.3f, norm median %.3f max %.3f, peak ratio median %.3f" % (
    np.median([r["band_vs_villain"] for r in fits]), np.median([r["norm"] for r in fits]), max(r["norm"] for r in fits),
    np.median([r["peak"] / r["peak_villain"] for r in fits])))
json.dump(rows, open(os.path.join(run, "installed_parity.json"), "w"), indent=1)
