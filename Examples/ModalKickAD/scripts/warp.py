#!/usr/bin/env python3
"""E6's perceptual arc-length warp for the Latent Kick: the blend path stays the straight line
z = z_A + tau (z_B - z_A); per pair, blend t maps to the tau where the cumulative band-envelope
distance (kick_metric, full instrument render) from A is t of the way. Table [33*33, 11].
Also evaluates evenness (CV of 5%-step distances) with and without the warp on all pairs,
and VILLAIN's own blend line on 24 pairs for reference.
usage: warp.py RUN OUT.json [--workers 6]"""
import argparse, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool
import kick_np as KN, kick_layout as L
from kick_metric import band_env_db
from train_np import load, forward

NK = 11; TF = np.linspace(0, 1, 21)
G = {}

def init(run):
    W, Z, _ = load(run); G.update(W=W, Z=Z, bands=KN.noise_bands(48000))

def env_at(z):
    u = forward(G["W"], z[None])[2][0]
    return band_env_db(KN.render(L.decode_u(u), bands=G["bands"])["out"])

def dist(A, B, floor_db=70.0):
    lo = B.max() - floor_db
    return float(np.abs(np.maximum(A, lo) - np.maximum(B, lo)).mean())

def arc(ab):
    a, b = ab; Z = G["Z"]
    envs = [env_at((1 - t) * Z[a] + t * Z[b]) for t in TF]
    d = np.array([dist(envs[i + 1], envs[i]) for i in range(len(TF) - 1)])
    s = np.concatenate([[0], np.cumsum(d)]); s = s / max(s[-1], 1e-9)
    knots = np.interp(np.linspace(0, 1, NK), s, TF)
    # evenness through the warp (re-render at the warped taus)
    tau = np.interp(TF, np.linspace(0, 1, NK), knots)
    envw = [env_at((1 - t) * Z[a] + t * Z[b]) for t in tau]
    dw = np.array([dist(envw[i + 1], envw[i]) for i in range(len(TF) - 1)])
    ev = lambda x: {"cv": float(x.std() / max(x.mean(), 1e-9)), "max_over_mean": float(x.max() / max(x.mean(), 1e-9)),
                    "first_over_mean": float(x[0] / max(x.mean(), 1e-9)), "total": float(x.sum())}
    return a, b, knots.tolist(), ev(d), ev(dw)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    pairs = list(itertools.combinations(range(33), 2))
    with Pool(a.workers, initializer=init, initargs=(a.run,)) as pool:
        rows = pool.map(arc, pairs, chunksize=4)
    T = np.zeros((33, 33, NK)); T[:, :] = np.linspace(0, 1, NK)
    for i, j, k, _, _ in rows:
        T[i, j] = k; T[j, i] = 1 - np.array(k)[::-1]
    summ = {}
    for name, idx in (("straight", 3), ("warped", 4)):
        summ[name] = {k: float(np.median([r[idx][k] for r in rows])) for k in ("cv", "max_over_mean", "first_over_mean")}
        summ[name]["cv_p90"] = float(np.percentile([r[idx]["cv"] for r in rows], 90))
    print(json.dumps(summ))
    json.dump({"run": os.path.abspath(a.run), "knots_t": np.linspace(0, 1, NK).tolist(), "table": T.tolist(), "eval": summ,
               "doc": "table[a][b][k] = tau at blend t = k/10 for A=a, B=b; piecewise linear"}, open(a.out, "w"))
