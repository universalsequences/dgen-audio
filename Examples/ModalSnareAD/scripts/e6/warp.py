#!/usr/bin/env python3
"""Perceptual arc-length warp of the blend knob, per snare pair: the path stays the straight line
z = z_A + tau (z_B - z_A); only the SPEED along it changes, so that blend t lands where the cumulative
band-envelope distance from A (full output, same noise) is t of the way.
1. For every unordered pair, render 41 points (tau = 0, .025, .., 1), cumulative step distance s(tau).
2. Knots: tau at t = 0, .1, .., 1 by inverting s / s(1) (piecewise linear). Table [32*32, 11] for ordered
   pairs (b->a uses 1 - knots reversed); a == a is the identity.
3. --eval N: re-render N pairs at t = 0, .05, .., 1 THROUGH the warp and measure evenness again.
usage: warp.py RUN OUT.json [--eval 100]"""
import argparse, itertools, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

NK = 11; TF = np.linspace(0, 1, 41)
G = {}


def init(run):
    W, Z, names = C.load_run(run); C.bank(); G.update(W=W, Z=Z, names=names)


def arc(ab):
    a, b = ab; W, Z = G["W"], G["Z"]
    envs = [C.band_env(C.render(W, (1 - t) * Z[a] + t * Z[b])[1]) for t in TF]
    d = np.array([C.band_from_env(envs[i], envs[i + 1]) for i in range(len(TF) - 1)])
    s = np.concatenate([[0], np.cumsum(d)]); s /= s[-1]
    knots = np.interp(np.linspace(0, 1, NK), s, TF)
    return a, b, knots.tolist(), d.tolist()


def tau_of(knots, t):
    x = t * (NK - 1); k = min(int(np.floor(x)), NK - 2); f = x - k
    return (1 - f) * knots[k] + f * knots[k + 1]


def ev(args):
    a, b, knots = args; W, Z = G["W"], G["Z"]
    envs = []; envs0 = []
    for t in np.arange(21) * 0.05:
        tau = tau_of(knots, t)
        envs.append(C.band_env(C.render(W, (1 - tau) * Z[a] + tau * Z[b])[1]))
    st = [C.band_from_env(envs[i], envs[i + 1]) for i in range(20)]
    return C.evenness(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--eval", type=int, default=100)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    pairs = list(itertools.combinations(range(32), 2))
    with Pool(a.workers, initializer=init, initargs=(a.run,)) as pool:
        rows = pool.map(arc, pairs, chunksize=4)
        T = np.zeros((32, 32, NK)); T[:, :] = np.linspace(0, 1, NK)
        for i, j, k, _ in rows:
            T[i, j] = k; T[j, i] = 1 - np.array(k)[::-1]
        rng = np.random.default_rng(6); sel = [pairs[i] for i in sorted(rng.choice(len(pairs), a.eval, replace=False))]
        evs = pool.map(ev, [(i, j, T[i, j].tolist()) for i, j in sel])
    s = {k: float(np.median([e[k] for e in evs])) for k in ("cv", "max_over_mean", "first_over_mean")}
    s["cv_p90"] = float(np.percentile([e["cv"] for e in evs], 90)); s["max_over_mean_p90"] = float(np.percentile([e["max_over_mean"] for e in evs], 90))
    dev = float(np.median(np.abs(T[np.triu_indices(32, 1)] - np.linspace(0, 1, NK)).max(-1)))
    print("warped walks (band metric, %d pairs):" % a.eval, s, "| median max |tau - t|", round(dev, 3))
    C.json_dump({"run": os.path.abspath(a.run), "knots_t": np.linspace(0, 1, NK).tolist(), "table": T.tolist(), "eval": s,
                 "median_max_dev": dev, "doc": "table[a][b][k] = tau at blend t = k/10 for A=a, B=b; piecewise linear"}, a.out)
