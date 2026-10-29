#!/usr/bin/env python3
"""E6 task 1: straight-line walks z(t) = (1-t) z_A + t z_B between snare codes, t = 0, .05, .., 1,
velocity 1, numpy twin with E1's fixed band noise (same realisation for every t).
Per pair: step distances (band-envelope dB + MR-STFT) on the full output and the exact head,
evenness (max/mean, CV, first/mean), degeneracy flags, slot-wise log-f linearity.
usage: measure_walks.py RUN OUT.json [--pairs all|N] [--workers 4]"""
import argparse, itertools, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TS = np.round(np.arange(21) * 0.05, 4)
G = {}


def init(run):
    W, Z, names = C.load_run(run); C.bank()
    G.update(W=W, Z=Z, names=names)
    # corpus ranges for the wild-pitch / centroid flags (decoder at the 32 codes)
    f4 = []; cents = []
    for s in range(len(Z)):
        p, y, h, n = C.render(W, Z[s]); f4.append(p["mode_freq"][4]); cents.append(C.centroid(y))
    G.update(f4=(min(f4), max(f4)), cent=(min(cents), max(cents)))


def walk(ab):
    a, b = ab; W, Z = G["W"], G["Z"]
    pts = []; envs_y = []; envs_h = []; ys = []; hs = []
    for t in TS:
        z = (1 - t) * Z[a] + t * Z[b]
        p, y, h, n = C.render(W, z)
        d = C.descriptors(p, y, h, n)
        d["beats"] = len(C.beat_pairs(p))
        d["logf"] = np.log(np.asarray(p["mode_freq"]))
        d["E"] = C.mode_energy(p)
        pts.append(d); envs_y.append(C.band_env(y)); envs_h.append(C.band_env(h)); ys.append(y); hs.append(h)
    steps = {"band": [C.band_from_env(envs_y[i], envs_y[i + 1]) for i in range(20)],
             "band_head": [C.band_from_env(envs_h[i], envs_h[i + 1]) for i in range(20)],
             "mr": [C.mr(ys[i], ys[i + 1]) for i in range(20)],
             "mr_head": [C.mr(hs[i], hs[i + 1]) for i in range(20)]}
    chord = {"band": C.band_from_env(envs_y[0], envs_y[-1]), "mr": C.mr(ys[0], ys[-1]),
             "band_head": C.band_from_env(envs_h[0], envs_h[-1])}
    # degeneracy
    pk0, pk1 = pts[0]["peak"], pts[-1]["peak"]; r0, r1 = pts[0]["rms_db"], pts[-1]["rms_db"]
    flags = []
    for i, d in enumerate(pts[1:-1], 1):
        f = []
        if d["peak"] < 0.03: f.append("silent")
        if d["peak"] > 1.05 and d["peak"] > 1.02 * max(pk0, pk1): f.append("hot")
        if d["rms_db"] < min(r0, r1) - 6: f.append("hole")
        if not (0.5 * G["f4"][0] <= d["f0"] <= 2 * G["f4"][1]): f.append("pitch")
        lo, hi = sorted((pts[0]["f0"], pts[-1]["f0"]))
        if d["f0"] < lo / 2 ** (2 / 12) or d["f0"] > hi * 2 ** (2 / 12): f.append("f0_overshoot")
        if not (0.5 * G["cent"][0] <= d["centroid"] <= 2 * G["cent"][1]): f.append("centroid")
        if d["beats"] > max(pts[0]["beats"], pts[-1]["beats"]): f.append("beat")
        if f: flags.append((float(TS[i]), f))
    # slot-wise log-f linearity: decoded log f(t) vs (1-t) log fA + t log fB, energy-weighted cents
    lin = []
    for i, t in enumerate(TS):
        ref = (1 - t) * pts[0]["logf"] + t * pts[-1]["logf"]
        Ew = np.sqrt(pts[i]["E"] * ((1 - t) * pts[0]["E"] + t * pts[-1]["E"]))
        lin.append(float(np.sum(Ew * np.abs(pts[i]["logf"] - ref)) / max(Ew.sum(), 1e-30) * 1200 / np.log(2)))
    desc = {k: [float(d[k]) for d in pts] for k in ("f0", "f_dom", "f_logmean", "centroid", "t30", "noise_head_db", "peak", "rms_db", "beats")}
    return {"a": int(a), "b": int(b), "steps": steps, "chord": chord,
            "even": {k: C.evenness(v) for k, v in steps.items()}, "flags": flags, "logf_dev_cents": lin, "desc": desc}


def summarize(rows, names):
    out = {}
    for k in ("band", "band_head", "mr", "mr_head"):
        mm = np.array([r["even"][k]["max_over_mean"] for r in rows]); cv = np.array([r["even"][k]["cv"] for r in rows])
        fm = np.array([r["even"][k]["first_over_mean"] for r in rows])
        out[k] = {"max_over_mean_median": float(np.median(mm)), "max_over_mean_p90": float(np.percentile(mm, 90)),
                  "max_over_mean_max": float(mm.max()), "cv_median": float(np.median(cv)), "cv_p90": float(np.percentile(cv, 90)),
                  "first_over_mean_median": float(np.median(fm)), "mean_step_median": float(np.median([r["even"][k]["mean"] for r in rows]))}
    worst = sorted(rows, key=lambda r: -r["even"]["band"]["cv"])[:10]
    out["worst_band_cv"] = [{"pair": f"{names[r['a']]}-{names[r['b']]}", "cv": r["even"]["band"]["cv"],
                             "max_over_mean": r["even"]["band"]["max_over_mean"],
                             "argmax_t": float(TS[int(np.argmax(r["steps"]["band"]))] + 0.025)} for r in worst]
    fl = [f for r in rows for _, fs in r["flags"] for f in fs]
    out["flags"] = {k: fl.count(k) for k in sorted(set(fl))}
    out["pairs_with_flags"] = sum(bool(r["flags"]) for r in rows)
    out["points"] = 19 * len(rows)
    out["logf_dev_mid_cents_median"] = float(np.median([r["logf_dev_cents"][10] for r in rows]))
    out["logf_dev_mid_cents_p90"] = float(np.percentile([r["logf_dev_cents"][10] for r in rows], 90))
    out["n_pairs"] = len(rows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--pairs", default="all"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    init(a.run); names = G["names"]
    pairs = list(itertools.combinations(range(len(G["Z"])), 2))
    if a.pairs != "all":
        rng = np.random.default_rng(6); pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), int(a.pairs), replace=False))]
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.run,)) as pool:
        rows = pool.map(walk, pairs, chunksize=4)
    s = summarize(rows, names)
    C.json_dump({"run": os.path.abspath(a.run), "ts": TS.tolist(), "summary": s, "rows": rows}, a.out)
    import json; print(json.dumps(s, indent=1)); print(f"{len(rows)} pairs in {time.time() - t0:.0f} s")
