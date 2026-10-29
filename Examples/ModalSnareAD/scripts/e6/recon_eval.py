#!/usr/bin/env python3
"""Reconstruction cost of a run at the 96 training renders (numpy twin, E1 band noise, no wires):
band/MR vs the VILLAIN renders, normalised by e2/calib.json; head-frequency error vs the table
(energy-weighted cents, live modes) and vs a reference run's decoder.
usage: recon_eval.py RUN [REF_RUN]"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from villain_common import read_wav_f32
run = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else None
cal = json.load(open(os.path.join(C.E2, "calib.json"))); tg = json.load(open(os.path.join(C.E2, "targets.json")))
pre = json.load(open(os.path.join(C.E2, "pretrain.json"))); n = pre["nfit"]
rows = list(zip(pre["ids"][:n], pre["snare"][:n], pre["vel"][:n]))


def one(r):
    rid, c, v = r
    W, Z, _ = C.load_run(run); p, y, h, nn = C.render(W, Z[c], v)
    t = read_wav_f32(os.path.join(C.R, "data/villain", rid + ".wav")).astype(float)
    k = cal[rid]; mr = C.score(y, t); bd = C.band_score(y, t)
    f = np.asarray(p["mode_freq"]); ft = np.asarray(tg[rid]["natural"]["mode_freq"]); E = C.mode_energy(p)
    live = E > 0.005 * E.sum(); cents = np.abs(1200 * np.log2(f / ft))
    out = {"mr_norm": (mr - k["mr_floor"]) / (k["mr_nearest"] - k["mr_floor"]), "band_norm": (bd - k["band_floor"]) / (k["band_nearest"] - k["band_floor"]),
           "band": bd, "cents_table": float(np.sum(E[live] * cents[live]) / E[live].sum()), "cents_max": float(cents[live].max())}
    if ref:
        Wr, Zr, _ = C.load_run(ref); fr = np.asarray(C.decode(Wr, Zr[c], v)["mode_freq"])
        out["cents_ref_max"] = float(np.abs(1200 * np.log2(f / fr))[live].max())
    return out


if __name__ == "__main__":
    with Pool(5) as pool:
        res = pool.map(one, rows)
    S = {k: float(np.median([r[k] for r in res])) for k in res[0]}
    S["cents_table_ew_mean"] = float(np.mean([r["cents_table"] for r in res])); S["cents_max_all"] = float(max(r["cents_max"] for r in res))
    if ref: S["cents_ref_max_all"] = float(max(r["cents_ref_max"] for r in res))
    S["both_below_0.5"] = float(np.mean([(r["mr_norm"] < 0.5) and (r["band_norm"] < 0.5) for r in res]))
    print(run, json.dumps({k: round(v, 4) for k, v in S.items()}))
    C.json_dump({"summary": S, "rows": res}, os.path.join(run, "recon.json"))
