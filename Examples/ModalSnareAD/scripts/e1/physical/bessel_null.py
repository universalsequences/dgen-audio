"""Null baseline for table_vs_bessel.py: same f0 search on random log-uniform mode frequencies
(same per-snare count, range, energies). If real tables are not closer to Bessel ratios than
random sets, the membrane-ratio prior has no support in the data."""
import json, os, sys
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 64
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1/physical")
R = np.array([d["ratio"] for d in json.load(open(os.path.join(HERE, "bessel_modes.json")))["modes"]])[:NR]
d = json.load(open(os.path.join(HERE, "../../data/villain_modes.json")))["renders"]
TUNE = 2 * 261.6255653 / 440
grid = np.exp(np.linspace(np.log(40), np.log(2000), 3000))
def score(f, w):
    best = 9
    for f0 in grid:
        ratios = f / f0
        dist = np.min(np.abs(np.log2(ratios[:, None] / R[None, :])), axis=1)
        dist = np.where(ratios < 0.97, 1.0, dist)
        best = min(best, float((w * dist).sum()))
    return 1200 * best
rng = np.random.default_rng(1)
real, null = [], []
for i in range(32):
    s = d[f"snare_{i:02d}_v100"]["slots"]
    f = np.array([m["f"] for m in s]) * TUNE
    amp = np.array([m["amp"] for m in s]); dec = np.array([m["dec0"] for m in s])
    E = amp ** 2 * dec / 2
    keep = (f > 40) & (E > 0.005 * E.sum())
    f, E = f[keep], E[keep]; w = E / E.sum()
    real.append(score(f, w))
    lo, hi = np.log(f.min()), np.log(f.max())
    null.append(np.median([score(np.exp(rng.uniform(lo, hi, len(f))), w) for _ in range(5)]))
real, null = np.array(real), np.array(null)
print("ratios", NR, "median cents real", np.median(real), "null", np.median(null), "real<null in", int((real < null).sum()), "/ 32")
json.dump({"real_cents": real.tolist(), "null_cents": null.tolist()}, open(os.path.join(HERE, f"bessel_null_{NR}.json"), "w"))
