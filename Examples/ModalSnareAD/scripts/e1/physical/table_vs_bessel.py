"""How Bessel-like are VILLAIN's per-snare head modes? (numpy, table-only)

For each fit snare (v100): mode energies E_k = amp^2 * dec0/2 (analytic energy of a decaying sine),
modes above 40 Hz with >0.5% of energy. Search f0 on a fine log grid; score = energy-weighted mean
|log2(f_k / (f0 * r_nearest))| in cents. Also fit 1/dec = a + b f^2 (energy-weighted LSQ in log space)
and report its log-residual. Writes table_vs_bessel.json."""
import json, os
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1/physical")
R = np.array([d["ratio"] for d in json.load(open(os.path.join(HERE, "bessel_modes.json")))["modes"]])
d = json.load(open(os.path.join(HERE, "../../data/villain_modes.json")))["renders"]
TUNE = 2 * 261.6255653 / 440
rows = []
for i in range(32):
    s = d[f"snare_{i:02d}_v100"]["slots"]
    f = np.array([m["f"] for m in s]) * TUNE
    amp = np.array([m["amp"] for m in s]); dec = np.array([m["dec0"] for m in s])
    E = amp ** 2 * dec / 2
    keep = (f > 40) & (E > 0.005 * E.sum())
    f, E, dec = f[keep], E[keep], dec[keep]
    w = E / E.sum()
    best = None
    for f0 in np.exp(np.linspace(np.log(40), np.log(2000), 6000)):
        ratios = f / f0
        dist = np.min(np.abs(np.log2(ratios[:, None] / R[None, :])), axis=1)
        dist = np.where(ratios < 0.97, 1.0, dist)  # modes far below f0 cannot be membrane modes
        sc = float((w * dist).sum())
        if best is None or sc < best[0]: best = (sc, f0, dist)
    sc, f0, dist = best
    # damping law
    A = np.stack([np.ones_like(f), f ** 2], 1)
    y = 1 / dec
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    pred = np.maximum(A @ coef, 1e-3)
    dres = float(np.sqrt((w * np.log(y / pred) ** 2).sum()))
    rows.append({"snare": i, "modes": int(keep.sum()), "f0": float(f0), "mean_cents": 1200 * sc,
                 "frac_energy_within_25c": float(w[dist * 1200 < 25].sum()),
                 "decay_law_a": float(coef[0]), "decay_law_b": float(coef[1]), "decay_log_rms_resid": dres})
    print(f"snare {i:2d} modes {keep.sum():2d} f0 {f0:7.1f} Hz  mean |dev| {1200*sc:6.1f} c  E<25c {rows[-1]['frac_energy_within_25c']:.2f}  decay-law log-rms {dres:.2f}")
json.dump(rows, open(os.path.join(HERE, "table_vs_bessel.json"), "w"), indent=1)
print("median mean-cents", np.median([r["mean_cents"] for r in rows]), "median E<25c", np.median([r["frac_energy_within_25c"] for r in rows]),
      "median decay resid", np.median([r["decay_log_rms_resid"] for r in rows]))
