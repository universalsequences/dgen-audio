"""Polynomial fits P_k(r) ~ J_m(j_mn r), r in [0, 1], for the first 24 membrane modes, so the
strike-position amplitude |J_m(j_mn r_s)| is a differentiable polynomial in the dgen graph.
Writes bessel_poly.json: {modes: [{m, n, ratio, coeffs (ascending, degree 12)}], max_abs_err}."""
import json, os
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1/physical")
tau = np.linspace(0, np.pi, 4001)
def J(m, x):
    x = np.atleast_1d(x)[:, None]
    return np.trapezoid(np.cos(m * tau[None, :] - x * np.sin(tau[None, :])), tau, axis=1) / np.pi
modes = json.load(open(os.path.join(HERE, "bessel_modes.json")))["modes"][:24]
r = np.linspace(0, 1, 401)
out, worst = [], 0
for d in modes:
    y = J(d["m"], d["j"] * r)
    c = np.polynomial.polynomial.polyfit(r, y, 12)
    err = np.abs(np.polynomial.polynomial.polyval(r, c) - y).max()
    worst = max(worst, err)
    out.append({"m": d["m"], "n": d["n"], "ratio": d["ratio"], "coeffs": c.tolist()})
json.dump({"modes": out, "max_abs_err": worst, "degree": 12}, open(os.path.join(HERE, "bessel_poly.json"), "w"), indent=1)
print("max abs err", worst)
