"""Circular-membrane (ideal, fixed edge) mode table: zeros j_{m,n} of J_m, sorted.
Writes bessel_modes.json: [{m, n, j, ratio}] with ratio = j / j_{0,1}. numpy only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import modal_ad as MAD  # noqa: E402

tau = np.linspace(0, np.pi, 4001)
def J(m, x):
    x = np.atleast_1d(x)[:, None]
    return np.trapezoid(np.cos(m * tau[None, :] - x * np.sin(tau[None, :])), tau, axis=1) / np.pi

def zeros(m, count, xmax=60.0):
    xs = np.linspace(max(0.5, float(m)), xmax, 6000)
    v = J(m, xs)
    out = []
    for i in range(len(xs) - 1):
        if v[i] * v[i + 1] < 0 and min(abs(v[i]), abs(v[i + 1])) < 0.2 and max(abs(v[i]), abs(v[i + 1])) > 1e-9:
            a, b = xs[i], xs[i + 1]
            for _ in range(60):
                c = 0.5 * (a + b)
                if J(m, a)[0] * J(m, c)[0] <= 0: b = c
                else: a = c
            out.append(0.5 * (a + b))
            if len(out) == count: break
    return out

modes = []
for m in range(0, 14):
    for n, j in enumerate(zeros(m, 8), start=1):
        modes.append({"m": m, "n": n, "j": float(j)})
modes.sort(key=lambda d: d["j"])
j01 = modes[0]["j"]
for d in modes: d["ratio"] = d["j"] / j01
modes = modes[:64]
json.dump({"doc": "ideal circular membrane, fixed edge: f_mn = f0 * j_mn / j_01; m>0 modes are degenerate pairs (cos/sin m theta)",
           "modes": modes}, open(os.path.join(MAD.stage("e1/physical"), "bessel_modes.json"), "w"), indent=1)
print([round(d["ratio"], 3) for d in modes[:24]])
