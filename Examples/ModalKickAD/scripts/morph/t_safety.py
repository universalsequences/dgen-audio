"""Blow-up sweep: every START->END pair at morph 40 / 300 / 2000 ms (curve -0.6 / 0 / 0.6, follow 1 / 0):
max |out| and finiteness (out_clip would show as ~100)."""
import json
from lib import *
m = load("Latent Kick Morph")
worst = []; n = 0
for A in NAMES:
    for B in NAMES:
        if A == B: continue
        for mt, cv, fo in ((40.0, -0.6, 1.0), (300.0, 0.0, 0.0), (2000.0, 0.6, 1.0)):
            y = render(m, 1.0, params={**zp(CODE[A]), **zp(CODE[B], "b"), "morph": mt, "curve": cv, "follow": fo})[:, 0]
            pk = float(np.abs(y).max()) if np.isfinite(y).all() else float("inf"); n += 1
            worst.append((pk, A, B, mt))
worst.sort(reverse=True)
print(n, "renders; worst peaks:", worst[:6])
json.dump(worst[:50], open("safety.json", "w"))
