"""Parity: morph build with START == END vs the installed per-hit Latent Kick (kick_a = kick_b = k,
blend 0). All 33 kicks at v1.0 / v0.4 + 3 knob settings. usage: t_parity.py [BUILD]"""
import json, sys
from lib import *
ref = load(INST); new = load(sys.argv[1] if len(sys.argv) > 1 else "build/h128")
rows = {}
def one(key, k, vel=1.0, kn=None, pitch=C4):
    n = NAMES[k]; kn = kn or {}
    a = render(ref, 1.0, vel, {"kick_a": k, "kick_b": k, **kn}, pitch)[:, 0]
    b = render(new, 1.0, vel, {**zp(CODE[n]), **zp(CODE[n], "b"), **kn}, pitch)[:, 0]
    rows[key] = r = relrms(b, a); print(key, "%.3e" % r, flush=True)
for k in range(33):
    for vel in (1.0, 0.4):
        one(f"{NAMES[k]} v{vel}", k, vel)
for kn in ({"tune": 1.4}, {"decay": 0.5, "bend": 1.8}, {"spread": 1.3, "tilt": 0.8, "shape": 2.0}):
    one(f"K63 {kn} p0.75", 11, 0.8, kn, C4 * 0.75)
v = np.array(list(rows.values()))
print("rel RMS median %.2e  p90 %.2e  max %.2e (%s)" % (np.median(v), np.percentile(v, 90), v.max(), max(rows, key=rows.get)))
json.dump(rows, open("parity.json", "w"), indent=1)
