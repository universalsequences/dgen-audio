"""Morph quality of every START->END pair at morph 300 ms (follow 1): envelope collapse below both
ends, peak overshoot, and the low-band peak at 250 ms vs the ends'."""
import json
from lib import *
from survey import env_db
m = load("Latent Kick Morph")
S = json.load(open("survey_singles.json"))
def lowpk(y, t, n=4096):
    i = int(t * SR); seg = y[i - n // 2:i + n // 2] * np.hanning(n)
    Sx = np.abs(np.fft.rfft(seg, 8 * n)); f = np.fft.rfftfreq(8 * n, 1 / SR); lo = (f > 20) & (f < 200)
    return float(f[lo][np.argmax(Sx[lo])]), float(20 * np.log10(Sx[lo].max() + 1e-9))
rows = []
for A in NAMES:
    for B in NAMES:
        if A == B: continue
        y = render(m, 1.0, params={**zp(CODE[A]), **zp(CODE[B], "b"), "morph": 300.0})[:, 0]
        e = env_db(y); eA = np.array(S[A]["env"]); eB = np.array(S[B]["env"])
        hi = int(100 * max(S[A]["t20"], S[B]["t20"])) + 1
        lo_env = np.minimum(eA, eB)[2:hi]; ok = lo_env > max(eA.max(), eB.max()) - 30
        collapse = float(np.max((lo_env - e[2:hi])[ok])) if ok.any() else 0.0
        rows.append(dict(A=A, B=B, collapse=collapse, over=float(np.abs(y).max() / max(S[A]["peak"], S[B]["peak"])),
                         sub250=lowpk(y, 0.25)[0], t20=float(np.argmax(e < e.max() - 20) * 0.01)))
json.dump(rows, open("survey_pairs.json", "w"))
c = np.array([r["collapse"] for r in rows]); o = np.array([r["over"] for r in rows])
print(len(rows), "pairs; collapse dB median %.1f p90 %.1f; share collapse<6dB %.2f; overshoot>1.1 %.2f" % (np.median(c), np.percentile(c, 90), (c < 6).mean(), (o > 1.1).mean()))
