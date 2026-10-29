"""Morph click check (hop-H builds vs the 1-sample-hop reference, which decodes every sample).
Fold metric: |second difference| normalised by its local RMS (window 4H), folded by n mod H
over the first 0.5 s; score = mean at the 3 boundary positions / median over all H positions.
A click at hop boundaries gives score >> 1; the reference (no boundaries) gives the baseline.
Also band metric hop-H vs hop-1 (dB, E7 band-envelope metric) on out and head."""
import json
from lib8 import *
from metric import band_score
ref = Instrument("build/h1tap")
def fold(x, H, n=24000):
    d = np.abs(x[2:n] - 2 * x[1:n - 1] + x[:n - 2]); d = np.concatenate([[0], d, [0]])
    w = np.ones(4 * H) / (4 * H); loc = np.sqrt(np.convolve(d ** 2, w, "same")) + 1e-12
    v = d / loc; m = (len(v) // H) * H; prof = v[H:m].reshape(-1, H).mean(0)   # skip the first hop (strike)
    return float(np.mean(prof[[H - 1, 0, 1]]) / np.median(prof))
res = {}
pairs = (("SP01", "ZV"), ("SP20", "SP13"), ("SP06", "YYUUU"), ("SP12", "SP09"))
for H in (128, 256):
    inst = Instrument(f"build/h{H}tap")
    for (A, B) in pairs:
        for mt in (10.0, 50.0, 400.0):
            p = {**zp(CODES[A]), **zp(CODES[B], "b"), "morph": mt}
            y = render(inst, 1.0, params=p); r = render(ref, 1.0, params=p)
            row = dict(out_band_vs_hop1=band_score(y[:, 0], r[:, 0]), head_band_vs_hop1=band_score(y[:, 1], r[:, 1]),
                       head_fold=fold(y[:, 1], H), head_fold_ref=fold(r[:, 1], H), out_fold=fold(y[:, 0], H), out_fold_ref=fold(r[:, 0], H))
            key = f"h{H} {A}->{B} {mt:g}ms"; res[key] = row
            print(key, {k: round(v, 3) for k, v in row.items()}, flush=True)
json.dump(res, open("click.json", "w"), indent=1)
