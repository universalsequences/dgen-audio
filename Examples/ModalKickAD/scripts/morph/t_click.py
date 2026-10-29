"""Morph click check (e8 method): hop-H builds vs the hop-1 build (decodes every sample).
Fold metric: |2nd difference| / local RMS (window 4H), folded by n mod H over 0.5 s (first hop
skipped); score = mean at the 3 boundary positions / median over all H positions. Plus band-metric
distance hop-H vs hop-1 (out and modal tap), and a calibration: a random-sign 1e-4 step at every
boundary added to the hop-1 modal tap."""
import json, sys
from lib import *
MAD.use("kick")
from kick_metric import band_score
ref = load("build/h1tap")
def fold(x, H, n=24000):
    d = np.abs(x[2:n] - 2 * x[1:n - 1] + x[:n - 2]); d = np.concatenate([[0], d, [0]])
    loc = np.sqrt(np.convolve(d ** 2, np.ones(4 * H) / (4 * H), "same")) + 1e-12
    v = d / loc; m = (len(v) // H) * H; prof = v[H:m].reshape(-1, H).mean(0)
    return float(np.mean(prof[[H - 1, 0, 1]]) / np.median(prof))
pairs = [("K55", "K62"), ("K60", "K58"), ("K73", "K72"), ("K68", "K77")]
res = {}
for H in (128, 256):
    inst = load(f"build/h{H}tap")
    for A, B in pairs:
        for mt in (20.0, 80.0, 400.0):
            p = {**zp(CODE[A]), **zp(CODE[B], "b"), "morph": mt}
            y = render(inst, 1.0, params=p); r = render(ref, 1.0, params=p)
            row = dict(out_band=band_score(y[:, 0], r[:, 0]), modal_band=band_score(y[:, 2], r[:, 2]),
                       out_rel=relrms(y[:, 0], r[:, 0]),
                       out_fold=fold(y[:, 0], H), out_fold_ref=fold(r[:, 0], H),
                       modal_fold=fold(y[:, 2], H), modal_fold_ref=fold(r[:, 2], H))
            if H == 128 and mt == 80.0:
                rng = np.random.default_rng(0); st = np.zeros(len(r))
                for b0 in range(H, len(r), H): st[b0:] += rng.choice([-1, 1]) * 1e-4
                row["calib_step1e-4"] = fold(r[:, 2] + st, H) - fold(r[:, 2], H)
            key = f"h{H} {A}->{B} {mt:g}ms"; res[key] = row
            print(key, {k: round(v, 4) for k, v in row.items()}, flush=True)
json.dump(res, open("click.json", "w"), indent=1)
