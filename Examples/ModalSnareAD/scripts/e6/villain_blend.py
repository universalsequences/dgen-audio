#!/usr/bin/env python3
"""E6 task 1b: VILLAIN Snare's own A/B Blend knob, t = 0, .05, .., 1 (velocity 1, C4, fresh state each
render -> the same xorshift noise realisation at every t), same step metrics as measure_walks.py.
Also writes concatenated listening walks to renders/villain_walks/.
usage: villain_blend.py OUT.json [N_PAIRS]"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from villain_common import load, C4
from measure_walks import TS

import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e6")
out = sys.argv[1]; npairs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
names = json.load(open(os.path.join(C.R, "e2/runs/outbox_d4/codes.json")))["order"]
inst = load(stems=False)
rng = np.random.default_rng(66)
pairs = []
while len(pairs) < npairs:
    a, b = sorted(rng.choice(32, 2, replace=False).tolist())
    if (a, b) not in pairs: pairs.append((a, b))
os.makedirs(os.path.join(HERE, "renders", "villain_walks"), exist_ok=True)
rows = []
for a, b in pairs:
    ys = []
    for t in TS:
        y, _ = inst.render(1.0, pitch=C4, vel=1.0, params={"snare_a": a, "snare_b": b, "blend": float(t)})
        ys.append((y if y.ndim == 1 else y[:, 0]).astype(float))
    env = [C.band_env(y) for y in ys]
    steps = {"band": [C.band_from_env(env[i], env[i + 1]) for i in range(20)], "mr": [C.mr(ys[i], ys[i + 1]) for i in range(20)]}
    rows.append({"a": a, "b": b, "pair": f"{names[a]}-{names[b]}", "steps": steps,
                 "even": {k: C.evenness(v) for k, v in steps.items()},
                 "chord": {"band": C.band_from_env(env[0], env[-1]), "mr": C.mr(ys[0], ys[-1])},
                 "peak": [float(np.abs(y).max()) for y in ys]})
    C.write_walk_wav(os.path.join(HERE, "renders", "villain_walks", f"{names[a]}-{names[b]}.wav"), ys)
    e = rows[-1]["even"]["band"]
    print(f"{names[a]:>9s}-{names[b]:<9s} band steps {np.round(steps['band'], 2)}  first/mean {e['first_over_mean']:.2f} max/mean {e['max_over_mean']:.2f} cv {e['cv']:.2f}")
summ = {k: {"first_over_mean_median": float(np.median([r["even"][k]["first_over_mean"] for r in rows])),
            "max_over_mean_median": float(np.median([r["even"][k]["max_over_mean"] for r in rows])),
            "cv_median": float(np.median([r["even"][k]["cv"] for r in rows])),
            "first_step_over_chord_median": float(np.median([r["steps"][k][0] / r["chord"][k] for r in rows]))}
        for k in ("band", "mr")}
print(json.dumps(summ, indent=1))
C.json_dump({"pairs": pairs, "summary": summ, "rows": rows}, out)
