#!/usr/bin/env python3
"""Does the analytic band-envelope surrogate (common.surrogate, no audio) track the real band-envelope
metric on rendered walks? Per-step distances along N random walks: Pearson/Spearman of surrogate vs
band metric (full output), and of per-pair evenness (CV, max/mean).
usage: validate_surrogate.py RUN [N]"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
run = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W, Z, names = C.load_run(run)
pairs = list(itertools.combinations(range(32), 2)); rng = np.random.default_rng(11)
pairs = [pairs[i] for i in rng.choice(len(pairs), N, replace=False)]
TS = np.arange(21) * 0.05


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


allb, alls, cvb, cvs, mmb, mms = [], [], [], [], [], []
for a, b in pairs:
    envs, surs = [], []
    for t in TS:
        p, y, h, n = C.render(W, (1 - t) * Z[a] + t * Z[b])
        envs.append(C.band_env(y)); surs.append(C.surrogate(p))
    sb = np.array([C.band_from_env(envs[i], envs[i + 1]) for i in range(20)])
    ss = np.array([C.sur_dist(surs[i], surs[i + 1]) for i in range(20)])
    allb += list(sb); alls += list(ss)
    cvb.append(sb.std() / sb.mean()); cvs.append(ss.std() / ss.mean()); mmb.append(sb.max() / sb.mean()); mms.append(ss.max() / ss.mean())
    print(f"{names[a]:>9s}-{names[b]:<9s} r={np.corrcoef(sb, ss)[0, 1]:+.2f} cv {cvb[-1]:.2f}/{cvs[-1]:.2f}")
allb, alls = np.array(allb), np.array(alls)
print(f"steps: pearson {np.corrcoef(allb, alls)[0, 1]:.3f} spearman {np.corrcoef(rank(allb), rank(alls))[0, 1]:.3f}; scale sur/band {np.median(alls / allb):.2f}")
print(f"per-pair CV: pearson {np.corrcoef(cvb, cvs)[0, 1]:.3f}; max/mean pearson {np.corrcoef(mmb, mms)[0, 1]:.3f}")
