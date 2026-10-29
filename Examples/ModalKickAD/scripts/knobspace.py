#!/usr/bin/env python3
"""Re-express a run's codes as PCA-rotated knob coordinates (z1 = largest axis), scaled so
max |z| = 2.5 (inside the instrument's [-3, 3] clip); the transform is absorbed exactly into
W1/b1. usage: knobspace.py RUN OUT"""
import json, os, sys, shutil
import numpy as np
from train_np import load, save, forward
run, out = sys.argv[1], sys.argv[2]
W, Z, ids = load(run)
mu = Z.mean(0); C = Z - mu
_, _, Vt = np.linalg.svd(C, full_matrices=False)
R = Vt.T                                  # columns = principal axes
K = C @ R
s = np.abs(K).max() / 2.5
Kz = K / s                                # knob codes
W2 = dict(W)
W2["W1"] = s * R.T @ W["W1"]
W2["b1"] = mu @ W["W1"] + W["b1"]
assert np.allclose(forward(W2, Kz)[2], forward(W, Z)[2], atol=1e-6)
c = json.load(open(os.path.join(run, "codes.json")))
save(W2, Kz, ids, out, {**c.get("meta", {}), "knobspace": {"from": run, "scale": float(s)}})
json.dump({"std": Kz.std(0).tolist(), "mean": Kz.mean(0).tolist()}, open(os.path.join(out, "codes_std.json"), "w"))
print("knob codes std", np.round(Kz.std(0), 3), "max", np.round(np.abs(Kz).max(0), 3))
