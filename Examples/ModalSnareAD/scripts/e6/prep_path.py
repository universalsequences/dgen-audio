#!/usr/bin/env python3
"""Inputs for dgen `ModalSnareAD latent-path` (E6 path-regularised fine-tune):
- anchor rows: the 96 fit rows (code, velocity) with the E2 decoder's OWN sigmoid positions as targets
  (self-distillation: keeps E2's audio-stage reconstruction, not the teacher) and pretrain.json's weights,
- the range box (per-column [min, max] of the teacher positions over the fit rows, margin .02, as E2),
- per-code mode energy shares (constant weights for the optional slot-wise log-f linearity term),
- optional supervised targets on the wire columns (E7) if e7/wire_targets.json exists,
- the analytic band-envelope surrogate's constants (common.py SUR_*).
usage: prep_path.py RUN OUT.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
run, out = sys.argv[1], sys.argv[2]
W, Z, names = C.load_run(run)
pre = json.load(open(os.path.join(C.E2, "pretrain.json")))
L = C.tp.LAY; P = L["P"]
nfit = pre["nfit"]; snare = pre["snare"][:nfit]; vel = pre["vel"][:nfit]
pi_cols = np.zeros(P, bool)
for s in L["segments"]:
    if s["squash"] == "pi": pi_cols[s["offset"]: s["offset"] + s["width"]] = True
S = []
for c, v in zip(snare, vel):
    u = C.ln.decode_np(W, Z[c], v)
    S.append(1 / (1 + np.exp(-np.where(pi_cols, 2 * u, u))))
S = np.array(S)
w = np.array(pre["w"]).reshape(-1, P)[:nfit]
s_t = np.array(pre["s"]).reshape(-1, P)[:nfit]
lo = np.clip(s_t.min(0) - 0.02, 0, 1); hi = np.clip(s_t.max(0) + 0.02, 0, 1)
E = []
for c in range(32):
    p = C.decode(W, Z[c], 1.0); e = C.mode_energy(p); E.append(e / e.sum())
doc = {"names": names, "snare": snare, "vel": vel, "P": P, "s_anchor": S.reshape(-1).tolist(), "w": w.reshape(-1).tolist(),
       "box_lo": lo.tolist(), "box_hi": hi.tolist(), "mode_eshare": np.array(E).reshape(-1).tolist(),
       "sur": {"T": C.SUR_T.tolist(), "LC": C.SUR_LC.tolist(), "SIG": C.SUR_SIG, "NORM": C.SUR_NORM,
               "INV_WIDTH": (1 / C.SUR_WIDTH).tolist(), "FLOOR": C.SUR_FLOOR, "S": C.SUR_S.reshape(-1).tolist(),
               "NBM": len(C.SUR_LC), "SR": C.SR}}
# Wire columns (E7): the run's own decoded wire knobs are the anchor (E7 distilled them toward
# e7/wire_targets.json, strand-share weighted); E2 gave them weight 0, so give them the median weight.
ws = [s for s in L["segments"] if s["name"].startswith("wire_")]
med = float(np.median(w[w > 0]))
for s_ in ws: w[:, s_["offset"]: s_["offset"] + s_["width"]] = med
doc["w"] = w.reshape(-1).tolist(); doc["wire_weight"] = med
json.dump(doc, open(out, "w"))
print(f"wrote {out}: {nfit} anchor rows, P={P}")

# Null space of the anchor rows' readout features A = [h2 1] (96 x 129): W3 changes inside it leave every
# anchor output exactly unchanged (latent-path --readout-null).
A = []
for c, v in zip(snare, vel):
    x = np.array([[*Z[c], min(max(v, 0), 1)]])
    h = np.tanh(x @ W["l1.W"] + W["l1.b"]); h = np.tanh(h @ W["l2.W"] + W["l2.b"])[0]
    A.append(np.append(h, 1.0))
A = np.array(A); U_, s_, Vt_ = np.linalg.svd(A)
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
r = int(np.sum(s_ > tol * s_[0])); N = Vt_[r:].T  # [129, k]
doc["null"] = N.reshape(-1).tolist(); doc["null_k"] = N.shape[1]
json.dump(doc, open(out, "w"))
print(f"null space: rank {r} of {A.shape[0]} rows (tol {tol}), k = {N.shape[1]}; sing. values kept down to {s_[r-1]:.3e}, dropped from {s_[r] if r < len(s_) else 0:.3e}; anchor drift bound ~ {s_[r] if r < len(s_) else 0:.1e} x |dW|")
