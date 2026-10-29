#!/usr/bin/env python3
"""Exact re-anchoring after the path fine-tune: with W1/W2 (and the codes) as fine-tuned, the output
layer is linear in W3/b3, so the minimum-norm change [dW3; db3] = pinv([h2 1]) (U_base - U_ft) makes
the decoder reproduce the BASE decoder's outputs u at all 96 fit rows (32 codes x 3 velocities) to
float precision. Reconstruction (and E5's frequency accuracy, and E7's wire knobs) is then exactly the
base run's at the training snares; only the in-between geometry is the fine-tuned one.
usage: reanchor.py BASE_RUN FT_RUN OUT_RUN"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
base, ft, out = sys.argv[1:4]
Wb, Zb, names = C.load_run(base); Wf, Zf, _ = C.load_run(ft)
pre = json.load(open(os.path.join(C.E2, "pretrain.json"))); n = pre["nfit"]
rows = list(zip(pre["snare"][:n], pre["vel"][:n]))


def h2(W, z, v):
    x = np.array([[*z, min(max(v, 0), 1)]])
    h = np.tanh(x @ W["l1.W"] + W["l1.b"]); return np.tanh(h @ W["l2.W"] + W["l2.b"])[0]


A = np.array([np.append(h2(Wf, Zf[c], v), 1.0) for c, v in rows])
Ub = np.array([C.ln.decode_np(Wb, Zb[c], v) for c, v in rows])
Uf = np.array([C.ln.decode_np(Wf, Zf[c], v) for c, v in rows])
X = np.linalg.pinv(A, rcond=1e-10) @ (Ub - Uf)
W = dict(Wf); W["l3.W"] = Wf["l3.W"] + X[:-1]; W["l3.b"] = Wf["l3.b"] + X[-1:]
Ur = np.array([C.ln.decode_np(W, Zf[c], v) for c, v in rows])
print(f"before: max |u_ft - u_base| {np.abs(Uf - Ub).max():.3e}; after re-anchor {np.abs(Ur - Ub).max():.3e}; "
      f"|dW3| {np.linalg.norm(X[:-1]):.3e} vs |W3| {np.linalg.norm(Wf['l3.W']):.3e}; rank {np.linalg.matrix_rank(A)}")
os.makedirs(out, exist_ok=True)
w = json.load(open(os.path.join(ft, "weights.json")))
w["l3"]["W"]["data"] = W["l3.W"].reshape(-1).tolist(); w["l3"]["b"]["data"] = W["l3.b"].reshape(-1).tolist()
json.dump(w, open(os.path.join(out, "weights.json"), "w"))
shutil.copy(os.path.join(ft, "codes.json"), out)
ck = json.load(open(os.path.join(ft, "checkpoint.json")))
ck["tensors"]["W3"] = w["l3"]["W"]["data"]; ck["tensors"]["b3"] = w["l3"]["b"]["data"]
json.dump(ck, open(os.path.join(out, "checkpoint.json"), "w"))
json.dump({"base": os.path.abspath(base), "ft": os.path.abspath(ft), "max_u_err_before": float(np.abs(Uf - Ub).max()),
           "max_u_err_after": float(np.abs(Ur - Ub).max()), "dW3_rel": float(np.linalg.norm(X[:-1]) / np.linalg.norm(Wf["l3.W"]))},
          open(os.path.join(out, "reanchor.json"), "w"), indent=1)
