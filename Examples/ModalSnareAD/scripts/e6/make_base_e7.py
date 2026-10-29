#!/usr/bin/env python3
"""E6 starting point: the E2 trainer checkpoint (e2/runs/outbox_d4) with E7's distilled readout
(e7/weights_wires_rebal.json: wire-knob and band_amp columns of W3/b3 refit) -> e6/runs/base_e7/
(checkpoint.json for `ModalSnareAD latent-path --init`, weights.json + codes.json for the numpy side)."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402

src = os.path.join(MAD.ROOT, "e2", "runs", "outbox_d4")
wpath = os.path.join(MAD.ROOT, "e7", "weights_wires_rebal.json")
dst = os.path.join(MAD.stage("e6"), "runs", "base_e7")
os.makedirs(dst, exist_ok=True)
w = json.load(open(wpath))
ck = json.load(open(os.path.join(src, "checkpoint.json")))
W3 = np.array(ck["tensors"]["W3"]); W3n = np.array(w["l3"]["W"]["data"])
print("W3 changed cols", np.nonzero(np.abs(W3n - W3).reshape(128, 360).max(0) > 0)[0])
ck["tensors"]["W3"] = w["l3"]["W"]["data"]; ck["tensors"]["b3"] = w["l3"]["b"]["data"]
print("l1 same", np.allclose(np.array(w["l1"]["W"]["data"][:512]), ck["tensors"]["W1z"]))
json.dump(ck, open(os.path.join(dst, "checkpoint.json"), "w"))
shutil.copy(wpath, os.path.join(dst, "weights.json"))
shutil.copy(os.path.join(src, "codes.json"), os.path.join(dst, "codes.json"))
print("wrote", dst)
