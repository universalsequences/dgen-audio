#!/usr/bin/env python3
"""Convert trained decoder weights into the files the Latent Snare dsp.lisp loads.

Input JSON (what the Swift/DGenLazy trainer should write):
    {"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}, "l3": {"W": W3, "b": b3}}
where each array is either nested lists, or {"shape": [...], "data": [flat row-major]}
(i.e. `Tensor.getData()` plus its shape). Orientation is DGenLazy's
`x.matmul(W)`: W is [in, out] (5x64, 64x64, 64xP); b is [out] or [1, out].
A PyTorch nn.Linear weight is [out, in]: pass --torch to transpose it.
Layer keys may also be any three keys in file order (e.g. "dense0", ...).

Output (into the instrument folder): w1/b1/w2/b2/w3/b3.json, each
{"shape": [r, c], "data": [...row-major...]}. dsp.lisp reads them with
(tensor @shape [r c] @file "w1.json"); DGenLisp stores tensors row-major for
matmul and for the shrink-row views the generated decoder uses.

    python3 export_weights.py trained.json "Latent Snare"   # then re-run the eseq check
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e3proto")


def arr(v):
    if isinstance(v, dict):
        return np.asarray(v["data"], np.float64).reshape(v["shape"])
    return np.asarray(v, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dest")
    ap.add_argument("--torch", action="store_true", help="weights are [out,in] (nn.Linear)")
    ap.add_argument("--layout", default=os.path.join(MAD.code("e3proto"), "decoder_layout.json"))
    a = ap.parse_args()
    lay = json.load(open(a.layout))
    src = json.load(open(a.src))
    keys = [k for k in ("l1", "l2", "l3") if k in src] or list(src.keys())
    if len(keys) != 3:
        sys.exit(f"expected 3 layers, got {list(src.keys())}")
    for i, k in enumerate(keys, 1):
        W = arr(src[k]["W"]); b = arr(src[k]["b"]).reshape(1, -1)
        if a.torch:
            W = W.T
        wspec = lay["weights"][f"l{i}.W"]; bspec = lay["weights"][f"l{i}.b"]
        if list(W.shape) != wspec["shape"]:
            hint = " (looks transposed: pass --torch)" if list(W.T.shape) == wspec["shape"] else ""
            sys.exit(f"layer {k}: W shape {list(W.shape)} != expected {wspec['shape']}{hint}")
        if list(b.shape) != bspec["shape"]:
            sys.exit(f"layer {k}: b shape {list(b.shape)} != expected {bspec['shape']}")
        if not (np.isfinite(W).all() and np.isfinite(b).all()):
            sys.exit(f"layer {k}: non-finite weights")
        for t, spec in ((W, wspec), (b, bspec)):
            with open(os.path.join(a.dest, spec["file"]), "w") as f:
                json.dump({"shape": spec["shape"], "data": [round(float(x), 7) for x in t.reshape(-1)]}, f)
        print(f"{k}: W {list(W.shape)} -> {wspec['file']}, b {list(b.shape)} -> {bspec['file']}")
    print(f"P = {lay['P']} outputs; segment layout in {a.layout}")


if __name__ == "__main__":
    main()
