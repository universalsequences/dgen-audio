#!/usr/bin/env python3
"""Turn per-snare latent codes into the Latent Snare map sidecar.

    python3 build_family.py CODES.json [--inst "e4/Latent Snare"] [--axes 1,2]
                            [--asset-path "instruments/<install>/data/family.json"]

CODES.json is the E2 hand-off shape: {"SP01": [z1, z2, z3, z4], ...} (a
{"codes": {...}} wrapper is accepted too). Writes

- <inst>/data/family.json: the `family-map` widget's sidecar (version 1).
  `scores` are the RAW latent coordinates of each snare on the two map axes
  plus a third (dot size), so the map is the latent space itself, not a PCA
  picture of it: the ring the panel draws at (z_x, z_y) is exactly where the
  decoder is being asked to decode. `lattice.members` holds each snare's real
  peak envelope (numpy twin of the voice, scripts/e3proto/reference_render.py, at its
  code, vel 1, knobs at default), which the widget's preview strip shows for
  the nearest snare. Extra keys (`codes`, `dims`) are ignored by the widget.
- <inst>/ui.lisp: rewrites the block between ";; BEGIN GENERATED LATENT
  CODES" and ";; END GENERATED LATENT CODES" (the panel needs the codes to
  set z1..z4 when a dot is clicked: eseqlisp has no JSON reader).

--axes picks which latent dims are the map's x / y (1-based). "auto" (the
default) picks the two dims with the largest spread across the codes; the
remaining two become the panel's knobs, the first of them also sizes the dots.
"""
import argparse, json, math, os, re, sys

import numpy as np

import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e4")
MAD.use("e3proto")
import reference_render as rr  # noqa: E402

BEGIN = ";; BEGIN GENERATED LATENT CODES"
END = ";; END GENERATED LATENT CODES"
ENV_SECONDS = 0.45
ENV_POINTS = 140


def load_codes(path):
    raw = json.load(open(path))
    if isinstance(raw, dict) and "codes" in raw and isinstance(raw["codes"], dict):
        raw = raw["codes"]
    names = sorted(raw)
    codes = np.array([[float(v) for v in raw[n]] for n in names], np.float64)
    if codes.ndim != 2 or codes.shape[1] != 4:
        sys.exit(f"{path}: expected 4 latent values per snare, got shape {codes.shape}")
    if not np.all(np.isfinite(codes)):
        sys.exit(f"{path}: non-finite latent code")
    if np.any(np.abs(codes) > 3):
        print(f"warning: codes outside the z param range [-3, 3] are clipped by the DSP", file=sys.stderr)
    return names, codes


def peak_envelope(y, sr=48000.0):
    n = int(ENV_SECONDS * sr)
    y = np.abs(y[:n])
    edges = np.linspace(0, n, ENV_POINTS + 1).astype(int)
    env = np.array([y[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    peak = env.max()
    return env / peak if peak > 0 and math.isfinite(peak) else env


def lisp_block(names, codes, dims, asset_path):
    knob_dims = [d for d in range(4) if d not in dims[:2]]
    rows = "\n".join(
        f'    ("{n}" {" ".join(f"{v:.5f}" for v in c)})' for n, c in zip(names, codes))
    return "\n".join([
        BEGIN,
        ";; Written by scripts/e4/build_family.py from the codes JSON; do not edit by hand.",
        f'(def ls-family-file () "{asset_path}")',
        f'(def ls-x-param () "z{dims[0] + 1}")',
        f'(def ls-y-param () "z{dims[1] + 1}")',
        f"(def ls-knob-params () '({' '.join(f'\"z{d + 1}\"' for d in knob_dims)}))",
        ";; (name z1 z2 z3 z4), in family.json member order.",
        "(def ls-codes ()",
        "  '(",
        rows + "))",
        END,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes")
    ap.add_argument("--inst", default=os.path.join(HERE, "Latent Snare"))
    ap.add_argument("--weights", default=None, help="weights dir (default: --inst)")
    ap.add_argument("--axes", default="auto", help='"auto" or "x,y" latent dims, 1-based')
    ap.add_argument("--asset-path", default="instruments/research-latent-snare-e4/data/family.json",
                    help="family.json path as the widget resolves it (relative to an instrument library root)")
    a = ap.parse_args()

    names, codes = load_codes(a.codes)
    if a.axes == "auto":
        order = list(np.argsort(-codes.std(axis=0), kind="stable"))
        xy = sorted(int(d) for d in order[:2])
    else:
        xy = [int(v) - 1 for v in a.axes.split(",")]
        assert len(xy) == 2 and all(0 <= d < 4 for d in xy) and xy[0] != xy[1], a.axes
    rest = [d for d in range(4) if d not in xy]
    dims = xy + rest  # x, y, size, (last knob)

    W = rr.load_weights(a.weights or a.inst)
    envs = []
    for c in codes:
        y = rr.render_ref(W, list(np.clip(c, -3, 3)), vel=1.0, seconds=ENV_SECONDS + 0.01)
        envs.append([round(float(v), 4) for v in peak_envelope(y)])

    family = {
        "version": 1,
        "voice": "latent-snare",
        "axes": [f"z{d + 1}" for d in dims[:3]],
        "names": names,
        "scores": [[round(float(c[d]), 5) for d in dims[:3]] for c in codes],
        "codes": [[round(float(v), 5) for v in c] for c in codes],
        "dims": {"x": dims[0] + 1, "y": dims[1] + 1, "size": dims[2] + 1},
        "lattice": {"members": envs, "pairs": {}},
    }
    os.makedirs(os.path.join(a.inst, "data"), exist_ok=True)
    out = os.path.join(a.inst, "data", "family.json")
    with open(out, "w") as f:
        json.dump(family, f, separators=(",", ":"))
        f.write("\n")

    ui = os.path.join(a.inst, "ui.lisp")
    text = open(ui).read()
    block = lisp_block(names, codes, dims, a.asset_path)
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(text):
        sys.exit(f"{ui}: no generated-codes block ({BEGIN} ... {END})")
    open(ui, "w").write(pat.sub(lambda _: block, text))
    print(f"{out}: {len(names)} snares, map x=z{dims[0]+1} y=z{dims[1]+1} size=z{dims[2]+1}; "
          f"knobs {', '.join(f'z{d+1}' for d in rest)}; ui.lisp block rewritten")


if __name__ == "__main__":
    main()
