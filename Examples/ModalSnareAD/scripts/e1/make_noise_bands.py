"""Precompute the fixed band-noise signals N_b(t) used by ModalSnareAD's noise model.

The filter bank is DATA (spec JSON), so the exact realtime filter spec (scripts/e3proto/NOISE_BANDS.md)
can be dropped in later and the bands regenerated; trained envelopes transfer as long as the
realtime instrument uses the same filters and the same per-band normalisation `norm`.

Model contract (same offline and realtime):
    w(t)   = one white noise source, zero-mean unit-variance (offline: Gaussian, seed below)
    N_b(t) = norm_b * cascade(stages x biquad_b)(w)(t)        # unit-RMS band noise
    norm_b = 1 / sqrt(sum_n h_b[n]^2)  (h_b = impulse response of the cascade; analytic, not empirical)
Biquad (RBJ cookbook, band-pass, constant 0 dB peak gain), Direct Form I/II, a0-normalised:
    w0 = 2 pi fc / sr, alpha = sin(w0) / (2 Q)
    b = [alpha, 0, -alpha] / (1 + alpha);  a = [1, -2 cos(w0) / (1 + alpha), (1 - alpha) / (1 + alpha)]
Offline signals are rendered after a `preroll` of noise so N_b is stationary from t = 0
(matches a free-running realtime filter bank).

usage: python3 make_noise_bands.py [--spec spec.json] [--write-default-spec]
"""
import argparse, json, os
import numpy as np

import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1")
OUT = os.path.join(HERE, "noise_bands")

def default_spec():
    lo, hi, n = 100.0, 18000.0, 16
    fcs = [float(lo * (hi / lo) ** (i / (n - 1))) for i in range(n)]
    return {
        "doc": "ModalSnareAD band-noise filter bank (see make_noise_bands.py header for the exact contract)",
        "sample_rate": 48000, "samples": 48000, "preroll": 8192, "seed": 20261028,
        "source": "gaussian_unit_variance", "filter": "rbj_bpf_0db_peak", "stages": 2,
        "bands": [{"fc": fc, "q": 1.8} for fc in fcs],
    }

def biquad_coeffs(fc, q, sr):
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    a0 = 1 + alpha
    b = np.array([alpha, 0.0, -alpha]) / a0
    a = np.array([1.0, -2 * np.cos(w0) / a0, (1 - alpha) / a0])
    return b, a

def lfilter(b, a, x):
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(x)):
        xn = x[n]
        yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, xn, y1, yn
        y[n] = yn
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(OUT, "spec.json"))
    ap.add_argument("--write-default-spec", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.write_default_spec or not os.path.exists(args.spec):
        json.dump(default_spec(), open(args.spec, "w"), indent=1)
    spec = json.load(open(args.spec))
    sr, n, pre = spec["sample_rate"], spec["samples"], spec["preroll"]
    rng = np.random.default_rng(spec["seed"])
    w = rng.standard_normal(n + pre)
    try:
        from scipy.signal import lfilter as sp_lfilter
    except Exception:
        sp_lfilter = None
    bands = np.zeros((len(spec["bands"]), n), dtype=np.float32)
    imp = np.zeros(1 << 16); imp[0] = 1
    for i, bd in enumerate(spec["bands"]):
        b, a = biquad_coeffs(bd["fc"], bd["q"], sr)
        filt = (lambda x: sp_lfilter(b, a, x)) if sp_lfilter else (lambda x: lfilter(b, a, x))
        y, h = w, imp
        for _ in range(spec["stages"]):
            y, h = filt(y), filt(h)
        norm = 1.0 / np.sqrt(np.sum(h * h))
        bd["b"], bd["a"], bd["norm"] = b.tolist(), a.tolist(), float(norm)
        bands[i] = (norm * y[pre:]).astype(np.float32)
        bd["empirical_rms"] = float(np.sqrt(np.mean(bands[i].astype(np.float64) ** 2)))
    spec["layout"] = "bands.f32: float32 little-endian, row-major [n_bands, samples]"
    json.dump(spec, open(os.path.join(OUT, "bands.json"), "w"), indent=1)
    bands.tofile(os.path.join(OUT, "bands.f32"))
    for bd in spec["bands"]:
        print(f"fc {bd['fc']:8.1f}  q {bd['q']:.2f}  norm {bd['norm']:.3f}  rms {bd['empirical_rms']:.3f}")

if __name__ == "__main__":
    main()
