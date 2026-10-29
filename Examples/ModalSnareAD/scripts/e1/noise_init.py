"""Data-driven init of the band-noise envelopes (ModalSnareAD raw param format).

Noise residual per snare = out - out_headonly (v100 stems). For each band b of the filter bank
(noise_bands/bands.json), band power per STFT frame is the residual power weighted by the band's
normalised power response G_b(f) = norm_b^2 |H_b(f)|^(2*stages); e_b(t) = sqrt(P_b(t)/sum G_b).
Then a coarse grid fit of e_b(t) ~ A ((1-L) exp(-t/d1) + L exp(-t/d2)) in log space.
Writes noise_init.json: {"raw": {noise_amp, noise_d1, noise_tail, noise_d2: [B*NB] row-major}}.
Used only as an initialisation; the fit itself is against out.wav."""
import json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e1")
MAD.use("dataset")
from villain_common import read_wav_f32
spec = json.load(open(os.path.join(HERE, "noise_bands", "bands.json")))
SR, NFFT, HOP = spec["sample_rate"], 1024, 240
freqs = np.fft.rfftfreq(NFFT, 1 / SR)
z = np.exp(-1j * 2 * np.pi * freqs / SR)
G = []
for bd in spec["bands"]:
    b, a = np.array(bd["b"]), np.array(bd["a"])
    H = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    G.append(bd["norm"] ** 2 * np.abs(H) ** (2 * spec["stages"]))
G = np.array(G)  # [NB, F]
win = np.hanning(NFFT)
def lm_inv(x, lo, hi):
    p = np.clip((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
d1g = np.exp(np.linspace(np.log(0.003), np.log(0.3), 18)); d2g = np.exp(np.linspace(np.log(0.03), np.log(1.5), 16)); Lg = np.array([0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7])
raw = {k: [] for k in ["noise_amp", "noise_d1", "noise_tail", "noise_d2"]}
for i in range(32):
    rid = f"snare_{i:02d}_v100"
    out = read_wav_f32(os.path.join(HERE, "..", "data/villain", rid + ".wav")).astype(np.float64)
    ho = read_wav_f32(os.path.join(HERE, "..", "data/villain/stems", rid + ".out_headonly.wav")).astype(np.float64)
    r = out - ho
    nfr = 1 + (len(r) - NFFT) // HOP
    P = np.array([np.abs(np.fft.rfft(r[j * HOP:j * HOP + NFFT] * win)) ** 2 for j in range(nfr)])  # [T, F]
    P *= 2 / (win ** 2).sum()  # power spectral density scaling ~ per-sample variance
    t = (np.arange(nfr) * HOP + NFFT / 2) / SR
    for bnd in range(len(G)):
        e = np.sqrt(np.maximum(P @ G[bnd] / (G[bnd] ** 2).sum() * 1.0, 1e-14))  # rough band RMS
        le = np.log(e + 1e-7); wgt = np.maximum(le - le.max() + 8, 0)  # ignore >8 nepers below peak
        best = (1e9, 0.01, 0.03, 0.1, 0.2)
        for d1 in d1g:
            for d2 in d2g:
                if d2 <= d1: continue
                for L in Lg:
                    shape = (1 - L) * np.exp(-t / d1) + L * np.exp(-t / d2)
                    lA = np.sum(wgt * (le - np.log(shape))) / max(wgt.sum(), 1e-9)
                    err = np.sum(wgt * (le - lA - np.log(shape)) ** 2)
                    if err < best[0]: best = (err, float(np.exp(lA)), d1, d2, L)
        _, A, d1, d2, L = best
        raw["noise_amp"].append(float(lm_inv(min(max(A, 2e-5), 1.9), 1e-5, 2.0)))
        raw["noise_d1"].append(float(lm_inv(d1, 0.002, 0.4)))
        raw["noise_d2"].append(float(lm_inv(d2, 0.02, 2.0)))
        raw["noise_tail"].append(float(logit(L)))
    print(rid, "done")
json.dump({"raw": raw, "doc": __doc__}, open(os.path.join(HERE, "noise_init.json"), "w"))
