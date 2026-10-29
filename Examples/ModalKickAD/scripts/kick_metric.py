"""Kick metrics. MR-STFT = E1's ModalDrum port (+ a 4096 window variant for kick lows);
band_score: 32 log bands from 25 Hz (W=2048 so the sub octave has bins), ~21 ms smoothing,
70 dB floor. relrms for the (almost) deterministic waveform."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
MAD.use("e1")
from metric import score as mr_score  # noqa: E402

_W, _HOP = 2048, 256
_fr = np.fft.rfftfreq(_W, 1 / 48000)
_edges = np.exp(np.linspace(np.log(25), np.log(20000), 33))
_B = np.array([((_fr >= a) & (_fr < b)).astype(float) for a, b in zip(_edges[:-1], _edges[1:])])
_B = _B[_B.sum(1) > 0]
_HANN = 0.5 * (1 - np.cos(2 * np.pi * np.arange(_W) / _W))

def band_env_db(x):
    x = np.concatenate([np.zeros(_W // 2), np.asarray(x, np.float64)])
    st = np.arange(0, len(x) - _W + 1, _HOP)
    P = np.abs(np.fft.rfft(x[st[:, None] + np.arange(_W)] * _HANN, axis=1)) ** 2
    E = P @ _B.T / np.maximum(_B.sum(1), 1)
    k = np.ones(4) / 4
    E = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, E)
    return 10 * np.log10(E + 1e-12)

def band_score(p, t, floor_db=70.0):
    n = min(len(p), len(t)); P = band_env_db(p[:n]); T = band_env_db(t[:n])
    lo = T.max() - floor_db
    return float(np.abs(np.maximum(P, lo) - np.maximum(T, lo)).mean())

def mr(p, t):
    return mr_score(p, t, windows=(64, 128, 256, 512, 1024, 2048, 4096))

def relrms(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2) / max(np.mean(b ** 2), 1e-30)))
