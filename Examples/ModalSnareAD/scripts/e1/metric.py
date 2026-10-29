"""Python port of ModalBestCheckpointScorer.multiScaleSpectralScore (MR-STFT, hann, hop w/4,
mean over bins of |sqrt(P)-sqrt(T)| + |0.5 log(P+eps^2) - 0.5 log(T+eps^2)|, averaged over windows)."""
import numpy as np
def score(p, t, windows=(64, 128, 256, 512, 1024, 2048), eps=1e-3):
    n = min(len(p), len(t)); p = np.asarray(p[:n], np.float64); t = np.asarray(t[:n], np.float64)
    tot = 0.0
    for w in windows:
        hop = w // 4; hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / w))
        starts = np.arange(0, n - w + 1, hop)
        idx = starts[:, None] + np.arange(w)[None, :]
        P = np.abs(np.fft.rfft(p[idx] * hann, axis=1)) ** 2
        T = np.abs(np.fft.rfft(t[idx] * hann, axis=1)) ** 2
        lin = np.abs(np.sqrt(P) - np.sqrt(T)); lg = np.abs(0.5 * np.log(P + eps * eps) - 0.5 * np.log(T + eps * eps))
        tot += (lin + lg).mean()
    return tot / len(windows)

# ---- realisation-robust band-envelope metric -------------------------------------------------
_W, _HOP = 1024, 256
_fr = np.fft.rfftfreq(_W, 1 / 48000)
_edges = np.exp(np.linspace(np.log(60), np.log(20000), 33))
_B = np.array([((_fr >= a) & (_fr < b)).astype(float) for a, b in zip(_edges[:-1], _edges[1:])])
def band_env_db(x):
    x = np.asarray(x, np.float64); hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(_W) / _W))
    st = np.arange(0, len(x) - _W + 1, _HOP)
    P = np.abs(np.fft.rfft(x[st[:, None] + np.arange(_W)] * hann, axis=1)) ** 2
    E = P @ _B.T / np.maximum(_B.sum(1), 1)
    k = np.ones(4) / 4  # ~21 ms smoothing in time
    E = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, E)
    return 10 * np.log10(E + 1e-12)
def band_score(p, t, floor_db=70.0):
    """Mean |dB diff| of 32 log-band, ~21 ms smoothed envelopes; both clamped at (target peak - floor_db)."""
    n = min(len(p), len(t)); P = band_env_db(p[:n]); T = band_env_db(t[:n])
    lo = T.max() - floor_db
    return float(np.abs(np.maximum(P, lo) - np.maximum(T, lo)).mean())
