"""E6 shared helpers: load a trainer run (weights.json + codes.json), the knob (PCA-standardised)
transform of the exported instrument, numpy renders through the E2 twin (scripts/e2/template.py) with the
E1 fixed band-noise table (same realisation for every render), descriptors and degeneracy flags."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e6"); R = MAD.ROOT
E2 = os.path.join(R, "e2")
MAD.use("dataset", "e1", "e2")
import template as tp, latent_np as ln  # noqa: E402
from metric import score, band_score, band_env_db  # noqa: E402

SR = 48000
SEG = tp.SEG
BANK = None


def bank():
    global BANK
    if BANK is None:
        BANK = ln.e1_bank()
    return BANK


def load_run(run):
    return ln.load_run(run)


def knob_transform(Z, shrink_to=2.5):
    """Same as scripts/e2/export_instrument.py: z_knob = (z - mu) V / s; returns (mu, M) with z = mu + z_knob @ M."""
    mu = Z.mean(0); U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    sc = s / np.sqrt(len(Z)); V = Vt.T
    Zs = (Z - mu) @ V / sc
    k = max(1.0, float(np.abs(Zs).max()) / shrink_to)
    sc = sc * k
    M = np.diag(sc) @ V.T  # [D, D]: z_raw = mu + z_knob @ M
    return mu, M, Zs / k


def decode(W, z, vel=1.0):
    return tp.decode(W, z, vel)


def render(W, z, vel=1.0, decay=1.0, tune=0.0, seconds=1.0):
    p = tp.decode(W, z, vel)
    y, h, n = tp.render_natural(p, level=1.0, bands=bank(), parts=True, decay=decay, tune=tune, seconds=seconds)
    return p, y, h, n


def mode_energy(p):
    a = np.asarray(p["mode_amp"]); d = np.asarray(p["mode_decay"])
    return a * a * d / 2


def centroid(y, sr=SR):
    n = min(len(y), int(0.3 * sr))
    S = np.abs(np.fft.rfft(y[:n])) ** 2; f = np.fft.rfftfreq(n, 1 / sr)
    return float((S * f).sum() / max(S.sum(), 1e-20))


def t_minus(y, db=30.0, sr=SR):
    """seconds until the ~10 ms RMS envelope falls db below its peak for good."""
    w = 480; n = len(y) // w
    e = np.sqrt(np.mean(y[: n * w].reshape(n, w) ** 2, axis=1) + 1e-20)
    e = 20 * np.log10(e); pk = e.max(); above = np.nonzero(e > pk - db)[0]
    return float((above[-1] + 1) * w / sr) if len(above) else 0.0


def descriptors(p, y, h, n):
    """fundamental (slot 4 role, Hz at tune 0 heard pitch), dominant head mode, centroid, decay -30 dB,
    noise/head energy ratio (pre drive), peak, rms dB."""
    E = mode_energy(p); f = np.asarray(p["mode_freq"])
    eh = float(np.sum(h * h)); en = float(np.sum(n * n))
    return {"f0": float(f[4]), "f_dom": float(f[int(np.argmax(E))]),
            "f_logmean": float(np.exp(np.sum(E * np.log(f)) / max(E.sum(), 1e-30))),
            "centroid": centroid(y), "t30": t_minus(y, 30), "noise_head_db": 10 * math.log10((en + 1e-20) / (eh + 1e-20)),
            "peak": float(np.abs(y).max()), "rms_db": 10 * math.log10(float(np.mean(y * y)) + 1e-20)}


def live_modes(p, share=0.01):
    E = mode_energy(p); return np.nonzero(E > share * E.sum())[0]


def beat_pairs(p, share=0.02, hz=12.0):
    """live mode pairs closer than `hz` (beating slower than ~1/12 s), both above `share` energy."""
    f = np.asarray(p["mode_freq"]); idx = live_modes(p, share); out = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            if abs(f[a] - f[b]) < hz: out.append((int(a), int(b), float(abs(f[a] - f[b]))))
    return out


def mr(a, b):
    return score(a, b)


def band(a, b):
    """symmetric band-envelope distance (mean |dB|, floor 70 dB below the louder one)."""
    n = min(len(a), len(b)); A = band_env_db(a[:n]); B = band_env_db(b[:n])
    lo = max(A.max(), B.max()) - 70
    return float(np.abs(np.maximum(A, lo) - np.maximum(B, lo)).mean())


def band_env(a):
    return band_env_db(a)


def band_from_env(A, B):
    lo = max(A.max(), B.max()) - 70
    return float(np.abs(np.maximum(A, lo) - np.maximum(B, lo)).mean())


def evenness(steps):
    s = np.asarray(steps, float); m = s.mean()
    return {"mean": float(m), "max_over_mean": float(s.max() / max(m, 1e-12)), "cv": float(s.std() / max(m, 1e-12)),
            "first_over_mean": float(s[0] / max(m, 1e-12)), "total": float(s.sum())}


def write_walk_wav(path, ys, gap=0.25, seconds=0.6):
    """concatenate hits (each cut to `seconds` with a 5 ms fade) with `gap` s of silence."""
    from villain_common import write_wav_f32
    n = int(seconds * SR); g = np.zeros(int(gap * SR)); fade = np.ones(n); fade[-240:] = np.linspace(1, 0, 240)
    parts = []
    for y in ys:
        y = np.asarray(y, float)[:n]
        if len(y) < n: y = np.concatenate([y, np.zeros(n - len(y))])
        parts += [y * fade, g]
    write_wav_f32(path, np.concatenate(parts))


def json_dump(o, path):
    json.dump(o, open(path, "w"), indent=1, default=lambda x: float(x) if np.isscalar(x) else np.asarray(x).tolist())


# ---- analytic band-envelope surrogate (differentiable twin lives in dgen LatentPath.swift) ----------
# Log band-power envelopes computed in closed form from the decoded parameters, no audio:
# head modes as Gaussian bumps in log f (glide included), the 16 band noises spread over the metric's
# 32 log bands by their filter response, gates/coupling/fade exact, the drive + clipper as a
# level-dependent gain on the envelope RMS (spectral spreading ignored).
SUR_T = np.array([0.001, 0.002, 0.004, 0.007, 0.011, 0.016, 0.023, 0.032] + list(0.04 + 0.03 * np.arange(32)))
SUR_EDGES = np.exp(np.linspace(np.log(60), np.log(20000), 33))
SUR_LC = 0.5 * (np.log(SUR_EDGES[:-1]) + np.log(SUR_EDGES[1:]))
SUR_SIG = 0.5 * (np.log(SUR_EDGES[1]) - np.log(SUR_EDGES[0]))
SUR_NORM = float(np.sum(np.exp(-0.5 * ((SUR_LC - SUR_LC[16]) / SUR_SIG) ** 2)))
SUR_WIDTH = np.diff(SUR_EDGES)
SUR_FLOOR = 1e-9


def noise_spread(sr=SR, n=1 << 15):
    """S[i, b]: share of unit-RMS band-noise i's power that falls in metric band b."""
    nb = tp.LAY["noise_bands"]; fc = np.array(nb["fc"]); Q = nb["Q"]
    f = np.fft.rfftfreq(n, 1 / sr); zinv = np.exp(-2j * np.pi * f / sr)
    w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q)
    b0, a1, a2 = al / (1 + al), -2 * np.cos(w0) / (1 + al), (1 - al) / (1 + al)
    H = (b0[:, None] * (1 - zinv ** 2)) / (1 + a1[:, None] * zinv + a2[:, None] * zinv ** 2)
    P = np.abs(H) ** (2 * nb["stages"]); P /= P.sum(1, keepdims=True)
    return np.array([[P[i, (f >= a) & (f < b)].sum() for a, b in zip(SUR_EDGES[:-1], SUR_EDGES[1:])] for i in range(len(fc))])


SUR_S = noise_spread()


def surrogate(p, decay=1.0):
    """[Tg, 32] dB band-density envelope of the decoded params p (velocity already in p)."""
    g = {k: float(np.ravel(v)[0]) for k, v in p.items() if np.size(v) == 1}
    tt = SUR_T[:, None]
    f = np.asarray(p["mode_freq"]); sh = f * np.asarray(p["mode_glide"])
    nyq = np.clip((0.48 * SR - (f + np.maximum(sh, 0))) / (0.08 * SR), 0, 1)
    T = np.maximum(0, tt - g["pre"])
    GH = 1 - g["gh_depth"] * smooth01((T - g["gh_time"]) / g["gh_fall"])
    GD = 1 - g["gd_depth"] * smooth01((T - g["gd_time"]) / g["gd_fall"])
    t = np.maximum(0, T - np.asarray(p["mode_t0"]))
    env = np.asarray(p["mode_amp"]) * nyq * GH * GD * np.exp(-t / (np.asarray(p["mode_decay"]) * decay)) * (1 - np.exp(-t / np.asarray(p["mode_rise"])))
    fk = f + sh * np.exp(-t / np.asarray(p["mode_gtau"]))  # [Tg, K]
    wk = np.exp(-0.5 * ((np.log(fk)[:, :, None] - SUR_LC[None, None, :]) / SUR_SIG) ** 2) / SUR_NORM  # [Tg,K,32]
    Hb = np.einsum("tk,tkb->tb", 0.5 * env * env, wk)
    E = np.sqrt(np.sum(env * env, axis=1, keepdims=True) + 1e-10)
    nt = np.maximum(0, tt - g["noise_onset"]); att = 1 - np.exp(-nt / g["noise_rise"])
    A, L = np.asarray(p["band_amp"]), np.asarray(p["band_tail"])
    benv = att * (A * (1 - L) * np.exp(-nt / (np.asarray(p["band_d1"]) * decay)) + A * L * np.exp(-nt / (np.asarray(p["band_d2"]) * decay))) \
        + np.asarray(p["band_couple"]) * E
    ng = 1 - g["gate_noise"] * (1 - GD)
    Np = (ng * benv) ** 2  # [Tg, 16]
    Nb = Np @ SUR_S
    xr = np.sqrt(0.5 * np.sum(env * env, axis=1, keepdims=True) + Np.sum(1, keepdims=True))
    Aa = np.sqrt(2) * xr
    ge = 1 - g["drive_amt"] + g["drive_amt"] * g["drive_norm"] * np.tanh(g["drive"] * Aa) / (Aa + 1e-6)
    v = ge * Aa / g["clip_ceil"]; gc = 1 - g["clip_mix"] + g["clip_mix"] / np.sqrt(np.sqrt(1 + v ** 4))
    f1 = 1 - smooth01((tt - (g["len"] - g["fade"])) / g["fade"]); fade = f1 * (1 - g["fade_curve"]) + f1 ** 4 * g["fade_curve"]
    G2 = (g["gain"] * fade * ge * gc) ** 2
    return 10 * np.log10(G2 * (Hb + Nb) / SUR_WIDTH + SUR_FLOOR)


def smooth01(x):
    c = np.clip(x, 0, 1)
    return c * c * (3 - 2 * c)


def sur_dist(A, B):
    return float(np.mean(np.sqrt((A - B) ** 2 + 0.01)))
