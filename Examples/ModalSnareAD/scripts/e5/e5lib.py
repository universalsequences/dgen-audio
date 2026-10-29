"""E5 helpers: table head (ungated head_modes), C wire twin, metrics."""
import ctypes as C, json, os, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e5")
R = MAD.ROOT
MAD.use("dataset"); MAD.use("e1")
from villain_common import read_wav_f32, head_resynth, write_wav_f32  # noqa
from metric import score as mr_score, band_score, band_env_db  # noqa  (E1 metrics)
MODES = json.load(open(os.path.join(R, "data/villain_modes.json")))["renders"]
D = os.path.join(R, "data/villain")
SR = 48000
VDET = [0.906, 1.062, 0.951, 1.114, 0.874, 1.147]

def env_of(rid, ungated=True):
    r = MODES[rid]; e = dict(r["chain"])
    for i, s in enumerate(r["slots"]):
        e[f"b{i}_f"] = s["f"]; e[f"b{i}_gf"] = s["g0"] * s["f"]; e[f"b{i}_tau"] = s["tau"]
        e[f"b{i}_amp"] = s["amp"]; e[f"b{i}_idec"] = 1 / s["dec0"]; e[f"b{i}_irise"] = 1 / s["rise0"]
        e[f"b{i}_t0"] = s["t0"]; e[f"b{i}_ph"] = s["ph"]
        e[f"b{i}_mh"] = 0.0 if ungated else s["mh"]; e[f"b{i}_md"] = 0.0 if ungated else s["md"]
    return e

def head_modes(rid, n=SR, vel=1.0, keep=None):
    """head_v * head_sum (ungated) -- what drives VILLAIN's strands. keep: slot indices to keep."""
    e = env_of(rid)
    if keep is not None:
        for i in range(32):
            if i not in keep: e[f"b{i}_amp"] = 0.0
    return head_resynth(e, n, vel=vel)

class WireP(C.Structure):
    _fields_ = [("nstr", C.c_int), ("det", C.c_float * 8)] + [(k, C.c_float) for k in
               ["zs", "gap", "wire_hz", "ratio2", "t60_ms", "ceil", "hp", "level", "push1", "push2", "stiff"]]
def _wires_lib():
    """C twin of the strand model (wires.c), built on first use into $MODAL_AD_ROOT/e5."""
    import subprocess
    lib, src = os.path.join(HERE, "wires.dylib"), os.path.join(MAD.code("e5"), "wires.c")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        subprocess.run(["cc", "-O2", "-dynamiclib", "-fPIC", src, "-o", lib], check=True)
    return C.CDLL(lib)
_lib = _wires_lib()
_fp = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_lib.wires.argtypes = [_fp, C.c_int, C.POINTER(WireP), _fp, C.c_void_p]
_lib.svf.argtypes = [_fp, C.c_int, C.c_float, C.c_float, C.c_int, _fp]

VILLAIN_DEFAULT = dict(nstr=6, det=VDET, ratio2=2.41, ceil=0.02, hp=0.1, push1=0.769, push2=0.4615, stiff=0.8)

def villain_wire_params(rid):
    c = MODES[rid]["chain"]
    return dict(VILLAIN_DEFAULT, zs=c["c_zs"], gap=c["c_gap"], wire_hz=c["c_wire_hz"],
                t60_ms=max(1.0, c["c_wire_ms"]), level=60 * c["c_rattle"])

def wires(head, **kw):
    p = dict(VILLAIN_DEFAULT); p.update(kw)
    s = WireP(); s.nstr = int(p["nstr"])
    for i, d in enumerate(p["det"][:8]): s.det[i] = d
    for k in ["zs", "gap", "wire_hz", "ratio2", "t60_ms", "ceil", "hp", "level", "push1", "push2", "stiff"]:
        setattr(s, k, float(p[k]))
    h = np.ascontiguousarray(head, np.float32); y = np.zeros_like(h)
    _lib.wires(h, len(h), C.byref(s), y, None)
    return y

def svf(x, hz, k, hp):
    x = np.ascontiguousarray(x, np.float32); y = np.zeros_like(x)
    _lib.svf(x, len(x), hz, k, int(hp), y); return y

def buzz_bp(x, lo, hi, k, steep=0.0):
    """VILLAIN buzz band-pass the strands go through before the drum bus."""
    b1 = svf(svf(x, lo, 1.414213562, 1), hi, k, 0)
    return b1 if steep == 0 else (1 - steep) * b1 + steep * svf(b1, hi, 1.414213562, 0)

def E(x): return float(np.sum(np.asarray(x, np.float64) ** 2))
def relrms(p, t): return float(np.sqrt(E(np.asarray(p, np.float64) - t) / E(t)))
def explained(p, t):
    """waveform energy explained: 1 - E(t-p)/E(t) (can be < 0)."""
    return 1 - E(np.asarray(t, np.float64) - p) / E(t)
def stem(rid, ch): return read_wav_f32(os.path.join(D, "stems", f"{rid}.{ch}.wav")).astype(np.float64)
def out(rid): return read_wav_f32(os.path.join(D, f"{rid}.wav")).astype(np.float64)

# ---------------------------------------------------------------- out-domain helpers
def fade_of(rid, n=SR):
    g = MODES[rid]["global"]; age = np.arange(n) / SR
    x = np.clip((age - (g["len_s"] - g["fade_s"])) / g["fade_s"], 0, 1); fu = x * x * (3 - 2 * x)
    f1 = 1 - fu; cfc = MODES[rid]["chain"]["c_fade_curve"]
    return (1 - cfc) * f1 + cfc * f1 ** 4

def strand_out_target(rid):
    """d = out - out(rattle=0): everything the strand stage adds to the final output (energy.py)."""
    return read_wav_f32(os.path.join(HERE, "targets", f"{rid}.strand_out.wav")).astype(np.float64)
def out_r0(rid):
    return read_wav_f32(os.path.join(HERE, "targets", f"{rid}.out_r0.wav")).astype(np.float64)

# ---------------------------------------------------------------- metrics
def band_energy_explained(p, t):
    """1 - sum|P-T| / sum T over the (linear) 32-band, 21 ms smoothed power envelopes of
    band_env_db. 1 = identical band envelopes, 0 = silence; noise-realisation robust."""
    P = 10 ** (band_env_db(p) / 10); T = 10 ** (band_env_db(t) / 10)
    return float(1 - np.abs(P - T).sum() / T.sum())

def scores(p, t):
    return dict(band=band_score(p, t), mr=mr_score(p, t), bee=band_energy_explained(p, t),
                eratio=E(p) / max(E(t), 1e-30))

# ---------------------------------------------------------------- noise baselines
_NB = json.load(open(os.path.join(R, "e1/noise_bands/bands.json")))
NBANDS = np.fromfile(os.path.join(R, "e1/noise_bands/bands.f32"), "<f4").reshape(16, -1).astype(np.float64)
_NFFT, _NHOP = 1024, 240
_fr = np.fft.rfftfreq(_NFFT, 1 / SR); _zz = np.exp(-1j * 2 * np.pi * _fr / SR)
GB = np.array([bd["norm"] ** 2 * np.abs((bd["b"][0] + bd["b"][1] * _zz + bd["b"][2] * _zz ** 2) /
               (bd["a"][0] + bd["a"][1] * _zz + bd["a"][2] * _zz ** 2)) ** (2 * _NB["stages"]) for bd in _NB["bands"]])
_win = np.hanning(_NFFT)

def _band_rms(x):
    """per-band RMS envelope e_b at frame centres (E1 noise_init projection)."""
    nfr = 1 + (len(x) - _NFFT) // _NHOP
    idx = np.arange(nfr)[:, None] * _NHOP + np.arange(_NFFT)[None, :]
    P = np.abs(np.fft.rfft(x[idx] * _win, axis=1)) ** 2 * 2 / (_win ** 2).sum()
    tc = (np.arange(nfr) * _NHOP + _NFFT / 2)
    return np.sqrt(np.maximum(P @ GB.T / (GB ** 2).sum(1), 1e-20)), tc

def _synth(env, tc, n):
    s = np.arange(n)
    return sum(np.interp(s, tc, env[:, b]) * NBANDS[b, :n] for b in range(16))

def oracle_noise(x, iters=4):
    """Upper bound for ANY 16-band envelope-noise model: the target's own per-band RMS
    envelopes on the fixed band noises, iteratively corrected (multiplicative) so the
    projected band envelopes of the synthesis match the target's."""
    n = len(x); eT, tc = _band_rms(x); env = eT.copy()
    for _ in range(iters):
        y = _synth(env, tc, n); eY, _ = _band_rms(y)
        env = env * np.clip(eT / np.maximum(eY, 1e-12), 0.25, 4)
    return _synth(env, tc, n)

def head_envelope(rid, n=SR, vel=1.0):
    """E(t) = sqrt(sum_k env_k^2) (E1 coupling drive) from the table, incl. head_v."""
    e = env_of(rid); c = MODES[rid]["chain"]; t = np.arange(n) / SR; T = np.maximum(0, t - c["c_pre"])
    acc = np.zeros(n)
    for i in range(32):
        tt = np.maximum(0, T - e[f"b{i}_t0"])
        acc += (e[f"b{i}_amp"] * np.exp(-tt * e[f"b{i}_idec"]) * (1 - np.exp(-tt * e[f"b{i}_irise"]))) ** 2
    hv = c["c_head"] * (vel ** c["c_hv"] if vel > 0 else 0)
    return hv * np.sqrt(acc + 1e-10)

def couple_noise(x, Eh):
    """E1's linear head-coupled term alone: sum_b C_b E(t) N_b, C_b >= 0 least-squares on band RMS."""
    eT, tc = _band_rms(x); Ef = np.interp(tc, np.arange(len(Eh)), Eh)
    C = np.maximum((eT * Ef[:, None]).sum(0) / max((Ef ** 2).sum(), 1e-20), 0)
    n = len(x); return sum(C[b] * Eh * NBANDS[b, :n] for b in range(16)), C

E1FIT = {p["id"]: p for p in json.load(open(os.path.join(R, "e1/fits/params.json")))["natural"]}
def head_e1fit(rid, n=SR, vel=1.0):
    """E1's fitted head (table x learned residuals amp/dec/freq) -- a stand-in for a decoder's head."""
    p = E1FIT[rid.replace("_v040", "_v100").replace("_v070", "_v100")]; e = env_of(rid)
    for i in range(32):
        e[f"b{i}_amp"] *= p["head_amp_mul"][i]; e[f"b{i}_idec"] /= p["head_dec_mul"][i]
        e[f"b{i}_f"] *= p["head_freq_mul"][i]; e[f"b{i}_gf"] *= p["head_freq_mul"][i]
    return head_resynth(e, n, vel=vel)
