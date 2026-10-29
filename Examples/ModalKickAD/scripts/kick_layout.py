"""Latent Kick decoder contract: output segments (natural-unit per-kick parameters of the
VILLAIN Kick voice, BEFORE the transform knobs), shared by kick_np.py (numpy twin),
targets.py (teacher rows), gen_dsp.py (realtime instrument) and the dgen trainer
(decoder_layout.json).

Squash of decoder output u:
  log : exp(ln lo + sigmoid(u) (ln hi - ln lo))
  lin : lo + sigmoid(u) (hi - lo)
  pi  : pi tanh(u)
"""
import json, math, os

K = 24      # VILLAIN Kick role-ordered slots: 0 sub, 1 beat partner, 2-13 body, 14-23 knock
K2 = 4      # slots 0..3 carry a second pitch drop (g2, tau2)
J = 5       # transient sines (VILLAIN's xA0..xA4: ticks / pulses)
NB = 18     # fixed noise bands: ModalSnareAD/scripts/e3proto/NOISE_BANDS.md filters, extended down to 50 Hz (half-octave)
H = 128
LATENT_D = int(os.environ.get("LATENT_D", "4"))
D_IN = LATENT_D          # the decoder sees z only; velocity / pitch / knobs are VILLAIN's own laws post-decoder

# (name, width, squash, lo, hi)  -- ranges cover the 33 VILLAIN kicks with margin
MODE_SEGS = [
    ("mode_f",     K, "log", 4.0, 12000.0),    # b_f at tuning 1 (exp lf0), Hz, BEFORE spread/beat
    ("mode_dec",   K, "log", 1e-3, 12.0),      # exp(ld0) s, BEFORE tilt/decay
    ("mode_amp",   K, "log", 1e-7, 14.0),      # b_amp (bank_norm * amu)
    ("mode_glide", K, "lin", -0.95, 14.0),     # g: sh = max(-0.9 f, bendE g f)
    ("mode_tau",   K, "log", 2e-4, 0.8),       # exp(lt) s, BEFORE drop_time
    ("mode_rise",  K, "log", 5e-7, 0.3),       # exp(lr) s, BEFORE attack
    ("mode_t0",    K, "lin", 0.0, 0.006),      # s
    ("mode_phase", K, "pi", 0.0, 0.0),
    ("mode_mga",   K, "lin", 0.0, 1.0),        # gate-A mask
    ("mode_mgb",   K, "lin", 0.0, 1.0),        # gate-B mask
    ("mode_msm",   K, "lin", 0.0, 1.0),        # body-shaper mask
    ("mode_glide2", K2, "lin", -0.5, 2.0),     # g2 (slots 0-3)
    ("mode_tau2",  K2, "log", 0.03, 2.0),      # exp(lt2) s (1 = unused)
]
TRANS_SEGS = [
    ("tr_f",    J, "lin", 0.0, 6000.0),        # Hz (0 = DC pulse)
    ("tr_dec",  J, "log", 5e-5, 1.5),          # 1/idec s
    ("tr_dl",   J, "lin", -50.0, 1200.0),      # absolute drop, Hz: sh = max(-0.9 f, bendE dl)
    ("tr_tau",  J, "log", 1e-3, 1.0),          # drop time constant s (1 = unused)
    ("tr_rise", J, "log", 1e-9, 1.5),          # 1/irise s
    ("tr_amp",  J, "lin", -2.5, 4.0),          # signed amplitude
    ("tr_phase", J, "pi", 0.0, 0.0),
    ("tr_t0",   J, "lin", 0.0, 0.012),
    ("tr_wb",   J, "lin", 0.0, 1.0),           # weight on bodyE
    ("tr_wt",   J, "lin", 0.0, 1.0),           # weight on tickE
    ("tr_mgb",  J, "lin", 0.0, 1.0),           # gate-B mask
    ("tr_msm",  J, "lin", 0.0, 1.0),           # shaper mask
]
SCALAR_SEGS = [
    # growl (one gliding clipped sine, 35 Hz base)
    ("gr_amp", 1, "lin", 0.0, 12.0), ("gr_dl", 1, "lin", 0.0, 200.0), ("gr_tau", 1, "log", 0.01, 1.5),
    ("gr_off", 1, "pi", 0.0, 0.0), ("gr_clip", 1, "lin", 0.9, 1.0), ("gr_a", 1, "lin", -0.01, 0.01),
    ("gr_b", 1, "lin", 0.0, 1.0), ("gr_d1", 1, "log", 0.02, 0.3), ("gr_d2", 1, "log", 0.02, 0.3),
    # burst (Gaussian/Hann windowed sine)
    ("bu_amp", 1, "lin", 0.0, 1.2), ("bu_samp", 1, "lin", 0.0, 1.0), ("bu_tc", 1, "lin", 0.0, 0.01),
    ("bu_w", 1, "log", 1e-4, 0.003), ("bu_hann", 1, "lin", 0.0, 1.0), ("bu_f", 1, "log", 300.0, 3000.0),
    ("bu_off", 1, "lin", -1.0, 25.0), ("bu_sm", 1, "lin", 0.0, 1.0),
    # time bases
    ("samp", 1, "lin", 0.0, 1.0), ("wk", 1, "lin", 0.0, 0.4), ("wtau", 1, "log", 1e-3, 1.5),
    ("nsamp", 1, "lin", 0.0, 1.0), ("fd_samp", 1, "lin", 0.0, 1.0),
    # native velocity curves: vcurve(k, e) = 1 - k + k vel^e ; velp(e) = vel^e
    ("vb_k", 1, "lin", 0.0, 1.0), ("vb_e", 1, "lin", 0.0, 2.0), ("vd_k", 1, "lin", 0.0, 0.5),
    ("vk_k", 1, "lin", 0.0, 1.0), ("vk_e", 1, "lin", 0.0, 2.0), ("vt_k", 1, "lin", 0.0, 1.0),
    ("vt_e", 1, "lin", 0.0, 2.0), ("va_k", 1, "lin", 0.0, 0.6), ("vpre_e", 1, "lin", 0.0, 2.0),
    ("vtex_e", 1, "lin", 0.0, 2.0), ("vpost_e", 1, "lin", 0.0, 2.0),
    # length, hold, release, fade
    ("len", 1, "log", 0.05, 1.2), ("len_dsc", 1, "lin", 0.0, 1.0), ("hold", 1, "log", 0.05, 0.5),
    ("rel", 1, "log", 0.005, 0.3), ("fd_s", 1, "lin", 0.0, 0.1), ("fd_frac", 1, "lin", 0.0, 0.4),
    # gate A (kick-gate); hC = 100 means never
    ("gA_dsc", 1, "lin", 0.0, 1.0), ("gA_hvk", 1, "lin", 0.0, 0.4), ("gA_hL", 1, "lin", 0.0, 1.2),
    ("gA_hH", 1, "lin", 0.0, 1.2), ("gA_hR", 1, "lin", -8.0, 1.0), ("gA_hC", 1, "log", 1e-4, 100.0),
    ("gA_rR", 1, "lin", 0.0, 1.2), ("gA_rC", 1, "log", 1e-4, 1.2), ("gA_wexp", 1, "lin", 0.0, 1.0),
    ("gA_wrat", 1, "lin", 0.0, 1.0), ("gA_wss", 1, "lin", 0.0, 1.0), ("gA_pe", 1, "log", 1.0, 8.0),
    ("gA_tmix", 1, "lin", 0.0, 0.6), ("gA_tail", 1, "log", 0.05, 1.5), ("gA_floor", 1, "lin", 0.0, 0.01),
    # gate B
    ("gB_dsc", 1, "lin", 0.0, 1.0), ("gB_hL", 1, "lin", 0.0, 1.0), ("gB_hH", 1, "lin", 0.0, 1.2),
    ("gB_hC", 1, "log", 1e-4, 100.0), ("gB_rL", 1, "lin", 0.0, 0.5), ("gB_rR", 1, "lin", 0.0, 1.2),
    ("gB_rC", 1, "log", 1e-4, 1.2), ("gB_wexp", 1, "lin", 0.0, 1.0), ("gB_wss", 1, "lin", 0.0, 1.0),
    ("gB_pe", 1, "log", 1.0, 8.0),
    ("oA_e", 1, "lin", 0.0, 1.0),               # out gate = GA^oA_e
    # body shaper on the MSM-masked bank
    ("sh_ec", 1, "lin", -0.05, 0.8), ("sh_p3", 1, "lin", 0.0, 2.0), ("sh_p4", 1, "lin", 0.0, 2.0),
    ("sh_p5", 1, "lin", 0.0, 2.0), ("sh_ed", 1, "lin", 0.0, 1.0), ("sh_ecl", 1, "log", 0.5, 100.0),
    ("sh_oc", 1, "lin", 0.0, 0.8), ("sh_ow", 1, "log", 0.1, 1.5), ("sh_rk", 1, "lin", 0.0, 1.0),
    ("sh_rg", 1, "lin", 0.5, 1.2), ("sat_on", 1, "lin", 0.0, 1.0), ("sat_g", 1, "log", 0.05, 1.5),
    ("sat_b", 1, "lin", -0.1, 0.1),
    ("bk_amt", 1, "lin", 0.0, 0.2), ("bk_del", 1, "lin", 0.0, 0.02), ("bhp_on", 1, "lin", 0.0, 1.0),
    # recording clipper
    ("ceil", 1, "lin", 0.4, 1.1), ("cneg", 1, "lin", 0.9, 1.2), ("coff", 1, "lin", -0.02, 0.0),
    ("kap", 1, "lin", 0.0, 1.0), ("kbp", 1, "lin", 0.0, 1.0), ("kan", 1, "lin", 0.0, 1.0),
    ("kbn", 1, "lin", 0.0, 1.0), ("kr", 1, "lin", 0.0, 1.0), ("kn", 1, "log", 2.0, 16.0),
    ("dry", 1, "lin", 0.3, 1.0), ("cmix", 1, "lin", 0.2, 1.0),
    # record filters, drive split, output
    ("dc_hz", 1, "lin", 0.0, 5.0), ("rl_hz", 1, "log", 2000.0, 21000.0), ("rl_q1", 1, "lin", 0.4, 0.8),
    ("rl_q2", 1, "lin", 0.5, 1.4), ("rl_two", 1, "lin", 0.0, 1.0), ("rl_mix", 1, "lin", 0.0, 1.0),
    ("dr_track", 1, "lin", 0.0, 1.0), ("lvl", 1, "lin", 0.9, 1.1), ("out_clip", 1, "log", 0.8, 100.0),
    ("np", 1, "lin", 0.0, 1.0),
]
NOISE_SEGS = [
    ("band_amp",    NB, "log", 1e-6, 2.0),     # A_b (unit-RMS band noise)
    ("band_d1",     NB, "log", 2e-4, 0.5),     # fast decay s
    ("band_tail",   NB, "log", 1e-5, 1.0),      # slow-stage share
    ("band_d2",     NB, "log", 0.01, 2.0),     # slow decay s
    ("band_couple", NB, "log", 1e-6, 2.0),     # C_b: modal-envelope (grit) drive
    ("noise_rise",  1, "log", 1e-6, 0.06),
    ("noise_onset", 1, "lin", 0.0, 0.03),
    ("noise_q",     1, "log", 1.0, 8.0),       # fast stage exp(-(t/d1)^q) (VILLAIN layer power 1,2,4,8)
    ("noise_ve",    1, "lin", 0.0, 2.0),       # vel^ve
    ("noise_ga",    1, "lin", 0.0, 1.0),       # GA^ga  (noise and hum gate)
    ("noise_gb",    1, "lin", 0.0, 1.0),       # GB^gb
    ("couple_ga",   1, "lin", 0.0, 1.0),       # coupling (grit) term gate GA^couple_ga
    # VILLAIN hum (aux layer): (1-e^-age/.1) e^-age/.22 (s + h2 (1 - 2 s^2)) LP12(white),
    # s = sin(2 pi tuning 92 age); exact: same white, same dgen 12 Hz low-pass
    ("hum_amt",     1, "lin", 0.0, 60.0),      # c_aux_amt * c_hu_amt
    ("hum_h2",      1, "lin", 0.0, 1.0),
    ("hum_ve",      1, "lin", 0.0, 2.0),
]
SEGS = MODE_SEGS + TRANS_SEGS + SCALAR_SEGS + NOISE_SEGS
P = sum(s[1] for s in SEGS)

BAND_LO, BAND_HI, BAND_Q, BAND_STAGES = 50.0, 18000.0, 1.8, 2
BAND_FC = [BAND_LO * (BAND_HI / BAND_LO) ** (i / (NB - 1)) for i in range(NB)]

# transform knobs (VILLAIN Kick names, ranges and defaults); all act post-decoder per hit
KNOBS = [  # name, default, min, max
    ("tune", 1, 0.5, 2), ("drop_time", 1, 0.25, 4), ("tilt", 0, -2, 2), ("spread", 1, 0.5, 1.6),
    ("beat", 1, 0, 4), ("decay", 1, 0.25, 2.5), ("bend", 1, 0, 2.5), ("attack", 1, 0.25, 3),
    ("body", 1, 0, 2), ("knock", 1, 0, 3), ("tick", 1, 0, 3), ("shape", 1, 0, 3), ("growl", 1, 0, 3),
    ("noise", 1, 0, 4), ("hold", 1, 0.25, 4), ("release", 1, 0.25, 4), ("length", 1, 0.5, 4),
    ("drive", 1, 1, 5), ("clip_amt", 1, 0, 1), ("ceiling", 1, 0.25, 2), ("level", 1, 0, 2),
]


def offsets():
    o, out = 0, {}
    for name, w, sq, lo, hi in SEGS:
        out[name] = (o, w, sq, lo, hi)
        o += w
    return out


def squash(u, sq, lo, hi):
    import numpy as np
    s = 1 / (1 + np.exp(-np.asarray(u, float)))
    if sq == "log":
        return np.exp(math.log(lo) + s * (math.log(hi) - math.log(lo)))
    if sq == "lin":
        return lo + s * (hi - lo)
    return math.pi * np.tanh(u)


def unsquash(v, sq, lo, hi, eps=1e-6):
    """natural value -> decoder output u (clamped inside the range)"""
    import numpy as np
    v = np.asarray(v, float)
    if sq == "pi":
        return np.arctanh(np.clip(v / math.pi, -1 + eps, 1 - eps))
    if sq == "log":
        p = (np.log(np.maximum(v, 1e-30)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        p = (v - lo) / (hi - lo)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def decode_u(u):
    """u [P] -> {name: array or float} natural units"""
    off = offsets()
    out = {}
    for name, (o, w, sq, lo, hi) in off.items():
        v = squash(u[o:o + w], sq, lo, hi)
        out[name] = v if w > 1 else float(v[0])
    return out


def encode(params):
    """{name: natural} -> u [P]"""
    import numpy as np
    off = offsets()
    u = np.zeros(P)
    for name, (o, w, sq, lo, hi) in off.items():
        u[o:o + w] = unsquash(np.atleast_1d(params[name]), sq, lo, hi)
    return u


def layout_json(norms=None):
    off = offsets()
    return {
        "doc": "Latent Kick decoder contract (ModalKickAD/scripts/kick_layout.py; voice = ModalKickAD/scripts/kick_np.py)",
        "version": "k1", "inputs": [f"z{i}" for i in range(1, LATENT_D + 1)],
        "input_ranges": {"z": [-3, 3]},
        "hidden": [H, H], "activation": "tanh", "P": P, "K": K, "K2": K2, "J": J, "B": NB,
        "matmul": "y[1,out] = x[1,in] @ W[in,out] + b[1,out]; W stored row-major (C order)",
        "weights": {"l1.W": {"file": "w1.json", "shape": [D_IN, H]}, "l1.b": {"file": "b1.json", "shape": [1, H]},
                    "l2.W": {"file": "w2.json", "shape": [H, H]}, "l2.b": {"file": "b2.json", "shape": [1, H]},
                    "l3.W": {"file": "w3.json", "shape": [H, P]}, "l3.b": {"file": "b3.json", "shape": [1, P]}},
        "segments": [{"name": n, "offset": off[n][0], "width": w, "squash": sq, "lo": lo, "hi": hi}
                     for n, w, sq, lo, hi in SEGS],
        "knobs": [{"name": n, "default": d, "min": a, "max": b} for n, d, a, b in KNOBS],
        "noise_bands": {"fc": BAND_FC, "Q": BAND_Q, "stages": BAND_STAGES,
                        "form": "RBJ BPF constant 0 dB peak, DF1, a0-normalised, w0 = 2 pi fc / sr; "
                                "N_b = norm_b * cascade(white), white = sqrt(12) (u - 0.5), u = dgen xorshift32 seed 1, "
                                "zero initial filter state at voice start (no pre-roll)",
                        "norm": norms},
    }
