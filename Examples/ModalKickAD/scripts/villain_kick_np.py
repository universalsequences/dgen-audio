#!/usr/bin/env python3
"""numpy twin of ONE VILLAIN Kick hit (the kick-hit macro), first hit on a fresh voice.

Fed the block-rate values from kick_table.BlockRate (the same arithmetic as the
latched __h copy inside kick-hit), it reproduces every stage of the DSP:
modal bank (24 slots with 1 or 2 pitch drops + time warp + low-cut fade + gates),
5+5 transient sines, growl, burst, body shaper/saturator, bark, body HP (TPT SVF),
noise layers (dgen xorshift32 seed-1 white, dgen RBJ biquads at 2pi/44100 rad/Hz),
crackle, hum, pulse, grit (S&H + slew), recording clipper, DC blocker, record LP,
drive split, level/fade/out gate and the output clip.

    from villain_kick_np import render_hit
    taps = render_hit(env, vel=1.0, pitch=C4, knobs={...}, n=48000)
    taps["out"], taps["modal"], ...
"""
import math
import numpy as np
import ctypes, os
def _filt_lib():
    """float64 lfilter in C (_filt.c), built on first use into $MODAL_AD_ROOT/kick."""
    import subprocess, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
    import modal_ad as MAD
    lib, src = os.path.join(MAD.stage("kick"), "_filt.dylib"), os.path.join(MAD.code("kick"), "_filt.c")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        subprocess.run(["cc", "-O2", "-dynamiclib", "-fPIC", src, "-o", lib], check=True)
    return ctypes.CDLL(lib)
_L = _filt_lib()
_D = ctypes.POINTER(ctypes.c_double)
_L.lfilt2.argtypes = [ctypes.c_double] * 5 + [_D, _D, ctypes.c_int]
_L.tpt_hp.argtypes = [ctypes.c_double] * 3 + [_D, _D, ctypes.c_int]


def lfilter(b, a, x):
    """scipy-compatible for order <= 2 with a[0] == 1"""
    b = list(b) + [0.0] * (3 - len(b)); a = list(a) + [0.0] * (3 - len(a))
    assert a[0] == 1
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.empty_like(x)
    _L.lfilt2(b[0], b[1], b[2], a[1], a[2], x.ctypes.data_as(_D), y.ctypes.data_as(_D), len(x))
    return y

SR = 48000.0
TWOPI = 6.283185307179586
W44 = 0.00014247585730565955  # dgen biquad: w0 = |fc| * 2pi/44100 regardless of the sample rate

KNOB_DEFAULTS = {"tune": 1, "decay": 1, "bend": 1, "attack": 1, "body": 1, "knock": 1, "tick": 1,
                 "shape": 1, "growl": 1, "air": 1, "air_time": 1, "air_tone": 1, "room": 1,
                 "room_time": 1, "room_tone": 1, "click": 1, "click_time": 1, "click_tone": 1,
                 "floor": 1, "crackle": 1, "hum": 1, "grit": 1, "hold": 1, "release": 1, "length": 1,
                 "drive": 1, "clip_amt": 1, "ceiling": 1, "level": 1}
KNOB_CLIP = {"tune": (0.5, 2), "decay": (0.25, 2.5), "bend": (0, 2.5), "attack": (0.25, 3), "body": (0, 2),
             "knock": (0, 3), "tick": (0, 3), "shape": (0, 3), "growl": (0, 3), "air": (0, 4),
             "air_time": (0.25, 4), "room": (0, 4), "room_time": (0.25, 4), "click": (0, 4),
             "click_time": (0.25, 4), "floor": (0, 4), "hum": (0, 4), "grit": (0, 4), "hold": (0.25, 4),
             "release": (0.25, 4), "length": (0.5, 4), "drive": (1, 5), "clip_amt": (0, 1),
             "ceiling": (0.25, 2), "level": (0, 2), "crackle": (0, 4),
             "air_tone": (0.25, 4), "room_tone": (0.25, 4), "click_tone": (0.25, 4)}


def white_noise(n):
    """dgen (noise): xorshift32, state 0 -> 1, value = (float)s / 2^32 (float32 rounding)."""
    out = np.empty(n, dtype=np.float32)
    s = 1
    inv = np.float32(1.0 / 4294967296.0)
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out[i] = np.float32(s) * inv
    return out.astype(np.float64)


def biquad(x, fc, q, gain, mode):
    """dgen biquad (RBJ, direct form I, coefficients /a0, b's * gain), fc in '44.1k Hz'."""
    w = abs(fc) * W44
    c, s = math.cos(w), math.sin(w)
    alpha = s * 0.5 / abs(q)
    a0 = 1 + alpha
    if mode == 0:
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
    elif mode == 1:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    elif mode == 2:
        b = [alpha, 0.0, -alpha]
    else:
        raise ValueError(mode)
    b = [bi * gain / a0 for bi in b]
    a = [1.0, -2 * c / a0, (1 - alpha) / a0]
    return lfilter(b, a, x)


def mixf(a, b, t):
    return a + (b - a) * t


def bandx(x, f1, q1, m1, u1, f2, q2, m2, u2):
    cap = 0.45 * SR
    b1 = mixf(x, biquad(x, min(f1, cap), q1, 1, int(round(m1))), u1)
    return mixf(b1, biquad(b1, min(f2, cap), q2, 1, int(round(m2))), u2)


def band2(x, hp, qh, lp, ql):
    cap = 0.45 * SR
    return biquad(biquad(x, min(hp, cap), qh, 1, 1), min(lp, cap), ql, 1, 0)


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def selector(i, *opts):
    if i <= 0:
        return 0.0
    for k, o in enumerate(opts):
        if i <= k + 1:
            return o
    return 0.0


def xpow(x, pi):
    x2 = x * x
    x4 = x2 * x2
    return selector(pi, x, x2, x4, x4 * x4)


def gpow(g, pi):
    return selector(pi, np.ones_like(g) if isinstance(g, np.ndarray) else 1.0, np.sqrt(np.maximum(g, 0)), g)


def kick_gate(T, hold, rel, wexp, wrat, wss, p, tmix, tail, floor_g):
    gx = np.maximum(0, T - hold)
    x = gx / rel
    xp = xpow(x, p)
    shaped = wexp * np.exp(-xp) + wrat * (1 / (1 + xp)) + wss * (1 - smooth01(x))
    g = mixf(shaped, np.exp(-gx / tail), tmix)
    return floor_g + (1 - floor_g) * g


def knee_side(m, k, w):
    return np.minimum(m, k) + w * np.tanh(np.maximum(0, m - k) / w)


def rat_side(m, c, n):
    return m * np.exp(-np.log(1 + np.exp(n * np.log(np.maximum(m / c, 1e-9)))) / n)


def layer_env(Tn, dl, rise, lin, hold, p, dec, floor_e):
    lt = np.maximum(0, Tn - dl)
    dxx = np.maximum(0, lt - hold) / dec
    return (Tn > dl) * mixf(1 - np.exp(-lt / rise), np.minimum(1, lt / rise), lin) * (floor_e + np.exp(-xpow(dxx, p)))


def reset_lp2(x, cut, q):
    kk = math.tan(math.pi * min(cut, 0.45 * SR) / SR)
    nn = 1 / (1 + kk / q + kk * kk)
    b0 = kk * kk * nn
    a1 = 2 * (kk * kk - 1) * nn
    a2 = (1 - kk / q + kk * kk) * nn
    return lfilter([b0, 2 * b0, b0], [1, a1, a2], x)


def tpt_hp(x, hz):
    """VILLAIN's 18 Hz body HP: TPT SVF, reset at the onset (sample 0)."""
    g = math.tan(math.pi * hz / SR)
    a1 = 1 / (1 + g * (g + 1.414213562))
    a2 = g * a1
    a3 = g * a2
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.empty_like(x)
    _L.tpt_hp(a1, a2, a3, x.ctypes.data_as(_D), y.ctypes.data_as(_D), len(x))
    return y


def knobs_full(knobs):
    k = dict(KNOB_DEFAULTS)
    k.update(knobs or {})
    return {n: min(max(float(v), KNOB_CLIP[n][0]), KNOB_CLIP[n][1]) for n, v in k.items()}


def modal_bank(e, T, Tw, fmul, bendE, inv_decE, inv_attE, bodyE, knockE, tickE, GA, GB, bl, slots=range(24)):
    """returns (bank_sum, bank_sum_sh, bank_sum_dc, per-slot list of (m, g))"""
    bank = np.zeros_like(T); bsh = np.zeros_like(T); bdc = np.zeros_like(T)
    for i in slots:
        f = e[f"b{i}_f"]
        t = np.maximum(0, T - e[f"b{i}_t0"])
        tw = np.maximum(0, Tw - e[f"b{i}_t0"])
        sh = max(-0.9 * f, bendE * e[f"b{i}_g"] * f)
        se = np.exp(-tw * e[f"b{i}_itau"])
        hz = f + sh * se
        phase = f * tw + sh * e[f"b{i}_tau"] * (1 - se)
        if f"b{i}_g2" in e:
            sh2 = max(-0.9 * f, bendE * e[f"b{i}_g2"] * f)
            se2 = np.exp(-tw * e[f"b{i}_itau2"])
            hz = hz + sh2 * se2
            phase = phase + sh2 * e[f"b{i}_tau2"] * (1 - se2)
        hz = fmul * hz
        grp = knockE if i >= 14 else bodyE
        lc = np.clip((mixf(hz, fmul * f, e["c_lcb"]) - e["c_lcf"]) / e["c_lcw"], 0, 1)
        g = (e[f"b{i}_amp"] * grp * np.exp(-t * inv_decE * e[f"b{i}_idec"])
             * (1 - np.exp(-t * inv_attE * e[f"b{i}_irise"]))
             * np.clip((0.48 * SR - hz) / (0.08 * SR), 0, 1)
             * (1 - e[f"b{i}_MLC"] * (1 - lc))
             * (1 - e[f"b{i}_MGA"] * (1 - GA)) * (1 - e[f"b{i}_MGB"] * (1 - GB)))
        m = g * np.sin(TWOPI * fmul * phase + e[f"b{i}_ph"])
        bank += m
        bsh += e[f"b{i}_MSM"] * m
        bdc += e[f"b{i}_MSM"] * g * g
    for side, w in (("A", 1 - bl), ("B", bl)):
        for j in range(5):
            p = f"x{side}{j}_"
            f = e[p + "f"]
            t = np.maximum(0, T - e[p + "t0"])
            sh = max(-0.9 * f, bendE * e[p + "dl"])
            se = np.exp(-t * e[p + "itau"])
            dx = np.exp(-t * inv_decE * e[p + "idec"])
            env = dx * ((1 - e[p + "te"]) * (1 - e[p + "rd"] * np.exp(-t * inv_attE * e[p + "irise"]))
                        + e[p + "te"] * 2.718281828 * t * inv_decE * e[p + "idec"])
            amt = e[p + "wb"] * bodyE + e[p + "wk"] * knockE + e[p + "wt"] * tickE + e[p + "wc"]
            m = (w * e[p + "amp"] * amt * env * (1 - e[p + "ga"] * (1 - GA)) * (1 - e[p + "gb"] * (1 - GB))
                 * np.sin(TWOPI * fmul * (f * t + sh * e[p + "tau"] * (1 - se)) + e[p + "ph"]))
            bank += m
            bsh += e[p + "sm"] * m
    return bank, bsh, bdc


def render_hit(e, vel=1.0, pitch=261.6255653, knobs=None, n=48000, white=None):
    K = knobs_full(knobs)
    age = np.arange(n) / SR
    vel = min(max(vel, 0), 1)
    v05 = math.sqrt(vel); v025 = math.sqrt(v05); v075 = v05 * v025
    v15 = vel * v05; v175 = vel * v075; v2 = vel * vel
    velp = lambda pi: selector(pi, 1.0, v025, v05, v075, vel, v15, v175, v2)
    vcurve = lambda k, pi: (1 - k) + k * velp(pi)
    note = 2 * min(max(pitch / 440, 0.125), 4)
    tuning = note * K["tune"]
    bendE = K["bend"] * vcurve(e["c_vb_k"], e["c_vb_p"])
    decE = K["decay"] * vcurve(e["c_vd_k"], e["c_vd_p"])
    knockE = K["knock"] * vcurve(e["c_vk_k"], e["c_vk_p"])
    tickE = K["tick"] * vcurve(e["c_vt_k"], e["c_vt_p"])
    bodyE = K["body"] * vcurve(e["c_vbo_k"], e["c_vbo_p"])
    attE = K["attack"] / vcurve(e["c_va_k"], e["c_va_p"])
    vpreE, vtexE, vpostE = velp(e["c_vpre"]), velp(e["c_vtex"]), velp(e["c_vpost"])

    T = mixf(age, age * tuning, e["c_samp"])
    fmul = mixf(tuning, 1, e["c_samp"])
    Tw = T + e["c_wk"] * bendE * e["c_wtau"] * (1 - np.exp(-T / e["c_wtau"]))
    len_s = e["c_len"] * K["length"] * mixf(1, max(1, K["decay"]), e["c_len_dsc"])
    hold_s = e["c_hold"] * K["hold"]
    rel_s = e["c_rel"] * K["release"]

    dscA = mixf(1, K["decay"], e["c_gA_dsc"])
    holdA = dscA * ((1 - e["c_gA_hvk"]) + e["c_gA_hvk"] * vel) * (
        e["c_gA_hL"] * len_s + e["c_gA_hH"] * hold_s + e["c_gA_hR"] * rel_s + e["c_gA_hC"])
    relA = max(0.0001, dscA * (e["c_gA_rL"] * len_s + e["c_gA_rR"] * rel_s + e["c_gA_rC"]))
    GA = kick_gate(age, holdA, relA, e["c_gA_wexp"], e["c_gA_wrat"], e["c_gA_wss"], e["c_gA_p"],
                   e["c_gA_tmix"], dscA * e["c_gA_tail"], e["c_gA_floor"])
    dscB = mixf(1, K["decay"], e["c_gB_dsc"])
    holdB = dscB * (e["c_gB_hL"] * len_s + e["c_gB_hH"] * hold_s + e["c_gB_hR"] * rel_s + e["c_gB_hC"])
    relB = max(0.0001, dscB * (e["c_gB_rL"] * len_s + e["c_gB_rR"] * rel_s + e["c_gB_rC"]))
    GB = kick_gate(age, holdB, relB, e["c_gB_wexp"], e["c_gB_wrat"], e["c_gB_wss"], e["c_gB_p"],
                   e["c_gB_tmix"], dscB * e["c_gB_tail"], e["c_gB_floor"])

    bank_sum, bank_sum_sh, bank_sum_dc = modal_bank(
        e, T, Tw, fmul, bendE, 1 / decE, 1 / attE, bodyE, knockE, tickE, GA, GB, e["bl"])
    onset_ramp = smooth01((age - e["c_on_del"]) / e["c_on_s"])

    # growl
    g_shift = max(-0.9 * e["c_gr_f"], bendE * e["c_gr_dl"])
    g_settle = np.exp(-T / e["c_gr_tau"])
    g_ph = TWOPI * fmul * (e["c_gr_f"] * T + g_shift * e["c_gr_tau"] * (1 - g_settle)) + e["c_gr_off"]
    g_hz = fmul * (e["c_gr_f"] + g_shift * g_settle)
    g_aud = np.clip((e["c_gr_cut"] * SR - g_hz) / (0.4 * e["c_gr_cut"] * SR), 0, 1)
    g_sin = np.sin(g_ph)
    g_wave = np.clip(g_sin, -1, e["c_gr_clip"]) - (e["c_gr_a"] + e["c_gr_b"] * g_sin)
    gr_env = ((0.5 * np.exp(-T / (K["decay"] * e["c_gr_d1"])) + 0.5 * np.exp(-T / (K["decay"] * e["c_gr_d2"])))
              * (1 - np.exp(-T / (K["attack"] * e["c_gr_rise"]))))
    growl_sig = K["body"] * K["growl"] * e["c_gr_amp"] * g_aud * gr_env * g_wave

    # burst
    Tb = mixf(age, age * tuning, e["c_bu_samp"])
    bdt = Tb - e["c_bu_tc"]
    bz = bdt / e["c_bu_w"]
    bgw = np.exp(-0.5 * np.minimum(60, bz * bz))
    bhw = 0.5 - 0.5 * np.cos(TWOPI * np.clip(bz, 0, 1))
    burst_sig = tickE * e["c_bu_amp"] * mixf(bgw, bhw, e["c_bu_hann"]) * np.sin(TWOPI * e["c_bu_f"] * bdt + e["c_bu_off"])

    modal = onset_ramp * bank_sum
    bank_all = modal + growl_sig + burst_sig
    bank_sh = onset_ramp * bank_sum_sh + e["c_bu_sm"] * burst_sig
    bank_dc = 0.5 * bank_sum_dc

    # body shaper
    xin = vpreE * bank_sh
    xcl = np.clip(xin, -e["c_sh_ecl"], e["c_sh_ecl"])
    sh_even = K["shape"] * (xcl * xcl * (e["c_sh_ec"] + xcl * (e["c_sh_p3"] + xcl * (e["c_sh_p4"] + xcl * e["c_sh_p5"])))
                            - e["c_sh_ec"] * e["c_sh_ed"] * vpreE * vpreE * bank_dc)
    sh_odd = K["shape"] * e["c_sh_oc"] * (e["c_sh_ow"] * np.tanh(xin / e["c_sh_ow"]) - xin)
    sh_y0 = xin + sh_even + sh_odd
    sh_y1 = mixf(sh_y0, np.sign(sh_y0) * rat_side(np.abs(sh_y0), 1 / max(0.000001, e["c_sh_rg"] * K["shape"]), e["c_sh_rn"]),
                 e["c_sh_rk"])
    sat_gain = max(0.01, e["c_sat_g"] * mixf(1, K["shape"], e["c_sat_on"]))
    sat_ref = math.tanh(e["c_sat_b"])
    sat_norm = 0.8 / (math.tanh(0.8 * sat_gain + e["c_sat_b"]) - sat_ref)
    sh_sat = sat_norm * (np.tanh(sat_gain * sh_y1 + e["c_sat_b"]) - sat_ref)
    shaped = mixf(sh_y1, sh_sat, e["c_sat_on"])

    # noise layers
    if white is None:
        white_u = white_noise(n)
    else:
        white_u = white
    u0 = white_u
    wh = (u0 * 2 - 1) * math.sqrt(SR / 16000)
    air_c, room_c, click_c = K["air_tone"], K["room_tone"], K["click_tone"]
    cr_thr = 1 - e["c_cr_rate"] / SR
    cr_lin = np.maximum(0, u0 - cr_thr) / max(1e-9, 1 - cr_thr)
    crk = K["crackle"] * e["c_cr_amp"] * mixf(cr_lin, cr_lin * cr_lin, e["c_cr_sq"])
    air_band = bandx(wh + e["c_cr_air"] * crk, e["c_air_hp"] * air_c, e["c_air_qh"], e["c_air_m1"], e["c_air_u1"],
                     e["c_air_lp"] * air_c, e["c_air_ql"], e["c_air_m2"], e["c_air_u2"])
    room_band = bandx(wh + e["c_cr_room"] * crk, e["c_room_hp"] * room_c, e["c_room_qh"], e["c_room_m1"], e["c_room_u1"],
                      e["c_room_lp"] * room_c, e["c_room_ql"], e["c_room_m2"], e["c_room_u2"])
    click_band = bandx(wh + e["c_cr_click"] * crk, e["c_click_hp"] * click_c, e["c_click_qh"], e["c_click_m1"],
                       e["c_click_u1"], e["c_click_lp"] * click_c, e["c_click_ql"], e["c_click_m2"], e["c_click_u2"])
    aux_band = bandx(wh + e["c_cr_aux"] * crk, e["c_aux_hp"], e["c_aux_qh"], e["c_aux_m1"], e["c_aux_u1"],
                     e["c_aux_lp"], e["c_aux_ql"], e["c_aux_m2"], e["c_aux_u2"])
    crk_band = band2(crk, e["c_cr_hp"], 0.707, e["c_cr_lp"], e["c_cr_q"])
    hum_nz = biquad(wh, min(e["c_hu_bw"], 0.2 * SR), 0.707, 1, 0)
    g_band = bandx(wh, e["c_gt_hp"], 0.707, e["c_gt_m1"], e["c_gt_u1"], e["c_gt_lp"], e["c_gt_q"], 0, e["c_gt_u2"])

    Tn = mixf(age, age * tuning, e["c_nsamp"])
    air_env = layer_env(Tn, e["c_air_del"], e["c_air_rise"], e["c_air_lin"], e["c_air_hold"], e["c_air_p"],
                        e["c_air_dec"] * K["air_time"], e["c_air_floor"])
    room_env = layer_env(Tn, e["c_room_del"], e["c_room_rise"], e["c_room_lin"], e["c_room_hold"], e["c_room_p"],
                         e["c_room_dec"] * K["room_time"], e["c_room_floor"])
    click_env = layer_env(Tn, e["c_click_del"], e["c_click_rise"], e["c_click_lin"], e["c_click_hold"], e["c_click_p"],
                          e["c_click_dec"] * K["click_time"], e["c_click_floor"])
    aux_env = layer_env(Tn, e["c_aux_del"], e["c_aux_rise"], e["c_aux_lin"], e["c_aux_hold"], e["c_aux_p"],
                        e["c_aux_dec"], e["c_aux_floor"])
    hum_s = np.sin(TWOPI * tuning * e["c_hu_hz"] * age)
    hum_env = (K["hum"] * e["c_hu_amt"] * (1 - np.exp(-age / e["c_hu_rise"])) * np.exp(-age / e["c_hu_ms"])
               * (hum_s + e["c_hu_h2"] * (1 - 2 * hum_s * hum_s)))
    gt_env = (smooth01((T - e["c_gt_del"]) / e["c_gt_ramp"]) * np.exp(-np.maximum(0, age - e["c_gt_del"]) / e["c_gt_dec"])
              * gpow(GA, e["c_gt_ga"]))
    air_gain = K["air"] * e["c_air_amt"] * velp(e["c_air_nv"]) * gpow(GA, e["c_air_ga"]) * gpow(GB, e["c_air_gb"])
    room_gain = K["room"] * e["c_room_amt"] * velp(e["c_room_nv"]) * gpow(GA, e["c_room_ga"]) * gpow(GB, e["c_room_gb"])
    click_gain = K["click"] * e["c_click_amt"] * velp(e["c_click_nv"]) * gpow(GA, e["c_click_ga"]) * gpow(GB, e["c_click_gb"])
    aux_gain = K["floor"] * e["c_aux_amt"] * velp(e["c_aux_nv"]) * gpow(GA, e["c_aux_ga"]) * gpow(GB, e["c_aux_gb"])
    Tf = mixf(age, age * tuning, e["c_fd_samp"])
    fade_s = min(0.9 * len_s, max(e["c_fd_s"], e["c_fd_frac"] * len_s))
    fu = smooth01((Tf - (len_s - fade_s)) / fade_s)
    out_gate = gpow(GA, e["c_oA"]) * gpow(GB, e["c_oB"])
    air_sig = air_gain * air_env * air_band
    room_sig = room_gain * room_env * room_band
    click_sig = click_gain * click_env * click_band

    # grit
    prev = np.concatenate([[0.0], bank_all[:-1]])
    g_der = (prev - bank_all) * (SR / (TWOPI * e["c_gt_derhz"] * tuning))
    inc = min(e["c_gt_shhz"] * tuning, 0.45 * SR) / SR
    held = np.empty(n)
    ph = 0.0
    cur = bank_all[0]
    for i in range(n):
        nxt = (0.0 if i == 0 else ph) + inc
        wrap = nxt >= 1
        ph = nxt - wrap
        if wrap or i == 0:
            cur = bank_all[i]
        held[i] = cur
    g_src = mixf(mixf(bank_all, np.abs(bank_all), e["c_gt_abs"]), 4 * np.maximum(0, g_der) ** 2, e["c_gt_der"])
    grit_sig = K["grit"] * e["c_gt_amt"] * gt_env * biquad(
        e["c_gt_sh"] * (held - bank_all) + e["c_gt_am"] * g_src * g_band, e["c_gt_post"], 0.707, 1, 1)

    bark = (K["shape"] * e["c_bk_amt"] * v05 * np.clip((T - e["c_bk_del"]) / e["c_bk_ramp"], 0, 1)
            * biquad(np.tanh(e["c_bk_g"] * bank_all + e["c_bk_b"]), e["c_bk_hp"], 0.707, 1, 1))
    body_raw = vpreE * (bank_all - bank_sh) + shaped + vpreE * bark
    body_sig = mixf(body_raw, tpt_hp(body_raw, e["c_bhp_hz"]), e["c_bhp_on"])

    pulse = e["c_pu_amt"] * K["click"] * bandx(click_env, e["c_pu_hp"] * click_c, e["c_pu_qh"], e["c_pu_m1"], 1,
                                                e["c_pu_lp"] * click_c, e["c_pu_ql"], 0, e["c_pu_u2"])
    crk_own = e["c_cr_own"] * np.clip((age - e["c_cr_del"]) / e["c_cr_ramp"], 0, 1) * crk_band
    hum_sig = hum_env * hum_nz
    aux_sig = aux_gain * (aux_env * aux_band + hum_sig + crk_own + pulse)
    texture = air_sig + room_sig + click_sig + aux_sig
    source = body_sig + vpreE * (grit_sig + vtexE * (1 - e["c_np"]) * texture)

    # recording chain
    ceil_p = e["c_ceil"] * K["ceiling"]
    ceil_n = ceil_p * e["c_cneg"]
    cx = source + e["c_coff"]
    c_up = (cx >= 0).astype(float)
    c_mag = np.abs(cx)
    c_cs = mixf(ceil_n, ceil_p, c_up)
    c_ka = mixf(e["c_kan"], e["c_kap"], c_up)
    c_kb = mixf(e["c_kbn"], e["c_kbp"], c_up)
    c_side = mixf(knee_side(c_mag, c_ka * c_cs, np.maximum(0.000001, c_kb * c_cs)), rat_side(c_mag, c_cs, e["c_kn"]), e["c_kr"])
    clipped = (2 * c_up - 1) * c_side - e["c_coff"]
    recorded = mixf(e["c_dry"] * source, clipped, e["c_cmix"] * K["clip_amt"])
    dc_r = 1 - TWOPI * e["c_dc_hz"] / SR
    dcd = lfilter([1, -1], [1, -dc_r], recorded)
    rl1 = reset_lp2(dcd, e["c_rl_hz"], e["c_rl_q1"])
    rl2 = reset_lp2(rl1, e["c_rl_hz"], e["c_rl_q2"])
    rec_lp = mixf(dcd, mixf(rl1, rl2, e["c_rl_two"]), e["c_rl_mix"])
    with_post = rec_lp + vpreE * vtexE * e["c_np"] * texture
    split = min(e["c_dr_hz"] * mixf(1, tuning, e["c_dr_track"]), 0.2 * SR)
    upper = biquad(with_post, split, 0.707, 1, 1)
    amount = (K["drive"] - 1) / 4
    saturated = np.tanh(K["drive"] * upper) / math.sqrt(K["drive"])
    colour = biquad(amount * (saturated - upper), split, 0.707, 1, 1)
    limited = with_post + colour
    output = vpostE * e["c_lvl"] * K["level"] * (1 - fu) * out_gate * limited
    out = np.clip(output, -e["c_out_clip"], e["c_out_clip"])
    return {"out": out, "modal": modal, "bank_all": bank_all, "body_sig": body_sig, "source": source,
            "texture": texture, "limited": limited,
            "_parts": {"growl": growl_sig, "burst": burst_sig, "shaped": shaped, "bark": bark, "grit": grit_sig,
                       "GA": GA, "GB": GB, "fu": fu, "onset_ramp": onset_ramp, "bank_sh": bank_sh}}
