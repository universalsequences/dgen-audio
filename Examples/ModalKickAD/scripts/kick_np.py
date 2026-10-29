#!/usr/bin/env python3
"""numpy twin of the LATENT KICK voice (what the realtime instrument and the trainer render).

Decoded per-kick params (kick_layout, natural units, before the transform knobs) +
transform knobs + velocity + pitch -> audio. The math is VILLAIN Kick's kick-hit macro,
verbatim, with three changes:
  1. VILLAIN's discrete power selectors (velocity powers, gate powers, kick-gate x^1,2,4,8)
     are continuous exponents x^e (exact at VILLAIN's table values);
  2. the texture layers (air/room/click/aux/hum/crackle/pulse) and grit are replaced by
     16 fixed unit-RMS band noises (ModalSnareAD/scripts/e3proto/NOISE_BANDS.md) with E1 envelopes
     (2-stage decay + modal-envelope coupling), x vel^ve x GA^ga;
  3. the modal low-cut fade, the onset ramp and the S&H/slew grit are dropped
     (ablations: no effect on any of the 33 kicks / replaced by the coupling term).

mode="full"    : instrument (incl. bark HP, 18 Hz body HP, DC blocker, record LP, drive split)
mode="trainer" : frame-parallel subset rendered by the dgen trainer (no IIR stages)

    P = targets.teacher(env)   # or kick_layout.decode_u(u)
    y = render(P, knobs={}, vel=1, pitch=C4, bands=noise_bands(48000))["out"]
"""
import math
import numpy as np
import kick_layout as L
from villain_kick_np import (lfilter, biquad, tpt_hp, reset_lp2, knee_side, rat_side, smooth01, mixf,
                             white_noise, SR, TWOPI)

C4 = 261.6255653
KNOB_DEF = {n: d for n, d, a, b in L.KNOBS}
KNOB_RNG = {n: (a, b) for n, d, a, b in L.KNOBS}


def knobs_full(knobs=None):
    k = dict(KNOB_DEF)
    k.update(knobs or {})
    return {n: min(max(float(v), KNOB_RNG[n][0]), KNOB_RNG[n][1]) for n, v in k.items()}


def band_coeffs(fc, sr=SR, q=L.BAND_Q):
    w0 = 2 * math.pi * fc / sr
    al = math.sin(w0) / (2 * q)
    return al / (1 + al), -2 * math.cos(w0) / (1 + al), (1 - al) / (1 + al)


def band_norm(fc, sr=SR, n=1 << 16):
    b0, a1, a2 = band_coeffs(fc, sr)
    h = np.zeros(n); h[0] = 1
    for _ in range(L.BAND_STAGES):
        h = lfilter([b0, 0, -b0], [1, a1, a2], h)
    return 1.0 / math.sqrt(float(np.sum(h * h)))


def noise_bands(n=48000, white_u=None):
    """[n, 16] unit-RMS band noise exactly as the instrument computes it on a fresh voice:
    white = sqrt(12)(u - 0.5), u = dgen xorshift32 seed 1, zero filter state at t=0."""
    u = white_noise(n) if white_u is None else white_u
    w = 3.4641016 * (u - 0.5)
    out = np.empty((n, L.NB))
    for i, fc in enumerate(L.BAND_FC):
        b0, a1, a2 = band_coeffs(fc)
        y = w
        for _ in range(L.BAND_STAGES):
            y = lfilter([b0, 0, -b0], [1, a1, a2], y)
        out[:, i] = band_norm(fc) * y
    return out


def vpow(x, e):
    """x^e, x >= 0 (0^0 = 1, 0^e = 0)"""
    x = np.asarray(x, float)
    return np.where(x > 0, np.exp(e * np.log(np.maximum(x, 1e-30))), 1.0 if e == 0 else 0.0)


def kick_gate(T, hold, rel, wexp, wrat, wss, pe, tmix, tail, floor_g):
    gx = np.maximum(0, T - hold)
    x = gx / rel
    xp = vpow(x, pe)
    shaped = wexp * np.exp(-xp) + wrat / (1 + xp) + wss * (1 - smooth01(x))
    g = mixf(shaped, np.exp(-gx / tail), tmix)
    return floor_g + (1 - floor_g) * g


def hit_values(P, knobs, vel, pitch):
    """everything the instrument computes on the event clock (per hit, post-decoder)"""
    K = knobs_full(knobs)
    vel = min(max(vel, 0.0), 1.0)
    vp = lambda e: float(vpow(vel, e))
    vc = lambda k, e: (1 - k) + k * vp(e)
    h = {"K": K, "vel": vel}
    note = 2 * min(max(pitch / 440, 0.125), 4)
    h["tuning"] = note * K["tune"]
    h["bendE"] = K["bend"] * vc(P["vb_k"], P["vb_e"])
    h["decE"] = K["decay"] * vc(P["vd_k"], 1.0)
    h["knockE"] = K["knock"] * vc(P["vk_k"], P["vk_e"])
    h["tickE"] = K["tick"] * vc(P["vt_k"], P["vt_e"])
    h["bodyE"] = K["body"]
    h["attE"] = K["attack"] / vc(P["va_k"], 1.0)
    h["vpreE"], h["vtexE"], h["vpostE"] = vp(P["vpre_e"]), vp(P["vtex_e"]), vp(P["vpost_e"])
    # slot transform knobs (VILLAIN: spread/beat about the sub's log f; tilt on log decay; drop_time on tau)
    lf0 = np.log(np.asarray(P["mode_f"], float))
    lf_sub = lf0[0]
    lf = lf0.copy()
    lf[1] = lf_sub + K["beat"] * (lf0[1] - lf_sub)
    lf[2:] = lf_sub + K["spread"] * (lf0[2:] - lf_sub)
    h["f"] = np.exp(lf)
    h["idec"] = np.exp(-(np.log(np.asarray(P["mode_dec"])) + K["tilt"] * (lf - lf_sub)))
    h["tau"] = K["drop_time"] * np.asarray(P["mode_tau"])
    h["tau2"] = K["drop_time"] * np.asarray(P["mode_tau2"])
    h["irise"] = 1 / np.asarray(P["mode_rise"])
    h["grpE"] = np.where(np.arange(L.K) >= 14, h["knockE"], h["bodyE"])
    len_s = P["len"] * K["length"] * mixf(1, max(1, K["decay"]), P["len_dsc"])
    hold_s = P["hold"] * K["hold"]
    rel_s = P["rel"] * K["release"]
    dscA = mixf(1, K["decay"], P["gA_dsc"])
    h["holdA"] = dscA * ((1 - P["gA_hvk"]) + P["gA_hvk"] * vel) * (
        P["gA_hL"] * len_s + P["gA_hH"] * hold_s + P["gA_hR"] * rel_s + P["gA_hC"])
    h["relA"] = max(1e-4, dscA * (P["gA_rR"] * rel_s + P["gA_rC"]))
    h["tailA"] = dscA * P["gA_tail"]
    dscB = mixf(1, K["decay"], P["gB_dsc"])
    h["holdB"] = dscB * (P["gB_hL"] * len_s + P["gB_hH"] * hold_s + P["gB_hC"])
    h["relB"] = max(1e-4, dscB * (P["gB_rL"] * len_s + P["gB_rR"] * rel_s + P["gB_rC"]))
    h["tailB"] = dscB * 1.0
    h["len_s"] = len_s
    h["fade_s"] = min(0.9 * len_s, max(P["fd_s"], P["fd_frac"] * len_s))
    return h


_HUM = {}


def hum_noise(n, white_u=None):
    """VILLAIN's hum noise: dgen biquad LP 12 Hz (2pi/44100 rad/Hz quirk) of (2u-1) sqrt(sr/16000)"""
    if n not in _HUM:
        u = white_noise(n) if white_u is None else white_u
        _HUM[n] = biquad((u * 2 - 1) * math.sqrt(SR / 16000), 12.0, 0.707, 1, 0)
    return _HUM[n]


def render(P, knobs=None, vel=1.0, pitch=C4, n=48000, bands=None, mode="full", taps=False, bands_white=None):
    h = hit_values(P, knobs, vel, pitch)
    K = h["K"]
    age = np.arange(n) / SR
    tuning = h["tuning"]
    T = mixf(age, age * tuning, P["samp"])
    fmul = mixf(tuning, 1, P["samp"])
    bendE, inv_decE, inv_attE = h["bendE"], 1 / h["decE"], 1 / h["attE"]
    Tw = T + P["wk"] * bendE * P["wtau"] * (1 - np.exp(-T / P["wtau"]))
    GA = kick_gate(age, h["holdA"], h["relA"], P["gA_wexp"], P["gA_wrat"], P["gA_wss"], P["gA_pe"],
                   P["gA_tmix"], h["tailA"], P["gA_floor"])
    GB = kick_gate(age, h["holdB"], h["relB"], P["gB_wexp"], 0.0, P["gB_wss"], P["gB_pe"], 0.0, h["tailB"], 0.0)

    # ---- modal bank (24 slots, 1 or 2 drops) ----
    bank = np.zeros(n); bsh = np.zeros(n); bdc = np.zeros(n); esq = np.zeros(n)
    for i in range(L.K):
        f = h["f"][i]
        t = np.maximum(0, T - P["mode_t0"][i])
        tw = np.maximum(0, Tw - P["mode_t0"][i])
        sh = max(-0.9 * f, bendE * P["mode_glide"][i] * f)
        se = np.exp(-tw / h["tau"][i])
        hz = f + sh * se
        ph = f * tw + sh * h["tau"][i] * (1 - se)
        if i < L.K2:
            sh2 = max(-0.9 * f, bendE * P["mode_glide2"][i] * f)
            se2 = np.exp(-tw / h["tau2"][i])
            hz = hz + sh2 * se2
            ph = ph + sh2 * h["tau2"][i] * (1 - se2)
        hz = fmul * hz
        g = (P["mode_amp"][i] * h["grpE"][i] * np.exp(-t * inv_decE * h["idec"][i])
             * (1 - np.exp(-t * inv_attE * h["irise"][i]))
             * np.clip((0.48 * SR - hz) / (0.08 * SR), 0, 1)
             * (1 - P["mode_mga"][i] * (1 - GA)) * (1 - P["mode_mgb"][i] * (1 - GB)))
        m = g * np.sin(TWOPI * fmul * ph + P["mode_phase"][i])
        bank += m; bsh += P["mode_msm"][i] * m; bdc += P["mode_msm"][i] * g * g; esq += g * g
    # ---- transients (ticks / pulses) ----
    trans = np.zeros(n)
    for j in range(L.J):
        f = P["tr_f"][j]
        t = np.maximum(0, T - P["tr_t0"][j])
        sh = max(-0.9 * f, bendE * P["tr_dl"][j])
        se = np.exp(-t / P["tr_tau"][j])
        env = np.exp(-t * inv_decE / P["tr_dec"][j]) * (1 - np.exp(-t * inv_attE / P["tr_rise"][j]))
        amt = P["tr_wb"][j] * h["bodyE"] + P["tr_wt"][j] * h["tickE"]
        m = (P["tr_amp"][j] * amt * env * (1 - P["tr_mgb"][j] * (1 - GB))
             * np.sin(TWOPI * fmul * (f * t + sh * P["tr_tau"][j] * (1 - se)) + P["tr_phase"][j]))
        trans += m; bsh += P["tr_msm"][j] * m
    modal = bank + trans
    # ---- growl ----
    gr_f = 35.0
    g_shift = max(-0.9 * gr_f, bendE * P["gr_dl"])
    g_settle = np.exp(-T / P["gr_tau"])
    g_ph = TWOPI * fmul * (gr_f * T + g_shift * P["gr_tau"] * (1 - g_settle)) + P["gr_off"]
    g_hz = fmul * (gr_f + g_shift * g_settle)
    g_aud = np.clip((0.02 * SR - g_hz) / (0.4 * 0.02 * SR), 0, 1)
    g_sin = np.sin(g_ph)
    g_wave = np.clip(g_sin, -1, P["gr_clip"]) - (P["gr_a"] + P["gr_b"] * g_sin)
    gr_env = ((0.5 * np.exp(-T / (K["decay"] * P["gr_d1"])) + 0.5 * np.exp(-T / (K["decay"] * P["gr_d2"])))
              * (1 - np.exp(-T / (K["attack"] * 0.004))))
    growl = K["body"] * K["growl"] * P["gr_amp"] * g_aud * gr_env * g_wave
    # ---- burst ----
    Tb = mixf(age, age * tuning, P["bu_samp"])
    bdt = Tb - P["bu_tc"]
    bz = bdt / P["bu_w"]
    bgw = np.exp(-0.5 * np.minimum(60, bz * bz))
    bhw = 0.5 - 0.5 * np.cos(TWOPI * np.clip(bz, 0, 1))
    burst = h["tickE"] * P["bu_amp"] * mixf(bgw, bhw, P["bu_hann"]) * np.sin(TWOPI * P["bu_f"] * bdt + P["bu_off"])
    bank_all = modal + growl + burst
    bank_sh = bsh + P["bu_sm"] * burst
    # ---- body shaper ----
    vpreE = h["vpreE"]
    xin = vpreE * bank_sh
    xcl = np.clip(xin, -P["sh_ecl"], P["sh_ecl"])
    sh_even = K["shape"] * (xcl * xcl * (P["sh_ec"] + xcl * (P["sh_p3"] + xcl * (P["sh_p4"] + xcl * P["sh_p5"])))
                            - P["sh_ec"] * P["sh_ed"] * vpreE * vpreE * 0.5 * bdc)
    sh_odd = K["shape"] * P["sh_oc"] * (P["sh_ow"] * np.tanh(xin / P["sh_ow"]) - xin)
    y0 = xin + sh_even + sh_odd
    y1 = mixf(y0, np.sign(y0) * rat_side(np.abs(y0), 1 / max(1e-6, P["sh_rg"] * K["shape"]), 4.0), P["sh_rk"])
    sat_gain = max(0.01, P["sat_g"] * mixf(1, K["shape"], P["sat_on"]))
    sat_ref = math.tanh(P["sat_b"])
    sat_norm = 0.8 / (math.tanh(0.8 * sat_gain + P["sat_b"]) - sat_ref)
    shaped = mixf(y1, sat_norm * (np.tanh(sat_gain * y1 + P["sat_b"]) - sat_ref), P["sat_on"])
    body_raw = vpreE * (bank_all - bank_sh) + shaped
    if mode == "full":
        bark = (K["shape"] * P["bk_amt"] * math.sqrt(h["vel"]) * np.clip((T - P["bk_del"]) / 0.012, 0, 1)
                * biquad(np.tanh(8 * bank_all + 0.3), 300, 0.707, 1, 1))
        body_raw = body_raw + vpreE * bark
        body = mixf(body_raw, tpt_hp(body_raw, 18.0), P["bhp_on"])
    else:
        body = body_raw
    # ---- 16-band noise ----
    if bands is None:
        bands = noise_bands(n)
    Tn = mixf(age, age * tuning, P["nsamp"])
    tn = np.maximum(0, Tn - P["noise_onset"])
    att = 1 - np.exp(-tn / P["noise_rise"])
    A = np.asarray(P["band_amp"]); Lt = np.asarray(P["band_tail"])
    ngate = vpow(GA, P["noise_ga"]) * vpow(GB, P["noise_gb"])
    cgate = vpow(GA, P["couple_ga"])
    env = (ngate[:, None] * att[:, None] * A[None, :] * ((1 - Lt)[None, :] * np.exp(-vpow(tn[:, None] / np.asarray(P["band_d1"])[None, :], P["noise_q"]))
                                                          + Lt[None, :] * np.exp(-tn[:, None] / np.asarray(P["band_d2"])[None, :]))
           + cgate[:, None] * np.asarray(P["band_couple"])[None, :] * np.sqrt(esq + 1e-10)[:, None])
    noise = K["noise"] * float(vpow(h["vel"], P["noise_ve"])) * np.sum(env * bands[:n], axis=1)
    if P["hum_amt"] > 0:
        hum_s = np.sin(TWOPI * tuning * 92.0 * age)
        hum = (P["hum_amt"] * float(vpow(h["vel"], P["hum_ve"])) * (1 - np.exp(-age / 0.1)) * np.exp(-age / 0.22)
               * (hum_s + P["hum_h2"] * (1 - 2 * hum_s * hum_s)) * hum_noise(n, bands_white))
        noise = noise + K["noise"] * ngate * hum
    tex = vpreE * h["vtexE"] * noise
    source = body + (1 - P["np"]) * tex
    # ---- recording chain ----
    ceil_p = P["ceil"] * K["ceiling"]
    ceil_n = ceil_p * P["cneg"]
    cx = source + P["coff"]
    up = (cx >= 0).astype(float)
    mag = np.abs(cx)
    cs = mixf(ceil_n, ceil_p, up)
    ka = mixf(P["kan"], P["kap"], up)
    kb = mixf(P["kbn"], P["kbp"], up)
    side = mixf(knee_side(mag, ka * cs, np.maximum(1e-6, kb * cs)), rat_side(mag, cs, P["kn"]), P["kr"])
    clipped = (2 * up - 1) * side - P["coff"]
    recorded = mixf(P["dry"] * source, clipped, P["cmix"] * K["clip_amt"])
    if mode == "full":
        dc_r = 1 - TWOPI * P["dc_hz"] / SR
        dcd = lfilter([1, -1], [1, -dc_r], recorded)
        rl1 = reset_lp2(dcd, P["rl_hz"], P["rl_q1"])
        rl2 = reset_lp2(rl1, P["rl_hz"], P["rl_q2"])
        rec = mixf(dcd, mixf(rl1, rl2, P["rl_two"]), P["rl_mix"])
    else:
        rec = recorded
    with_post = rec + P["np"] * tex
    if mode == "full" and K["drive"] > 1:
        split = min(180.0 * mixf(1, tuning, P["dr_track"]), 0.2 * SR)
        upper = biquad(with_post, split, 0.707, 1, 1)
        amount = (K["drive"] - 1) / 4
        colour = biquad(amount * (np.tanh(K["drive"] * upper) / math.sqrt(K["drive"]) - upper), split, 0.707, 1, 1)
        limited = with_post + colour
    else:
        limited = with_post
    Tf = mixf(age, age * tuning, P["fd_samp"])
    fu = smooth01((Tf - (h["len_s"] - h["fade_s"])) / h["fade_s"])
    out_gate = vpow(GA, P["oA_e"])
    output = h["vpostE"] * P["lvl"] * K["level"] * (1 - fu) * out_gate * limited
    out = np.clip(output, -P["out_clip"], P["out_clip"])
    r = {"out": out, "modal": modal}
    if taps:
        r.update({"bank_all": bank_all, "body_sig": body, "source": source, "noise": tex, "limited": limited,
                  "GA": GA, "GB": GB, "fu": fu, "esq": esq, "ngate": ngate, "cgate": cgate})
    return r
