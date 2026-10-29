#!/usr/bin/env python3
"""Teacher rows: VILLAIN Kick block-rate values -> latent-kick template params (kick_layout).

Everything except the noise bands is an EXACT re-expression of VILLAIN's values
(discrete selector indices -> the continuous exponents they select). The 16 band
envelopes are fitted to VILLAIN's texture + grit (rendered by the VILLAIN twin), in an
analysis filterbank with the band crosstalk inverted, as 2-stage decays + modal-envelope
coupling (NNLS per band over a (d1, d2) grid), shared onset/rise by grid.

    python3 targets.py            # -> kick/data/teacher.json (all manifest renders)
"""
import json, math, os, sys
import numpy as np
import kick_layout as L
from kick_common import DATA
from kick_table import BlockRate
import villain_kick_np as V
import kick_np as KN

VP = [0, .25, .5, .75, 1, 1.5, 1.75, 2]          # VILLAIN velocity power index 1..8
GP = [0, .5, 1]                                   # gate power index 1..3
XP = [1, 2, 4, 8]                                 # kick-gate shape power index 1..4
vp_e = lambda i: VP[int(round(i)) - 1]
gp_e = lambda i: GP[int(round(i)) - 1]
xp_e = lambda i: XP[int(round(i)) - 1]


def teacher_nonnoise(e):
    S = range(L.K)
    P = {
        "mode_f": [math.exp(e[f"b{i}_lf0"]) for i in S],
        "mode_dec": [math.exp(e[f"b{i}_ld0"]) for i in S],
        "mode_amp": [e[f"b{i}_amp"] for i in S],
        "mode_glide": [e[f"b{i}_g"] for i in S],
        "mode_tau": [math.exp(e[f"b{i}_lt"]) for i in S],
        "mode_rise": [math.exp(e[f"b{i}_lr"]) for i in S],
        "mode_t0": [e[f"b{i}_t0"] for i in S],
        "mode_phase": [math.atan2(math.sin(e[f"b{i}_ph"]), math.cos(e[f"b{i}_ph"])) for i in S],
        "mode_mga": [e[f"b{i}_MGA"] for i in S],
        "mode_mgb": [e[f"b{i}_MGB"] for i in S],
        "mode_msm": [e[f"b{i}_MSM"] for i in S],
        "mode_glide2": [e[f"b{i}_g2"] for i in range(L.K2)],
        "mode_tau2": [math.exp(e[f"b{i}_lt2"]) for i in range(L.K2)],
    }
    # transients: VILLAIN plays A with weight (1-bl) and B with bl; the latent voice has ONE set.
    # At blend 0 (all presets) that is exactly A. For blends we take the dominant side.
    side = "A" if e["bl"] < 0.5 else "B"
    w = (1 - e["bl"]) if side == "A" else e["bl"]
    X = lambda k: [e[f"x{side}{j}_{k}"] for j in range(L.J)]
    P.update({"tr_f": X("f"), "tr_dec": [1 / v for v in X("idec")], "tr_dl": X("dl"),
              "tr_tau": [1 / v for v in X("itau")], "tr_rise": [1 / v for v in X("irise")],
              "tr_amp": [w * v for v in X("amp")],
              "tr_phase": [math.atan2(math.sin(v), math.cos(v)) for v in X("ph")], "tr_t0": X("t0"),
              "tr_wb": X("wb"), "tr_wt": X("wt"), "tr_mgb": X("gb"), "tr_msm": X("sm")})
    c = lambda k: e["c_" + k]
    for k in ["gr_amp", "gr_dl", "gr_tau", "gr_off", "gr_clip", "gr_a", "gr_b", "gr_d1", "gr_d2",
              "bu_amp", "bu_samp", "bu_tc", "bu_w", "bu_hann", "bu_f", "bu_off", "bu_sm",
              "samp", "wk", "wtau", "nsamp", "fd_samp", "vb_k", "vd_k", "vk_k", "vt_k", "va_k",
              "len", "len_dsc", "hold", "rel", "fd_s", "fd_frac",
              "gA_dsc", "gA_hvk", "gA_hL", "gA_hH", "gA_hR", "gA_hC", "gA_rR", "gA_rC", "gA_wexp", "gA_wrat",
              "gA_wss", "gA_tmix", "gA_tail", "gA_floor", "gB_dsc", "gB_hL", "gB_hH", "gB_hC", "gB_rL", "gB_rR",
              "gB_rC", "gB_wexp", "gB_wss", "sh_ec", "sh_p3", "sh_p4", "sh_p5", "sh_ed", "sh_ecl", "sh_oc",
              "sh_ow", "sh_rk", "sh_rg", "sat_on", "sat_g", "sat_b", "bk_amt", "bk_del", "bhp_on",
              "ceil", "cneg", "coff", "kap", "kbp", "kan", "kbn", "kr", "kn", "dry", "cmix",
              "dc_hz", "rl_hz", "rl_q1", "rl_q2", "rl_two", "rl_mix", "dr_track", "lvl", "out_clip", "np"]:
        P[k] = c(k)
    P["gr_off"] = math.atan2(math.sin(P["gr_off"]), math.cos(P["gr_off"]))
    P.update({"vb_e": vp_e(c("vb_p")), "vk_e": vp_e(c("vk_p")), "vt_e": vp_e(c("vt_p")),
              "vpre_e": vp_e(c("vpre")), "vtex_e": vp_e(c("vtex")), "vpost_e": vp_e(c("vpost")),
              "gA_pe": xp_e(c("gA_p")), "gB_pe": xp_e(c("gB_p")), "oA_e": gp_e(c("oA"))})
    return P


# ---------------------------------------------------------------------------- noise fit
SR = 48000.0
HOP = 96                     # 2 ms analysis frames
SMOOTH = 96
_BANDS = None
_M = None


def _analysis(x):
    """[frames, 16] band power (unit-RMS band filters), 2 ms frames, ~2 ms smoothing"""
    out = []
    for fc in L.BAND_FC:
        b0, a1, a2 = KN.band_coeffs(fc)
        y = x
        for _ in range(L.BAND_STAGES):
            y = V.lfilter([b0, 0, -b0], [1, a1, a2], y)
        y = KN.band_norm(fc) * y
        p = np.convolve(y * y, np.ones(SMOOTH) / SMOOTH, mode="same")
        out.append(p[::HOP])
    return np.array(out).T


def _setup():
    global _BANDS, _M
    if _BANDS is None:
        _BANDS = KN.noise_bands(48000)
        # crosstalk: analysis power of each unit band noise, long-time average
        _M = np.array([_analysis(_BANDS[:, b])[50:].mean(0) for b in range(L.NB)]).T   # [analysis b', source b]
    return _BANDS, _M


D1 = np.exp(np.linspace(np.log(2.5e-4), np.log(0.4), 22))
D2 = np.exp(np.linspace(np.log(0.012), np.log(1.8), 16))
ONS = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
RISES = [1e-6, 1e-4, 4e-4, 1e-3, 2.5e-3, 6e-3, 0.015]


TAUS = [0.0003, 0.001, 0.003, 0.009, 0.027, 0.08, 0.24, 0.7]


def fit_noise(target, esq, gate, P_nn, ons=ONS, rises=RISES, cgate=None, q=1.0):
    """Waveform least squares (the band noise and VILLAIN's texture are linear filters of the
    SAME seed-1 white): target ~ sum_b N_b(t) [sum_k c_bk att(t) e^(-t/tau_k) + c_bE E(t)],
    over a grid of shared (onset, rise); then each band's (noise-free) envelope curve is
    reduced to the template's 2-stage decay + coupling by NNLS over a (d1, d2) grid."""
    bands, _ = _setup()
    n = len(target)
    t = np.arange(n) / SR
    E = np.sqrt(esq + 1e-10)
    best = None
    dec = 8                                                   # LS on every 8th sample (6000 rows) for the grid
    for on in ons:
        tn = np.maximum(0, t - on)
        for ri in rises:
            att = 1 - np.exp(-tn / ri)
            basis = np.stack([att * np.exp(-(tn / tk) ** q) for tk in TAUS[:4]] + [att * np.exp(-tn / tk) for tk in TAUS[4:]] + [E], 1)
            gb_ = np.concatenate([np.repeat(gate[:, None], len(TAUS), 1), cgate[:, None]], 1)
            X = (bands[:, :, None] * (gb_ * basis)[:, None, :]).reshape(n, -1)
            c, *_ = np.linalg.lstsq(X[::dec], target[::dec], rcond=None)
            r = float(np.sum((X @ c - target) ** 2))
            if best is None or r < best[0]:
                best = (r, on, ri, c, basis)
    r, onset, rise, c, basis = best
    c = c.reshape(L.NB, -1)
    tn = np.maximum(0, t - onset)
    att = 1 - np.exp(-tn / rise)
    A, D1o, Lt, D2o, C = [], [], [], [], []
    sub_t = slice(0, n, 24)
    for b in range(L.NB):
        env = np.maximum(basis[:, :-1] @ c[b, :-1], 0)[sub_t]
        if env.max() < 1e-7:
            A.append(1e-6); D1o.append(0.01); Lt.append(0.0); D2o.append(0.1); C.append(1e-6)
            continue
        wgt = 1 / (env + 0.02 * env.max())
        y = env * wgt
        bb = (1e30, 1e-6, 0.01, 0.0, 0.1, 1e-6)
        a, d1, lt, d2, _ = _reduce_curve(basis[:, :-1] @ c[b, :-1], att, tn, E * 0, sub_t, q)
        A.append(a); D1o.append(d1); Lt.append(lt); D2o.append(d2); C.append(max(c[b, -1], 1e-6))
    out = {"band_amp": A, "band_d1": D1o, "band_tail": Lt, "band_d2": D2o, "band_couple": C,
           "noise_rise": max(rise, 1e-6), "noise_onset": onset}
    return refine_energy(out, target, E, att, tn, gate, cgate, q=q)


def _param_env(p, att, tn, E, gate, cgate, q=1.0):
    """(gated) band envelope, and its two parts ungated"""
    A = np.asarray(p["band_amp"]); Lt = np.asarray(p["band_tail"])
    ea = att[:, None] * A * ((1 - Lt) * np.exp(-(tn[:, None] / np.asarray(p["band_d1"])) ** q)
                             + Lt * np.exp(-tn[:, None] / np.asarray(p["band_d2"])))
    ec = np.asarray(p["band_couple"]) * E[:, None]
    return gate[:, None] * ea + cgate[:, None] * ec, ea, ec


def _reduce_curve(env, att, tn, E, sub_t, q=1.0):
    """noise-free envelope curve -> (A, d1, L, d2, C) by NNLS over the (d1, d2) grid"""
    env = np.maximum(env, 0)[sub_t]
    if env.max() < 1e-7:
        return 1e-6, 0.01, 0.0, 0.1, 1e-6
    wgt = 1 / (env + 0.05 * env.max())
    y = env * wgt
    bb = (1e30, 1e-6, 0.01, 0.0, 0.1, 1e-6)
    for d1 in D1:
        p1 = (att * np.exp(-(tn / d1) ** q))[sub_t]
        for d2 in D2:
            if d2 <= d1 * 1.5:
                continue
            p2 = (att * np.exp(-tn / d2))[sub_t]
            X = np.stack([p1, p2, E[sub_t]], 1) * wgt[:, None]
            for sub in ((0, 1, 2), (0, 1), (0, 2), (1, 2), (0,), (1,), (2,)):
                cc, *_ = np.linalg.lstsq(X[:, sub], y, rcond=None)
                if np.any(cc < 0):
                    continue
                rr = np.sum((X[:, sub] @ cc - y) ** 2)
                if rr < bb[0]:
                    full = np.zeros(3); full[list(sub)] = cc
                    bb = (rr, full[0] + full[1], d1, full[1] / max(full[0] + full[1], 1e-30), d2, full[2])
                break
    _, a, d1, lt, d2, cpl = bb
    return min(max(a, 1e-6), 2.0), d1, lt, d2, max(cpl, 1e-6)


def _band_power(x, smooth):
    """[n, NB] analysis power of x through the unit-RMS band filters, moving-average smoothed"""
    out = []
    for fc in L.BAND_FC:
        b0, a1, a2 = KN.band_coeffs(fc)
        y = x
        for _ in range(L.BAND_STAGES):
            y = V.lfilter([b0, 0, -b0], [1, a1, a2], y)
        y = KN.band_norm(fc) * y
        out.append(np.convolve(y * y, np.ones(smooth) / smooth, mode="same"))
    return np.array(out).T


def refine_energy(p, target, E, att, tn, gate, cgate, iters=5, smooth=480, q=1.0):
    """analysis-by-synthesis: scale each band's two parts by sqrt(target / model) band power
    (both analysed the same way, realised noise on both sides), then re-reduce the layer part
    to the template's parametric envelope. Fixes the coherent-LS under-estimate of incoherent
    parts (grit AM, crackle)."""
    bands, _ = _setup()
    n = len(target)
    sub_t = slice(0, n, 24)
    Pt = _band_power(target, smooth)
    floor = 1e-6 * Pt.max()
    for _ in range(iters):
        env, ea, ec = _param_env(p, att, tn, E, gate, cgate, q)
        Pm = _band_power(np.sum(env * bands[:n], 1), smooth)
        corr = np.sqrt(np.clip((Pt + floor) / (Pm + floor), 1 / 16, 16))
        red = [_reduce_curve(ea[:, b] * corr[:, b], att, tn, E * 0, sub_t, q) for b in range(L.NB)]
        # the coupling part keeps its shape: scale by the energy-weighted mean correction
        wc = (cgate[:, None] * ec) ** 2
        cs = np.sum(wc * corr, 0) / np.maximum(np.sum(wc, 0), 1e-30)
        p = dict(p)
        for k, key in enumerate(("band_amp", "band_d1", "band_tail", "band_d2")):
            p[key] = [r[k] for r in red]
        p["band_couple"] = list(np.maximum(np.asarray(p["band_couple"]) * np.where(np.sum(wc, 0) > 0, cs, 1), 1e-6))
    return p


LAYERS = {"air": ("air", "room", "click", "floor"), "room": ("room",), "click": ("click",), "aux": ("floor",)}


def noise_exps(e, W):
    """velocity / GA / GB exponents of the band noise = those of the texture layer (or grit)
    carrying the most energy in VILLAIN's render"""
    off = {"air": 0, "room": 0, "click": 0, "floor": 0, "grit": 0}
    best = (-1, None)
    for lay, knob in (("air", "air"), ("room", "room"), ("click", "click"), ("aux", "floor")):
        r = V.render_hit(e, white=W, knobs={**off, knob: 1})
        en = float(np.sum(r["texture"] ** 2))
        if en > best[0]:
            best = (en, lay)
    r = V.render_hit(e, white=W, knobs={**off, "grit": 1})
    eg = float(np.sum(r["_parts"]["grit"] ** 2))
    if eg > best[0]:
        return 0.0, gp_e(e["c_gt_ga"]), 0.0, "grit", 1.0
    lay = best[1]
    return vp_e(e[f"c_{lay}_nv"]), gp_e(e[f"c_{lay}_ga"]), gp_e(e[f"c_{lay}_gb"]), lay, float(xp_e(e[f"c_{lay}_p"]))


def teacher(e, W=None):
    P = teacher_nonnoise(e)
    if W is None:
        W = V.white_noise(48000)
    r = V.render_hit(e, white=W)
    target = r["texture"] + r["_parts"]["grit"]
    vp = lambda i: VP[int(round(i)) - 1]
    P["hum_amt"] = e["c_aux_amt"] * e["c_hu_amt"]
    P["hum_h2"] = e["c_hu_h2"]
    P["hum_ve"] = vp(e["c_aux_nv"])
    ve, ga, gb, lay, q = noise_exps(e, W)
    if P["hum_amt"] > 0:                  # hum lives in the aux layer: it sets the shared noise gate
        ve, ga, gb = vp(e["c_aux_nv"]), gp_e(e["c_aux_ga"]), gp_e(e["c_aux_gb"])
    P.update({"noise_ve": ve, "noise_ga": ga, "noise_gb": gb, "couple_ga": gp_e(e["c_gt_ga"])})
    t = KN.render({**P, **_dummy_noise(), "noise_ve": ve, "noise_ga": ga, "noise_gb": gb, "couple_ga": P["couple_ga"]},
                  taps=True, mode="trainer")
    target = target - t["noise"]          # the exact hum part (dummy bands are ~0)
    P.update(fit_noise(target, t["esq"], t["ngate"], P, cgate=t["cgate"], q=q))
    P["noise_q"] = q
    P["noise_layer"] = lay
    return P


def _dummy_noise():
    return {"band_amp": [1e-6] * L.NB, "band_d1": [0.01] * L.NB, "band_tail": [0.0] * L.NB,
            "band_d2": [0.1] * L.NB, "band_couple": [1e-6] * L.NB, "noise_rise": 1e-3, "noise_onset": 0.0,
            "noise_ve": 1.0, "noise_ga": 0.0, "noise_gb": 0.0, "couple_ga": 0.0, "noise_q": 1.0}



def clamp_to_layout(P):
    """values exactly as the decoder can express them (squash round trip)"""
    u = L.encode(P)
    return L.decode_u(u), u


if __name__ == "__main__":
    man = json.load(open(os.path.join(DATA, "villain", "manifest.json")))
    br = BlockRate()
    W = V.white_noise(48000)
    ids = sys.argv[1:]
    out = {}
    for it in man["items"]:
        if it["kind"] not in ("fit", "blend", "pc") or (ids and it["id"] not in ids):
            continue
        if it["kind"] == "fit" and not it["id"].endswith("v100"):
            continue   # velocity is VILLAIN's own law post-decoder: one row per kick
        e = br.eval(**it["params"])
        P = teacher(e, W)
        Pc, u = clamp_to_layout(P)
        lay = P.pop("noise_layer")
        out[it["id"]] = {"params": it["params"], "noise_layer": lay,
                         "P": {k: (list(map(float, v)) if isinstance(v, (list, np.ndarray)) else float(v))
                               for k, v in P.items()}, "u": [float(x) for x in u]}
        print(it["id"], "ok", flush=True)
    json.dump({"doc": "teacher rows in kick_layout natural units (P) and decoder space (u)", "P_dim": L.P,
               "rows": out}, open(os.path.join(DATA, "teacher.json"), "w"))
