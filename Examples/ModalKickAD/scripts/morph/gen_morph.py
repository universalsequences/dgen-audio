#!/usr/bin/env python3
"""Latent Kick Morph dsp generator (v2): the shipped Latent Kick voice (ModalKickAD/scripts/gen_dsp.py,
the VILLAIN Kick table-row template) re-decoded on a hop-rate clock so the latent can travel from
START (z1..z4) to END (b1..b4) while a hit rings, the e8 (Latent Snare Morph) way.

ONE decoder chain on one event clock `dclk` (the hit, or a hop tick while the morph moves or a
watched control changed). Its input is z(t) = zA + w (zB - zA) (w = 0 on the hit, so the hit
frame decodes START). The same post-decoder expressions are latched twice:
  (latch X hit)  : the STRIKE, held for the whole hit (amplitudes, phases, onsets, rises, drop
                   times, time warp, transients / tick, growl, burst, shaper, clipper shape, hum)
  (latch X dclk) : the RING at z(t): slot frequencies + drop depths, slot decays, slot levels
                   (follow), noise band levels / decays / coupling, clipper ceiling / mix / dry.
(dgen keeps the voice in one fused SIMD loop only when every tensor latch comes from ONE
decoder chain: a second chain split it into taped scalar loops, 61 -> 207 us/block. v1 =
gen_morph_v1.py.) Gates / length: decoded at START (main chain) and END (a small hit-clock
decode of those columns only, scalar outputs); the two envelopes are crossfaded by w(t) at
audio rate and clamped never to re-open.
Morph corrections on top of the closed-form voice (all exactly 0 / 1 with START = END):
  phase  Cp += dTw (hz(t) - hz_A(t)),  hz = fmul (f + sh e^-tw/tau + sh2 e^-tw/tau2)
  decay  Ld += (idec(t) - idec_A) dt, env e^-(t idec_A + Ld)   (per slot; noise fast / slow too)
  level  follow: a slot / band may only FALL from its strike level toward z(t)'s (e8 rule)
Frequencies, drop depths and log levels ramp linearly across the hop; decay rates step per hop.

usage: gen_morph.py OUT_DIR [--hop 128] [--always] [--tap] [--exp a,b]"""
import argparse, json, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
KICK = MAD.stage("kick")
MAD.use("kick")
import kick_layout as L  # noqa: E402
import gen_dsp as G  # noqa: E402

SRC = os.path.join(KICK, "stage", "Latent Kick")  # the A/B Latent Kick (scripts/build_instrument.py)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--hop", type=int, default=128)
ap.add_argument("--always", action="store_true", help="decode on every hop tick (no change gating)")
ap.add_argument("--tap", action="store_true", help="research outputs: 4 modal, 5 noise")
ap.add_argument("--exp", default="", help="perf experiments (comma list): noramp, noacc, noclamp")
a = ap.parse_args()
HOP = a.hop
EXP = set(x for x in a.exp.split(",") if x)
K, K2, J, NB = L.K, L.K2, L.J, L.NB
JP, NBP = 8, 20
D = L.LATENT_D
off = L.offsets()
fmt, tlit, squash_expr = G.fmt, G.tlit, G.squash_expr
KN = {n: (d, lo, hi) for n, d, lo, hi in L.KNOBS}

# Gate / length columns are NOT taken along the path: VILLAIN's gates switch between "never"
# (hC = 100) and finite holds, and in-between latents decode negative holds that would shut the
# kick mid-morph. START's come from the main decode at the hit, END's from a small hit-clock
# decode at z = END (these columns only); the per-sample envelopes are crossfaded by w(t).
GEND = ["len", "len_dsc", "hold", "rel", "fd_s", "fd_frac", "fd_samp", "oA_e",
        "gA_dsc", "gA_hvk", "gA_hL", "gA_hH", "gA_hR", "gA_hC", "gA_rR", "gA_rC", "gA_tail",
        "gA_wexp", "gA_wrat", "gA_wss", "gA_pe", "gA_tmix", "gA_floor",
        "gB_dsc", "gB_hL", "gB_hH", "gB_hC", "gB_rL", "gB_rR", "gB_rC", "gB_wexp", "gB_wss", "gB_pe"]
GATE_SHAPE = ["fd_samp", "oA_e", "gA_wexp", "gA_wrat", "gA_wss", "gA_pe", "gA_tmix", "gA_floor",
              "gB_wexp", "gB_wss", "gB_pe"]
# knobs whose change re-decodes mid-hit (the ring reads them live); the rest act per hit
RING_KNOBS = ["spread", "beat", "tilt", "decay", "bend", "ceiling", "clip_amt"]

g_cols, g_lines = [], []
for nm in GEND:
    oo, wd, sq, lo, hi = off[nm]
    o_ = len(g_cols); g_cols.append(oo)
    g_lines.append(f"(def nb-{nm} {squash_expr(f'(sum (shrink nb-u @ranges [0:1,{o_}:{o_ + 1}]))', sq, lo, hi)})")
NG = len(g_cols)

o = []
w = o.append


def tl(name, val, clk="hit"):
    w(f"(def {name} (latch {val} {clk}))")


def sl(name, val, clk="hit"):
    w(f"(def {name} (event-latch {val} {clk}))")


def ramp_t(name, tgt, n):
    """linear ramp from the value at the last hop boundary to the dclk target over one hop"""
    if "noramp" in EXP:
        w(f"(def {name} {tgt})"); return
    # one history: step 1/(samples left in the hop) of the remaining distance, so the value
    # moves linearly from where it was at the hop boundary and lands on the target at hop end
    w(f"(make-tensor-history {name}-h @shape [1 {n}])")
    w(f"(def {name}-p (read-tensor-history {name}-h))")
    w(f"(def {name} (gswitch hit {tgt} (+ {name}-p (* (- {tgt} {name}-p) hop-step))))")
    w(f"(write-tensor-history {name}-h {name})")


def ramp_s(name, tgt):
    if "noramp" in EXP:
        w(f"(def {name} {tgt})"); return
    w(f"(make-history {name}-h)")
    w(f"(def {name}-p (read-history {name}-h))")
    w(f"(def {name} (gswitch hit {tgt} (+ {name}-p (* (- {tgt} {name}-p) hop-step))))")
    w(f"(write-history {name}-h {name})")


def clamp_down(name, val):
    """never rises within a hit (a morph may not re-open a gate or a fade)"""
    if "noclamp" in EXP:
        w(f"(def {name} {val})"); return
    w(f"(make-history {name}-h)")
    w(f"(def {name} (gswitch hit {val} (min {val} (read-history {name}-h))))")
    w(f"(write-history {name}-h {name})")


def gate_block(P):
    """VILLAIN's gate / length scalars from decoded values {P}name (event-held knobs e-*)"""
    return f"""(def {P}G-len (* {P}len e-length (mix 1 (max 1 e-decay) {P}len_dsc)))
(def {P}G-holds (* {P}hold e-hold))
(def {P}G-rels (* {P}rel e-release))
(def {P}G-dscA (mix 1 e-decay {P}gA_dsc))
(def {P}G-holdA (* {P}G-dscA (+ (- 1 {P}gA_hvk) (* {P}gA_hvk e-vel)) (+ (* {P}gA_hL {P}G-len) (* {P}gA_hH {P}G-holds) (* {P}gA_hR {P}G-rels) {P}gA_hC)))
(def {P}G-relA (max 0.0001 (* {P}G-dscA (+ (* {P}gA_rR {P}G-rels) {P}gA_rC))))
(def {P}G-dscB (mix 1 e-decay {P}gB_dsc))
(def {P}G-holdB (* {P}G-dscB (+ (* {P}gB_hL {P}G-len) (* {P}gB_hH {P}G-holds) {P}gB_hC)))
(def {P}G-relB (max 0.0001 (* {P}G-dscB (+ (* {P}gB_rL {P}G-len) (* {P}gB_rR {P}G-rels) {P}gB_rC))))
(def {P}G-fade (min (* 0.9 {P}G-len) (max {P}fd_s (* {P}fd_frac {P}G-len))))
(def {P}G-tailA (* {P}G-dscA {P}gA_tail))
(def {P}G-tailB {P}G-dscB)"""


w(f""";; Latent Kick Morph -- GENERATED by ModalKickAD/scripts/morph/gen_morph.py (hop {HOP}), do not edit.
;;
;; The Latent Kick voice (a 4->128->128->534 tanh MLP decoding one VILLAIN Kick table row:
;; 24 role-ordered modal slots with VILLAIN's pitch-drop law, 5 transient sines, growl, burst,
;; body shaper, gates, recording chain, 18 noise bands + hum), re-decoded on a hop-rate clock
;; so the kick travels from START to END while it rings:
;;   z(t) = zA + w (zB - zA),  w = amount * p^(4^curve),  p = clip(age / morph, 0, 1)
;; zA = live (mod z1..z4), zB = live (mod b1..b4), sampled every {HOP} samples (one hop ahead).
;; One decoder chain runs on `dclk` (the hit, or a hop tick while the morph moves or a watched
;; control changed); the hit frame decodes START. Its outputs are latched on the hit (the
;; STRIKE: slot amplitudes, phases, onsets, rises, drop times, time warp, transients / tick,
;; growl, burst, shaper, hum) and on dclk (the RING: slot frequencies + drop depths, slot
;; decays, slot / band levels for follow, band decays + coupling, clipper ceiling / mix).
;; Frequency changes integrate into a per-slot phase correction, decay changes into a per-slot
;; log-amplitude correction; both are exactly 0 with START = END, where the voice is the
;; closed-form Latent Kick. follow: a slot / band level may only fall toward z(t).
;; Gates / length: START's and END's (END decoded once at the hit) crossfaded by w(t).
;; Note (2 pitch / 440) and tune are latched at the hit, like the other latent instruments.

(def gate (in 1 @name gate))
(def pitch (in 2 @name pitch))
(def velocity (in 3 @name velocity))
(def trigger (in 4 @name trigger))
(def clock (in 5 @name clock))""")
for i in range(1, D + 1):
    w(f"(def mod{i} (in {5 + i} @name mod{i} @modulator {i}))")
for i in range(1, D + 1):
    w(f"(param z{i} @default 0 @min -3 @max 3 @mod true @mod-mode additive)")
for i in range(1, D + 1):
    w(f"(param b{i} @default 0 @min -3 @max 3 @mod true @mod-mode additive)")
w("(param morph @default 400 @min 10 @max 4000 @unit ms)")
w("(param curve @default 0 @min -1 @max 1)")
w("(param amount @default 1 @min 0 @max 1)")
w("(param follow @default 1 @min 0 @max 1)")
for n, d, lo, hi in L.KNOBS:
    w(f"(param {n} @default {d} @min {lo} @max {hi})")
w("""
(defmacro onset-of (trig)
  (make-history previous)
  (def high (gt trig 0.5))
  (def rising (* high (lte (read-history previous) 0.5)))
  (write-history previous high)
  rising)

(defmacro smooth01 (x) (* (clip x 0 1) (clip x 0 1) (- 3 (* 2 (clip x 0 1)))))
;; x^e for x >= 0 (VILLAIN's discrete power selectors, as continuous exponents)
(defmacro vpow (x e) (gswitch (gt x 0) (exp (* e (log (max x 0.000000000000000000000000000001)))) (lte e 0)))
(defmacro kick-gate (T hold rel wexp wrat wss pe tmix tail floor_g)
  (def gx (max 0 (- T hold)))
  (def x (/ gx rel))
  (def xp (exp (* pe (log (max x 0.000000000000000000000000000001)))))
  (def shaped (+ (* wexp (exp (- 0 xp))) (* wrat (/ 1 (+ 1 xp))) (* wss (- 1 (smooth01 x)))))
  (def g (mix shaped (exp (/ (- 0 gx) tail)) tmix))
  (+ floor_g (* (- 1 floor_g) g)))
(defmacro knee-side (m k w) (+ (min m k) (* w (tanh (/ (max 0 (- m k)) w)))))
(defmacro rat-side (m c n) (* m (exp (/ (- 0 (log (+ 1 (exp (* n (log (max (/ m c) 0.000000001))))))) n))))
(defmacro reset-lp2 (x cut q reset)
  (make-history s1)
  (make-history s2)
  (def kk (tan (/ (* 3.14159265 (min cut (* 0.45 samplerate))) samplerate)))
  (def nn (/ 1 (+ 1 (/ kk q) (* kk kk))))
  (def b0 (* kk kk nn))
  (def a1 (* 2 (- (* kk kk) 1) nn))
  (def a2 (* (+ (- 1 (/ kk q)) (* kk kk)) nn))
  (def keep (- 1 reset))
  (def y1 (* keep (read-history s1)))
  (def y2 (* keep (read-history s2)))
  (def y (+ (* b0 x) y1))
  (write-history s1 (+ (* 2 b0 x) (- y2 (* a1 y))))
  (write-history s2 (- (* b0 x) (* a2 y)))
  y)

(def hit (onset-of trigger))
;; seconds since the last onset (0 on the onset frame); silent before the first hit, after 8 s
(make-history elapsed)
(make-history armed)
(def played (max hit (read-history armed)))
(write-history armed played)
(def age-samples (gswitch hit 0 (min (* 8 samplerate) (+ 1 (read-history elapsed)))))
(write-history elapsed age-samples)
(def alive (* played (lt age-samples (* 8 samplerate))))
(def age (* alive (/ age-samples samplerate)))""")
w(f"""
;; hop counter restarts on the hit; hop-wrap marks every hop boundary (ramps restart there)
(make-history hopc)
(def hop-n (gswitch hit 0 (% (+ (read-history hopc) 1) {HOP})))
(write-history hopc hop-n)
(def hop-wrap (eq hop-n 0))
(def hop-step (/ 1 (- {HOP} hop-n)))""")
if a.always:
    w("(def dclk (max hit (* hop-wrap alive)))")
else:
    coef = [1.0, 1.3127, 1.7351, 2.2193, 2.9137, 3.3719, 4.1413, 4.7713, 5.3171, 6.1379, 6.7231, 7.3991]
    terms = [f"(* {coef[i]} z{i + 1})" for i in range(4)] + [f"(* {coef[4 + i]} b{i + 1})" for i in range(4)]
    terms += [f"(* {coef[8 + i]} mod{i + 1})" for i in range(4)]
    kc = [0.000917, 0.7193, 0.8317, 0.9173] + [0.311 + 0.1031 * i for i in range(len(RING_KNOBS))]
    for c, n in zip(kc, ["morph", "curve", "amount", "follow"] + RING_KNOBS):
        terms.append(f"(* {c:.6g} {n})")
    w(";; A tick decodes only while the morph moves (START and END can differ) or a watched control has")
    w(";; changed since the last decode. Clocks must not read (mod p): knobs and raw modN inputs only.")
    w("(def w-sig (+ " + " ".join(terms) + "))")
    w("""(make-history w-last)
(def w-changed (eq (eq w-sig (read-history w-last)) 0))
(def ab-differ (gt (+ (abs (- z1 b1)) (abs (- z2 b2)) (abs (- z3 b3)) (abs (- z4 b4))
                      (abs mod1) (abs mod2) (abs mod3) (abs mod4)) 0))""")
    w(f"(def morphing (* (gt amount 0) ab-differ (lt age (+ (* morph 0.001) (/ {2 * HOP} samplerate)))))")
    w("(def dclk (max hit (* hop-wrap alive (max w-changed morphing))))")
    w("(write-history w-last (gswitch dclk w-sig (read-history w-last)))")

w(f"""
;; ==== the decoder (dclk, z = z(t); z(t) = START on the hit frame) =====================
(def q-hit (event-hold hit dclk))
(def q-age (event-hold age dclk))""")
for i in range(1, D + 1):
    w(f"(def q-a{i} (event-hold (clip (mod z{i}) -3 3) dclk))")
for i in range(1, D + 1):
    w(f"(def q-b{i} (event-hold (clip (mod b{i}) -3 3) dclk))")
w("""(def q-mt (event-hold (clip morph 1 10000) dclk))
(def q-curve (event-hold (clip curve -1 1) dclk))
(def q-amt (event-hold (clip amount 0 1) dclk))
(def e-vel (event-hold (clip velocity 0 1) dclk))
(def e-pitch (event-hold pitch dclk))""")
for n, d, lo, hi in L.KNOBS:
    w(f"(def e-{n} (event-hold (clip {n} {lo} {hi}) dclk))")
w(f""";; morph position one hop ahead (the ramp then arrives on time); exactly 0 on the hit
(def q-p (* (- 1 q-hit) (clip (/ (+ q-age (/ {HOP} samplerate)) (* q-mt 0.001)) 0 1)))
(def q-w (* q-amt (gt q-p 0) (exp (* (exp (* q-curve 1.3862943611198906)) (log (max q-p 1e-9))))))""")
for i in range(1, D + 1):
    w(f"(def e-z{i} (+ q-a{i} (* q-w (- q-b{i} q-a{i}))))")
for name, (f, shp) in G.WEIGHT_FILES.items():
    w(f'(def {name.replace(".", "-")} (tensor @shape [{shp[0]} {shp[1]}] @file "{f}"))')
for i in range(D):
    oh = " ".join("1" if j == i else "0" for j in range(D))
    w(f"(def onehot{i} (tensor @shape [1 {D}] @data [{oh}]))")
w("(def dec-x (+ " + " ".join(f"(* e-z{i + 1} onehot{i})" for i in range(D)) + "))")
w("(def dec-h1 (tanh (+ (matmul dec-x l1-W) l1-b)))")
w("(def dec-h2 (tanh (+ (matmul dec-h1 l2-W) l2-b)))")
w("(def dec-u (+ (matmul dec-h2 l3-W) l3-b))")
for name, (oo, wd, sq, lo, hi) in off.items():
    u = f"(shrink dec-u @ranges [0:1,{oo}:{oo + wd}])"
    if wd == 1:
        u = f"(sum {u})"
    w(f"(def r-{name} {squash_expr(u, sq, lo, hi)})")

beat_oh = tlit([1 if i == 1 else 0 for i in range(K)])
spread_m = tlit([1 if i >= 2 else 0 for i in range(K)])
knock_m = tlit([1 if i >= 14 else 0 for i in range(K)])
sub_oh = tlit([1] + [0] * (K - 1))
norms = G.norms_table()
nt = {sr: tlit(norms[sr] + [1.0] * (NBP - NB)) for sr in G.NORM_RATES}
pad24 = f"@padding [0:0,0:{K - K2}]"
padJ = f"@padding [0:0,0:{JP - J}]"
padN = f"@padding [0:0,0:{NBP - NB}]"
dumN = tlit([0] * NB + [1] * (NBP - NB))
w(f"""
;; ---- post-decoder: VILLAIN's velocity curves, note tracking and transform knobs ----
(def e-lv (log (max e-vel 0.0001)))
(defmacro velp (e) (* (gt e-vel 0) (exp (* e e-lv))))
(defmacro vcurve (k e) (+ (- 1 k) (* k (velp e))))
(def e-note (* 2 (clip (/ e-pitch 440) 0.125 4)))
(def e-tuning (* e-note e-tune))
(def e-bendE (* e-bend (vcurve r-vb_k r-vb_e)))
(def e-decE (* e-decay (vcurve r-vd_k 1)))
(def e-knockE (* e-knock (vcurve r-vk_k r-vk_e)))
(def e-tickE (* e-tick (vcurve r-vt_k r-vt_e)))
(def e-bodyE e-body)
(def e-attE (/ e-attack (vcurve r-va_k 1)))
(def e-vpreE (velp r-vpre_e))
(def e-vtexE (velp r-vtex_e))
(def e-vpostE (velp r-vpost_e))
(def e-lf0 (log r-mode_f))
(def e-lfsub (sum (shrink e-lf0 @ranges [0:1,0:1])))
(def e-kmul (+ (* e-beat {beat_oh}) (* e-spread {spread_m})))
(def e-kmul0 (+ e-kmul {sub_oh}))
(def e-lf (+ e-lfsub (* e-kmul0 (- e-lf0 e-lfsub))))
(def e-f (exp e-lf))
(def e-idec (/ (exp (* -1 (+ (log r-mode_dec) (* e-tilt (- e-lf e-lfsub))))) e-decE))
(def e-tau (* e-drop_time r-mode_tau))
(def e-sh (max (* -0.9 e-f) (* e-bendE r-mode_glide e-f)))
(def e-grp (+ (* e-bodyE (- 1 {knock_m})) (* e-knockE {knock_m})))
(def e-f4 (shrink e-f @ranges [0:1,0:{K2}]))
(def e-sh2 (max (* -0.9 e-f4) (* e-bendE r-mode_glide2 e-f4)))
(def e-tau2 (* e-drop_time r-mode_tau2))
(def e-tsh (max (* -0.9 r-tr_f) (* e-bendE r-tr_dl)))
(def e-nvel (* e-noise (velp r-noise_ve)))
(def e-hvel (* e-noise (velp r-hum_ve) r-hum_amt))
;; START gates / length (from the main decode; latched on the hit)
{gate_block("r-")}
""")
# post-decoder expressions shared by the strike and the ring latches
SH = {
    "f": "e-f", "sh": "e-sh", "sh2": f"(pad e-sh2 {pad24})", "idec": "e-idec", "lam": "(log r-mode_amp)",
    "id1": f"(+ (pad (/ 1 r-band_d1) {padN}) {dumN})",
    "id2": f"(+ (pad (/ 1 r-band_d2) {padN}) {dumN})",
    "lf": f"(pad (log (max (* r-band_amp (- 1 r-band_tail)) 1e-30)) {padN})",
    "ls": f"(pad (log (max (* r-band_amp r-band_tail) 1e-30)) {padN})",
    "cpl": f"(pad r-band_couple {padN})",
}
i_strike = len(o)
w(";; ---- STRIKE: latched on the hit ------------------------------------------------")
for nm, v in [
    ("m-f", SH["f"]), ("m-amp", "(* r-mode_amp e-grp)"), ("m-idec", SH["idec"]),
    ("m-irise", "(/ 1 (* r-mode_rise e-attE))"), ("m-t0", "r-mode_t0"), ("m-ph", "r-mode_phase"),
    ("m-sh", SH["sh"]), ("m-tau", "e-tau"), ("m-itau", "(/ 1 e-tau)"),
    ("m-mga", "r-mode_mga"), ("m-mgb", "r-mode_mgb"), ("m-msm", "r-mode_msm"),
    ("m-sh2", SH["sh2"]), ("m-sh2tau2", f"(pad (* e-sh2 e-tau2) {pad24})"),
    ("m-itau2", f"(+ (pad (/ 1 e-tau2) {pad24}) {tlit([0] * K2 + [1] * (K - K2))})"),
    ("m-lamA", SH["lam"]),
    ("x-f", f"(pad r-tr_f {padJ})"), ("x-sh", f"(pad e-tsh {padJ})"), ("x-tau", f"(pad r-tr_tau {padJ})"),
    ("x-itau", f"(pad (/ 1 r-tr_tau) {padJ})"), ("x-idec", f"(pad (/ 1 (* r-tr_dec e-decE)) {padJ})"),
    ("x-irise", f"(pad (/ 1 (* r-tr_rise e-attE)) {padJ})"),
    ("x-amp", f"(pad (* r-tr_amp (+ (* r-tr_wb e-bodyE) (* r-tr_wt e-tickE))) {padJ})"),
    ("x-ph", f"(pad r-tr_phase {padJ})"), ("x-t0", f"(pad r-tr_t0 {padJ})"),
    ("x-mgb", f"(pad r-tr_mgb {padJ})"), ("x-msm", f"(pad r-tr_msm {padJ})"),
    ("n-fast", f"(pad (* r-band_amp (- 1 r-band_tail)) {padN})"),
    ("n-slow", f"(pad (* r-band_amp r-band_tail) {padN})"),
    ("n-id1", SH["id1"]), ("n-id2", SH["id2"]), ("n-lfA", SH["lf"]), ("n-lsA", SH["ls"]),
]:
    tl(nm, v)
scal = {
    "tuning": "e-tuning", "bendE": "e-bendE", "decay": "e-decay", "attack": "e-attack", "body": "e-body",
    "growl": "e-growl", "shape": "e-shape", "tickE": "e-tickE", "vpreE": "e-vpreE", "vtexE": "e-vtexE",
    "vel": "e-vel", "nvel": "e-nvel", "hvel": "e-hvel", "lvlA": "(* r-lvl e-vpostE)", "drive": "e-drive",
}
for k_, v in scal.items():
    sl(f"g-{k_}", v)
for name, (oo, wd, sq, lo, hi) in off.items():
    if wd == 1 and name not in GEND and name not in ("ceil", "cmix", "dry"):
        sl(f"p-{name}", f"r-{name}")
for nm in ["holdA", "relA", "tailA", "holdB", "relB", "tailB", "len", "fade"]:
    sl(f"A-{nm}", f"r-G-{nm}")
for nm in GATE_SHAPE:
    sl(f"A-{nm}", f"r-{nm}")

i_end = len(o)
w(f"""
;; ---- END gates / length: a small hit-clock decode at z = END (these columns only) ----
{chr(10).join(f"(def nb-z{i} (event-hold (clip (mod b{i}) -3 3) hit))" for i in range(1, D + 1))}
{chr(10).join(f"(def nbk-{n} (event-hold (clip {n} {KN[n][1]} {KN[n][2]}) hit))" for n in ("length", "decay", "hold", "release"))}
(def nbk-vel (event-hold (clip velocity 0 1) hit))
(def nb-x (+ {" ".join(f"(* nb-z{i + 1} onehot{i})" for i in range(D))}))
(def nb-h1 (tanh (+ (matmul nb-x l1-W) l1-b)))
(def nb-h2 (tanh (+ (matmul nb-h1 l2-W) l2-b)))
(def nb-W3 (tensor @shape [{L.H} {NG}] @file "w3g.json"))
(def nb-b3 (tensor @shape [1 {NG}] @file "b3g.json"))
(def nb-u (+ (matmul nb-h2 nb-W3) nb-b3))
{chr(10).join(g_lines)}
{gate_block("nb-").replace("e-length", "nbk-length").replace("e-decay", "nbk-decay").replace(" e-hold)", " nbk-hold)").replace("e-release", "nbk-release").replace("e-vel", "nbk-vel")}""")
for nm in ["holdA", "relA", "tailA", "holdB", "relB", "tailB", "len", "fade"]:
    sl(f"B-{nm}", f"nb-G-{nm}")
for nm in GATE_SHAPE:
    sl(f"B-{nm}", f"nb-{nm}")

# emit the END decode before the strike latches: dgen orders ready nodes by creation, and an
# END event-hold created after them splits the latches from the voice loop (taped, scalar)
i_end2 = len(o)
o[:] = o[:i_strike] + o[i_end:i_end2] + o[i_strike:i_end]
w(";; ---- RING: the same expressions latched on dclk (z(t)), ramped across the hop ------")
for nm, key in [("L-f", "f"), ("L-sh", "sh"), ("L-sh2", "sh2"), ("L-idec", "idec"), ("L-ml", "lam"),
                ("L-id1", "id1"), ("L-id2", "id2"), ("L-lf", "lf"), ("L-ls", "ls"), ("L-cpl", "cpl")]:
    tl(nm, SH[key], "dclk")
for nm, v in [("ceil", "(* r-ceil e-ceiling)"), ("cmix", "(* r-cmix e-clip_amt)"), ("dry", "r-dry")]:
    sl(f"L-{nm}", v, "dclk")
ramp_t("m-fr", "L-f", K)
ramp_t("m-shr", "L-sh", K)
ramp_t("m-sh2r", "L-sh2", K)
ramp_t("m-lr", "L-ml", K)
ramp_t("n-lfr", "L-lf", NBP)
ramp_t("n-lsr", "L-ls", NBP)
ramp_t("n-cpl", "L-cpl", NBP)
for nm in ["ceil", "cmix", "dry"]:
    ramp_s(f"g-{nm}", f"L-{nm}")
w("""(def g-follow (clip follow 0 1))
;; level follow: each slot / band may only FALL from its strike level toward the level decoded
;; at z(t) (follow 0 = strike levels kept, 1 = down to z(t)'s); a morph never adds energy
(def m-c (exp (* g-follow (min (- m-lr m-lamA) 0))))
(def n-fc (exp (* g-follow (min (- n-lfr n-lfA) 0))))
(def n-sc (exp (* g-follow (min (- n-lsr n-lsA) 0))))
;; morph position at audio rate (gate / length crossfade): exactly 0 on the hit
(def g-w (* (clip amount 0 1) (exp (* (exp (* (clip curve -1 1) 1.3862943611198906))
                                     (log (max (clip (/ age (* (clip morph 1 10000) 0.001)) 0 1) 1e-9))))))

;; ---- time bases, gates -----------------------------------------------------
(def T (mix age (* age g-tuning) p-samp))
(def fmul (mix g-tuning 1 p-samp))
(def tstep (/ (mix 1 g-tuning p-samp) samplerate))
(def Tw-e (exp (/ (- 0 T) p-wtau)))
(def Tw (+ T (* p-wk g-bendE p-wtau (- 1 Tw-e))))
(def dTw (* tstep (+ 1 (* p-wk g-bendE Tw-e))))""")
w(";; START and END gate envelopes, crossfaded by w(t), never re-opening")
for S in "AB":
    w(f"(def GA-{S} (kick-gate age {S}-holdA {S}-relA {S}-gA_wexp {S}-gA_wrat {S}-gA_wss {S}-gA_pe {S}-gA_tmix {S}-tailA {S}-gA_floor))")
    w(f"(def GB-{S} (kick-gate age {S}-holdB {S}-relB {S}-gB_wexp 0 {S}-gB_wss {S}-gB_pe 0 {S}-tailB 0))")
clamp_down("GA", "(mix GA-A GA-B g-w)")
clamp_down("GB", "(mix GB-A GB-B g-w)")
acc_cp = "(gswitch hit 0 (- m-Cp-adv (floor m-Cp-adv)))"
acc_ld = "(gswitch hit 0 (+ (read-tensor-history mLd) (* (gt mt 0) tstep (- L-idec m-idec))))"
w(f"""
;; ---- 24-slot modal bank, VILLAIN drop law (second drop on slots 0..{K2 - 1}) + morph corrections ----
(def mt (max 0 (- T m-t0)))
(def mtw (max 0 (- Tw m-t0)))
(def mse (exp (* (- 0 mtw) m-itau)))
(def mse2 (exp (* (- 0 mtw) m-itau2)))
(def mhz2 (* m-sh2 mse2))
(def mph2 (* m-sh2tau2 (- 1 mse2)))
(def mhz (* fmul (+ m-f (* m-sh mse) mhz2)))
;; instantaneous slot frequency at z(t): the drop law with ramped f / depths, latched time constants
(def mhz-r (* fmul (+ m-fr (* m-shr mse) (* m-sh2r mse2))))
;; phase correction (cycles, wrapped): integral of (hz(t) - hz_A(t)) over the slot's warped time
(make-tensor-history mCp @shape [1 {K}])
(def m-Cp-adv (+ (read-tensor-history mCp) (* (gt mtw 0) dTw (- mhz-r mhz))))
(def m-Cp {acc_cp if "noacc" not in EXP else tlit([0] * K)})
(write-tensor-history mCp m-Cp)
;; decay correction (log domain): Ld = integral of (idec(t) - idec_A) dt once the slot has started
(make-tensor-history mLd @shape [1 {K}])
(def m-Ld {acc_ld if "noacc" not in EXP else tlit([0] * K)})
(write-tensor-history mLd m-Ld)
(def mg (* m-amp (exp (- (* (- 0 mt) m-idec) m-Ld)) (- 1 (exp (* (- 0 mt) m-irise)))
          (clip (/ (- (* 0.48 samplerate) mhz-r) (* 0.08 samplerate)) 0 1)
          (- 1 (* m-mga (- 1 GA))) (- 1 (* m-mgb (- 1 GB))) m-c))
(def mm (* mg (sin (+ (* twopi fmul (+ (* m-f mtw) (* m-sh m-tau (- 1 mse)) mph2)) m-ph (* twopi m-Cp)))))
(def bank (sum mm))
(def bank-msm (sum (* m-msm mm)))
(def bank-dc (sum (* m-msm mg mg)))
(def bank-esq (sum (* mg mg)))
""")
mid = open(os.path.join(MAD.code("kick/morph"), "voice_mid.tmpl")).read()
mid = eval("f" + repr(mid), {"NB": NB, "NBP": NBP, "tlit": tlit, "L": L, "fmt": fmt, "nt": nt})
if "noacc" in EXP:
    mid = mid.replace("(def n-Lf (gswitch hit 0 (+ (read-tensor-history nLf) (* n-dt (- L-id1 n-id1)))))", f"(def n-Lf {tlit([0] * NBP)})")
    mid = mid.replace("(def n-Ls (gswitch hit 0 (+ (read-tensor-history nLs) (* n-dt (- L-id2 n-id2)))))", f"(def n-Ls {tlit([0] * NBP)})")
w(mid)
w(";; output envelope (length fade x out gate) per side, crossfaded, never re-opening")
for S in "AB":
    w(f"(def outenv-{S} (* (- 1 (smooth01 (/ (- (mix age (* age g-tuning) {S}-fd_samp) (- {S}-len {S}-fade)) {S}-fade))) (vpow GA-{S} {S}-oA_e)))")
clamp_down("outenv", "(mix outenv-A outenv-B g-w)")
w("""(def out0 (clip (* level g-lvlA outenv limited) (- 0 p-out_clip) p-out_clip))
;; a retrigger cuts the previous hit: carry its last sample and let it fall over ~1 ms
(make-history last-out)
(make-history carry-h)
(def carry (gswitch hit (read-history last-out) (* (read-history carry-h) (exp (/ -1000 samplerate)))))
(write-history carry-h carry)
(def voice-out (+ (* alive out0) carry))
(write-history last-out voice-out)
(out voice-out 1 @name left)
(out voice-out 2 @name right)
(use-defmacro voice-amp)
(out (voice-amp 0 voice-out voice-out) 3 @name amp @amp true)""")
if a.tap:
    w("(out modal 4 @name modal_tap)")
    w("(out tex 5 @name noise_tap)")

os.makedirs(a.out, exist_ok=True)
for f in ("w1.json", "b1.json", "w2.json", "b2.json", "w3.json", "b3.json", "decoder_layout.json", "codes_std.json"):
    shutil.copy(os.path.join(SRC, f), a.out)
d3 = json.load(open(os.path.join(SRC, "w3.json"))); db = json.load(open(os.path.join(SRC, "b3.json")))
W3 = d3["data"]; P = d3["shape"][1]
json.dump({"shape": [d3["shape"][0], NG], "data": [W3[r * P + c] for r in range(d3["shape"][0]) for c in g_cols]},
          open(os.path.join(a.out, "w3g.json"), "w"))
json.dump({"shape": [1, NG], "data": [db["data"][c] for c in g_cols]}, open(os.path.join(a.out, "b3g.json"), "w"))
for stale in ("w3r.json", "b3r.json"):
    if os.path.exists(os.path.join(a.out, stale)):
        os.remove(os.path.join(a.out, stale))
open(os.path.join(a.out, "dsp.lisp"), "w").write("\n".join(o) + "\n")
open(os.path.join(a.out, "instrument.json"), "w").write('{"version":1,"run_mode":"instrument"}\n')
print("wrote", os.path.join(a.out, "dsp.lisp"), f"hop {HOP}, END gate cols {NG}")
