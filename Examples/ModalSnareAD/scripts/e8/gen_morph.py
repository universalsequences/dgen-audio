#!/usr/bin/env python3
"""E8: Latent Snare Morph dsp generator (v2: split strike / ring decoders).

Builds an INCREMENTAL-state version of the installed E7 Latent Snare voice so the latent can move
from A (z1..z4) to B (b1..b4) while a hit rings. Two event clocks share the trained weights:
  hit clock  : layers 1-2 + the STRIKE columns of layer 3 at z = A (mode amp/phase/rise/t0/glide,
               pre, noise onset/attack/levels, gain + the freq/decay columns the seeds need).
               Latched on the hit: the strike never morphs.
  dclk clock : hit OR hop tick (only while the morph moves or a watched control changed):
               layers 1-2 + the RING columns at z(t) (mode freq, mode/band decay, band coupling,
               drive/clip, length fade, gates, wire knobs). Frequencies, band coupling and every
               ring scalar are ramped linearly across the hop; decay rates step per hop (they only
               bend the slope of a continuous envelope).
Layer-3 columns are re-packed into w3s/b3s (strike) and w3r/b3r (ring); same dot products, so
with A = B the decoded values are bit-identical to the installed decoder's.

usage: gen_morph.py OUT_DIR [--hop N] [--headtap] [--always] [--exp a,b]"""
import argparse, json, os, re, shutil
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e8")
SRC = os.path.join(MAD.ROOT, "e6", "stage", "Latent Snare")  # the A/B Latent Snare (scripts/e6/build_instrument.py)
K, NB = 32, 16

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--hop", type=int, default=256)
ap.add_argument("--headtap", action="store_true", help="out 3 = head, out 4 = wire-out (research)")
ap.add_argument("--always", action="store_true", help="decode on every hop tick (no change gating)")
ap.add_argument("--exp", default="", help="perf experiments (comma list): noramp")
a = ap.parse_args()
EXP = set(x for x in a.exp.split(",") if x)
HOP = a.hop

src = open(os.path.join(SRC, "dsp.lisp")).read()
squash = [l for l in src.splitlines() if l.startswith("(def raw-")]
assert len(squash) == 37, len(squash)
noise_block = src[src.index(";; ---- 16-band noise"):src.index(";; env_b = (1 - e^(-t/rise))")]
svf_macros = src[src.index(";; TPT state-variable filter"):src.index("(def hit (onset-of trigger))")]

SEG = {}
for l in squash:
    nm = l.split()[1][4:]
    i, j = map(int, re.search(r"\[0:1,(\d+):(\d+)\]", l).groups())
    SEG[nm] = (i, j, l)
STRIKE = ["mode_freq", "mode_amp", "mode_decay", "mode_rise", "mode_t0", "mode_phase", "mode_glide",
          "mode_gtau", "band_amp", "band_d1", "band_tail", "band_d2", "noise_rise", "noise_onset", "pre", "gain"]
RING = ["mode_freq", "mode_amp", "band_amp", "band_tail", "gain", "mode_decay", "band_d1", "band_d2", "band_couple", "drive", "drive_amt", "drive_norm",
        "len", "fade", "fade_curve", "gh_depth", "gh_time", "gh_fall", "gd_depth", "gd_time", "gd_fall",
        "gate_noise", "clip_ceil", "clip_mix", "wire_thr", "wire_level", "wire_lo", "wire_hi", "wire_k"]
assert set(STRIKE) | set(RING) == set(SEG), set(SEG) - set(STRIKE) - set(RING)

def load(f):
    d = json.load(open(os.path.join(SRC, f))); return np.array(d["data"]).reshape(d["shape"])

def pack(names, prefix_u, prefix_raw):
    cols, lines = [], []
    for nm in names:
        i, j, l = SEG[nm]
        o = len(cols); cols += list(range(i, j))
        l2 = l.replace(f"[0:1,{i}:{j}]", f"[0:1,{o}:{o + j - i}]").replace("dec-u", prefix_u)
        lines.append(l2.replace("(def raw-", f"(def {prefix_raw}-", 1))
    return cols, lines

s_cols, s_lines = pack(STRIKE, "h-u", "hraw")
r_cols, r_lines = pack(RING, "e-u", "eraw")
NS, NR = len(s_cols), len(r_cols)

L = []
def emit(s=""): L.append(s)

def ramp_t(name, tgt, n):
    """linear ramp from the value at the last hop boundary to the latched target over one hop;
    jumps on the hit. (s = start, held per hop; r = ramp value)"""
    if "noramp" in EXP: emit(f"(def {name} {tgt})"); return
    emit(f"(make-tensor-history {name}-a @shape [1 {n}])")
    emit(f"(make-tensor-history {name}-r @shape [1 {n}])")
    emit(f"(def {name}-s (gswitch hit {tgt} (gswitch hop-wrap (read-tensor-history {name}-r) (read-tensor-history {name}-a))))")
    emit(f"(write-tensor-history {name}-a {name}-s)")
    emit(f"(def {name} (gswitch hit {tgt} (+ {name}-s (* (- {tgt} {name}-s) hop-frac))))")
    emit(f"(write-tensor-history {name}-r {name})")

def sh_t(name, val, n):
    if "nosh" in EXP: emit(f"(def {name} {val})"); return
    emit(f"(make-tensor-history {name}-h @shape [1 {n}])")
    emit(f"(def {name} (gswitch hit {val} (read-tensor-history {name}-h)))")
    emit(f"(write-tensor-history {name}-h {name})")

def sh_s(name, val):
    emit(f"(make-history {name}-h)")
    emit(f"(def {name} (gswitch hit {val} (read-history {name}-h)))")
    emit(f"(write-history {name}-h {name})")

def hold_t(name, val, n):
    """audio-rate hold of a dclk-domain value (read only on dclk frames)."""
    emit(f"(make-tensor-history {name}-h @shape [1 {n}])")
    emit(f"(def {name} (gswitch dclk {val} (read-tensor-history {name}-h)))")
    emit(f"(write-tensor-history {name}-h {name})")

def hold_s(name, val):
    emit(f"(make-history {name}-h)")
    emit(f"(def {name} (gswitch dclk {val} (read-history {name}-h)))")
    emit(f"(write-history {name}-h {name})")

def ramp_s(name, tgt):
    if "noramp" in EXP: emit(f"(def {name} {tgt})"); return
    emit(f"(make-history {name}-a)")
    emit(f"(make-history {name}-r)")
    emit(f"(def {name}-s (gswitch hit {tgt} (gswitch hop-wrap (read-history {name}-r) (read-history {name}-a))))")
    emit(f"(write-history {name}-a {name}-s)")
    emit(f"(def {name} (gswitch hit {tgt} (+ {name}-s (* (- {tgt} {name}-s) hop-frac))))")
    emit(f"(write-history {name}-r {name})")

def mlp(p, zs, vel, w3, b3, n3):
    emit(f"(def {p}-x (+ (* {zs[0]} onehot0) (* {zs[1]} onehot1) (* {zs[2]} onehot2) (* {zs[3]} onehot3) (* {vel} onehot4)))")
    emit(f"(def {p}-h1 (tanh (+ (matmul {p}-x l1-W) l1-b)))")
    emit(f"(def {p}-h2 (tanh (+ (matmul {p}-h1 l2-W) l2-b)))")
    emit(f"(def {p}-W3 (tensor @shape [128 {n3}] @file \"{w3}\"))")
    emit(f"(def {p}-b3 (tensor @shape [1 {n3}] @file \"{b3}\"))")
    emit(f"(def {p}-u (+ (matmul {p}-h2 {p}-W3) {p}-b3))")

emit(f""";; Latent Snare Morph (E8) -- GENERATED by scripts/e8/gen_morph.py (hop {HOP}), do not edit.
;;
;; The E7 Latent Snare (5->128->128->360 tanh MLP -> 32-mode modal head, 16 noise bands,
;; 2-strand snare wires, drive + soft clip) with INCREMENTAL voice state, so the decoded
;; parameters can change while a hit rings. While a hit rings the latent moves from A to B:
;;   z(t) = zA + w (zB - zA),  w = amount * p^(4^curve),  p = clip(age / morph, 0, 1)
;; zA = live (mod z1..z4), zB = live (mod b1..b4), sampled every {HOP} samples.
;; Two event clocks over the same weights:
;;   hit  : layers 1-2 + the STRIKE columns of layer 3 (w3s) at z = A on the onset frame:
;;          mode amplitude/phase/rise/onset delay/glide, pre-delay, noise onset/attack/levels,
;;          output gain. Latched on the hit: the strike never morphs.
;;   dclk : the hit, or a hop tick while the morph moves or a watched knob / raw mod input
;;          changed: layers 1-2 + the RING columns (w3r) at z(t): mode frequencies, mode and
;;          band decays, band coupling, gates, drive/clip, length fade, wire knobs.
;;          Frequencies, coupling and ring scalars ramp linearly across the hop; decay rates
;;          step per hop (they only bend the slope of a continuous envelope).
;; Mode state per sample (t = time since the mode's onset):
;;   P += f (1/sr + G),  G -= G eg      (glide: G = glide gtau eg e^(-t/gtau))
;;   D -= D ed,  R -= R er               (D = amp e^(-t/dec), R = e^(-t/rise); e = 1 - e^(-1/(x sr)))
;;   env = nyq gates D (1 - R),  head = sum env sin(2 pi P)
;; States are seeded on the hit frame at t = tau = (onset sample - 1) - onset time, in
;; (-1/sr, 0], so the first active sample lands on the closed form's fractional onset. With
;; A = B (or amount 0) and fixed knobs the output equals the closed-form E7 voice to float32
;; rounding.

(def gate (in 1 @name gate))
(def pitch (in 2 @name pitch))
(def velocity (in 3 @name velocity))
(def trigger (in 4 @name trigger))
(def clock (in 5 @name clock))
(def mod1 (in 6 @name mod1 @modulator 1))
(def mod2 (in 7 @name mod2 @modulator 2))
(def mod3 (in 8 @name mod3 @modulator 3))
(def mod4 (in 9 @name mod4 @modulator 4))
(param z1 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param z2 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param z3 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param z4 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param tune @default 0 @min -24 @max 24 @unit st)
(param decay @default 1 @min 0.25 @max 2)
(param level @default 0.5 @min 0 @max 2)
(param b1 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param b2 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param b3 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param b4 @default 0 @min -3 @max 3 @mod true @mod-mode additive)
(param morph @default 400 @min 10 @max 4000 @unit ms)
(param curve @default 0 @min -1 @max 1)
(param amount @default 1 @min 0 @max 1)
(param follow @default 1 @min 0 @max 1)

(defmacro onset-of (trig)
  (make-history previous)
  (def high (gt trig 0.5))
  (def rising (* high (lte (read-history previous) 0.5)))
  (write-history previous high)
  rising)

;; per-sample decrement e = 1 - e^(-a); series below a = 0.05 (rel error < 1e-9)
(defmacro dec-eps (aa)
  (gswitch (gt aa 0.05) (- 1 (exp (* -1 aa)))
    (* aa (- 1 (* aa (- 0.5 (* aa (- 0.16666666666666666 (* aa (- 0.041666666666666664 (* aa 0.008333333333333333)))))))))))

{svf_macros}
(def hit (onset-of trigger))
;; samples since the last onset (0 on the onset frame), capped at 30 s; age in seconds
(make-history elapsed)
(make-history armed)
(def played (max hit (read-history armed)))
(write-history armed played)
(def samples (gswitch hit 0 (min (* 30 samplerate) (+ 1 (read-history elapsed)))))
(write-history elapsed samples)
(def alive (* played (lt samples (* 30 samplerate))))
(def age (* alive (/ samples samplerate)))

;; hop counter restarts on the hit; hop-wrap marks every hop boundary (ramps restart there)
(make-history hopc)
(def hop-n (gswitch hit 0 (% (+ (read-history hopc) 1) {HOP})))
(write-history hopc hop-n)
(def hop-wrap (eq hop-n 0))
(def hop-frac (/ (+ hop-n 1) {HOP}))""")

if a.always:
    emit("(def dclk (max hit (* hop-wrap alive)))")
else:
    emit(""";; A tick decodes only while the morph moves (A and B can differ) or a watched control has
;; changed since the last decode. Clocks must not read (mod p): knobs and raw modN inputs only.
(def w-sig (+ z1 (* 1.3127 z2) (* 1.7351 z3) (* 2.2193 z4) (* 2.9137 b1) (* 3.3719 b2) (* 4.1413 b3) (* 4.7713 b4)
              (* 5.3171 mod1) (* 6.1379 mod2) (* 6.7231 mod3) (* 7.3991 mod4)
              (* 0.0123 tune) (* 0.311 decay) (* 0.000917 morph) (* 0.7193 curve) (* 0.8317 amount) (* 0.9173 follow)))
(make-history w-last)
(def w-changed (eq (eq w-sig (read-history w-last)) 0))
(def ab-differ (gt (+ (abs (- z1 b1)) (abs (- z2 b2)) (abs (- z3 b3)) (abs (- z4 b4))
                      (abs mod1) (abs mod2) (abs mod3) (abs mod4)) 0))""")
    emit(f"(def morphing (* (gt amount 0) ab-differ (lt age (+ (* morph 0.001) (/ {2 * HOP} samplerate)))))")
    emit("(def dclk (max hit (* hop-wrap alive (lt age 2.1) (max w-changed morphing))))")
    emit("(write-history w-last (gswitch dclk w-sig (read-history w-last)))")

emit(f"""
(def l1-W (tensor @shape [5 128] @file "w1.json"))
(def l1-b (tensor @shape [1 128] @file "b1.json"))
(def l2-W (tensor @shape [128 128] @file "w2.json"))
(def l2-b (tensor @shape [1 128] @file "b2.json"))
(def onehot0 (tensor @shape [1 5] @data [1 0 0 0 0]))
(def onehot1 (tensor @shape [1 5] @data [0 1 0 0 0]))
(def onehot2 (tensor @shape [1 5] @data [0 0 1 0 0]))
(def onehot3 (tensor @shape [1 5] @data [0 0 0 1 0]))
(def onehot4 (tensor @shape [1 5] @data [0 0 0 0 1]))
(def isr (/ 1 samplerate))

;; ==== STRIKE decoder (hit clock, z = A) =============================================
(def h-a1 (event-hold (clip (mod z1) -3 3) hit))
(def h-a2 (event-hold (clip (mod z2) -3 3) hit))
(def h-a3 (event-hold (clip (mod z3) -3 3) hit))
(def h-a4 (event-hold (clip (mod z4) -3 3) hit))
(def h-vel (event-hold (clip velocity 0.4 1) hit))
(def h-tune (event-hold tune hit))
(def h-decay (event-hold decay hit))
(def h-note (event-hold (clip (/ pitch 261.6255653) 0.25 4) hit))""")
mlp("h", ["h-a1", "h-a2", "h-a3", "h-a4"], "h-vel", "w3s.json", "b3s.json", NS)
emit("\n".join(s_lines))
emit(f"""(def h-ratio (* h-note (exp (* h-tune 0.057762265046662105))))
(def h-f (* hraw-mode_freq h-ratio))
(def h-sh (* h-f hraw-mode_glide))
(def h-idec (/ 1 (* hraw-mode_decay h-decay)))
(def h-irise (/ 1 hraw-mode_rise))
(def h-itau (/ 1 hraw-mode_gtau))
(def h-eg (dec-eps (* h-itau isr)))
;; onset sample of each mode and its seed time tau in (-1/sr, 0]
(def h-on (* (+ hraw-pre hraw-mode_t0) samplerate))
(def h-n0 (+ (floor h-on) 1))
(def h-tau (* (- (- h-n0 1) h-on) isr))
(def h-gx (exp (* -1 h-tau h-itau)))
(def h-bid1 (/ 1 (* hraw-band_d1 h-decay)))
(def h-bid2 (/ 1 (* hraw-band_d2 h-decay)))
(def h-non (* hraw-noise_onset samplerate))
(def h-nn0 (+ (floor h-non) 1))
(def h-ntau (* (- (- h-nn0 1) h-non) isr))
;; strike values: sampled from the hit domain on the hit frame into audio-rate holds
;; (a gswitch on hit over a history; between hits the event value is never read)
""")
for nm, v, n in (("m-er", "(dec-eps (* h-irise isr))", K), ("m-eg", "h-eg", K), ("m-n0", "h-n0", K), ("m-gpos", "(max hraw-mode_glide 0)", K)):
    sh_t(nm, v, n)
for nm, v in (("g-nn0", "h-nn0"), ("g-nirise", "(/ 1 hraw-noise_rise)"), ("g-nonset", "hraw-noise_onset"), ("g-pre", "hraw-pre"), ("g-gain", "hraw-gain"), ("g-lgainA", "(log hraw-gain)")):
    sh_s(nm, v)
for nm, v, n in (("m-lamA", "(log hraw-mode_amp)", K), ("b-lfA", "(log (max (* hraw-band_amp (- 1 hraw-band_tail)) 1e-30))", NB), ("b-lsA", "(log (max (* hraw-band_amp hraw-band_tail) 1e-30))", NB)):
    sh_t(nm, v, n)
emit(f""";; state seeds: read on the hit frame only (by the gswitch on hit), so no hold
(def s-D0 (* hraw-mode_amp (exp (* -1 h-tau h-idec))))
(def s-R0 (exp (* -1 h-tau h-irise)))
(def s-G0 (* hraw-mode_glide hraw-mode_gtau h-eg h-gx))
(def s-P0 (+ (* hraw-mode_phase 0.15915494309189535) (* h-f h-tau) (* h-sh hraw-mode_gtau (- 1 h-gx))))
(def s-F0 (* hraw-band_amp (- 1 hraw-band_tail) (exp (* -1 h-ntau h-bid1))))
(def s-S0 (* hraw-band_amp hraw-band_tail (exp (* -1 h-ntau h-bid2))))
;; ==== RING decoder (dclk, z = z(t)) =================================================
(def e-hit (event-hold hit dclk))
(def e-age (event-hold age dclk))
(def e-a1 (event-hold (clip (mod z1) -3 3) dclk))
(def e-a2 (event-hold (clip (mod z2) -3 3) dclk))
(def e-a3 (event-hold (clip (mod z3) -3 3) dclk))
(def e-a4 (event-hold (clip (mod z4) -3 3) dclk))
(def e-b1 (event-hold (clip (mod b1) -3 3) dclk))
(def e-b2 (event-hold (clip (mod b2) -3 3) dclk))
(def e-b3 (event-hold (clip (mod b3) -3 3) dclk))
(def e-b4 (event-hold (clip (mod b4) -3 3) dclk))
(def e-mt (event-hold (clip morph 1 10000) dclk))
(def e-curve (event-hold (clip curve -1 1) dclk))
(def e-amt (event-hold (clip amount 0 1) dclk))
(def e-vel (event-hold (clip velocity 0.4 1) dclk))
(def e-tune (event-hold tune dclk))
(def e-decay (event-hold decay dclk))
(def e-note (event-hold (clip (/ pitch 261.6255653) 0.25 4) dclk))
;; morph position one hop ahead (the ramp then arrives on time); exactly 0 on the hit
(def e-p (* (- 1 e-hit) (clip (/ (+ e-age (/ {HOP} samplerate)) (* e-mt 0.001)) 0 1)))
(def e-w (* e-amt (gt e-p 0) (pow (max e-p 1e-9) (exp (* e-curve 1.3862943611198906)))))
(def e-z1 (+ e-a1 (* e-w (- e-b1 e-a1))))
(def e-z2 (+ e-a2 (* e-w (- e-b2 e-a2))))
(def e-z3 (+ e-a3 (* e-w (- e-b3 e-a3))))
(def e-z4 (+ e-a4 (* e-w (- e-b4 e-a4))))""")
mlp("e", ["e-z1", "e-z2", "e-z3", "e-z4"], "e-vel", "w3r.json", "b3r.json", NR)
emit("\n".join(r_lines))
emit("(def e-ratio (* e-note (exp (* e-tune 0.057762265046662105))))")
emit(";; ring values cross to audio rate through dclk holds (gswitch on dclk over a history)")
hold_t("L-f", "(* eraw-mode_freq e-ratio)", K)
hold_t("L-ed", "(dec-eps (* (/ 1 (* eraw-mode_decay e-decay)) isr))", K)
hold_t("L-be1", "(dec-eps (* (/ 1 (* eraw-band_d1 e-decay)) isr))", NB)
hold_t("L-be2", "(dec-eps (* (/ 1 (* eraw-band_d2 e-decay)) isr))", NB)
hold_t("L-couple", "eraw-band_couple", NB)
emit(""";; level follow: each mode / band / the output gain may only FALL from its strike level toward
;; the level decoded at z(t) (follow 0 = strike levels kept, 1 = down to z(t)'s); a morph never
;; adds energy, so modes that are quiet in B fade instead of ringing on with B's decay
(def e-follow (event-hold (clip follow 0 1) dclk))
(def e-lamA (event-hold m-lamA dclk))
(def e-lfA (event-hold b-lfA dclk))
(def e-lsA (event-hold b-lsA dclk))
(def e-lgainA (event-hold g-lgainA dclk))""")
hold_t("L-mc", "(exp (* e-follow (min (- (log eraw-mode_amp) e-lamA) 0)))", K)
hold_t("L-bfc", "(exp (* e-follow (min (- (log (max (* eraw-band_amp (- 1 eraw-band_tail)) 1e-30)) e-lfA) 0)))", NB)
hold_t("L-bsc", "(exp (* e-follow (min (- (log (max (* eraw-band_amp eraw-band_tail) 1e-30)) e-lsA) 0)))", NB)
hold_s("L-gc", "(exp (* e-follow (min (- (log eraw-gain) e-lgainA) 0)))")
hold_s("L-dv", "eraw-drive")
hold_s("L-damt", "eraw-drive_amt")
hold_s("L-dnorm", "eraw-drive_norm")
hold_s("L-fstart", "(- eraw-len eraw-fade)")
hold_s("L-ifade", "(/ 1 eraw-fade)")
hold_s("L-fcurve", "eraw-fade_curve")
hold_s("L-ghd", "eraw-gh_depth")
hold_s("L-ght", "eraw-gh_time")
hold_s("L-ghif", "(/ 1 eraw-gh_fall)")
hold_s("L-gdd", "eraw-gd_depth")
hold_s("L-gdt", "eraw-gd_time")
hold_s("L-gdif", "(/ 1 eraw-gd_fall)")
hold_s("L-gnoise", "eraw-gate_noise")
hold_s("L-cceil", "eraw-clip_ceil")
hold_s("L-cinv", "(/ 1 eraw-clip_ceil)")
hold_s("L-cmix", "eraw-clip_mix")
hold_s("L-wgv", "(* 0.06 (* eraw-wire_thr 1.2))")
hold_s("L-wlv", "(* 60 eraw-wire_level)")
hold_s("L-wglo", "(wire-svf-g eraw-wire_lo)")
hold_s("L-wghi", "(wire-svf-g eraw-wire_hi)")
hold_s("L-wk", "eraw-wire_k")
emit(";; ring values ramped across each hop")
ramp_t("m-f", "L-f", K)
ramp_t("m-c", "L-mc", K)
ramp_t("b-fc", "L-bfc", NB)
ramp_t("b-sc", "L-bsc", NB)
ramp_t("b-couple", "L-couple", NB)
for nm in ["dv", "damt", "dnorm", "fstart", "ifade", "fcurve", "ghd", "ght", "ghif", "gdd", "gdt", "gdif",
           "gnoise", "cceil", "cinv", "cmix", "wgv", "wlv", "wglo", "wghi", "wk", "gc"]:
    ramp_s(f"g-{nm}", f"L-{nm}")

emit(f"""
;; ---- modal bank (incremental) ----------------------------------------------------
(def head-T (max 0 (- age g-pre)))
(def gh-x (clip (* (- head-T g-ght) g-ghif) 0 1))
(def gate-h (- 1 (* g-ghd gh-x gh-x (- 3 (* 2 gh-x)))))
(def gd-x (clip (* (- head-T g-gdt) g-gdif) 0 1))
(def gate-d (- 1 (* g-gdd gd-x gd-x (- 3 (* 2 gd-x)))))
;; nyquist guard from the ramped frequency and the hit's upward glide
(def m-nyq (clip (/ (- (* 0.48 samplerate) (+ m-f (* m-f m-gpos))) (* 0.08 samplerate)) 0 1))
(def m-act (gte samples m-n0))
(make-tensor-history mP @shape [1 {K}])
(make-tensor-history mG @shape [1 {K}])
(make-tensor-history mD @shape [1 {K}])
(make-tensor-history mR @shape [1 {K}])
(def mP-1 (read-tensor-history mP))
(def mG-1 (read-tensor-history mG))
(def mD-1 (read-tensor-history mD))
(def mR-1 (read-tensor-history mR))
(def mP-adv (+ mP-1 (* m-f (+ isr mG-1))))
(def m-P (gswitch hit s-P0 (gswitch m-act (- mP-adv (floor mP-adv)) mP-1)))
(def m-G (gswitch hit s-G0 (gswitch m-act (- mG-1 (* mG-1 m-eg)) mG-1)))
(def m-D (gswitch hit s-D0 (gswitch m-act (- mD-1 (* mD-1 L-ed)) mD-1)))
(def m-R (gswitch hit s-R0 (gswitch m-act (- mR-1 (* mR-1 m-er)) mR-1)))
(write-tensor-history mP m-P)
(write-tensor-history mG m-G)
(write-tensor-history mD m-D)
(write-tensor-history mR m-R)
(def mode-env (* m-act m-nyq m-c (* gate-h gate-d) m-D (- 1 m-R)))
(def head (sum (* mode-env (sin (* twopi m-P)))))
(def head-e (sqrt (+ (sum (* mode-env mode-env)) 1e-10)))
""")
emit(noise_block)
emit(f""";; band env = att(t) (F + S) + C E(t); F, S decaying states seeded like the modes
(def noise-t (max 0 (- age g-nonset)))
(def noise-att (- 1 (exp (* -1 noise-t g-nirise))))
(def n-act (gte samples g-nn0))
(make-tensor-history nF @shape [1 {NB}])
(make-tensor-history nS @shape [1 {NB}])
(def nF-1 (read-tensor-history nF))
(def nS-1 (read-tensor-history nS))
(def n-F (gswitch hit s-F0 (gswitch n-act (- nF-1 (* nF-1 L-be1)) nF-1)))
(def n-S (gswitch hit s-S0 (gswitch n-act (- nS-1 (* nS-1 L-be2)) nS-1)))
(write-tensor-history nF n-F)
(write-tensor-history nS n-S)
(def band-env (+ (* noise-att (+ (* b-fc n-F) (* b-sc n-S))) (* b-couple head-e)))
(def noise-out (* (- 1 (* g-gnoise (- 1 gate-d))) (sum (* band-env bands))))

;; ---- snare wires (E5 W2; drive 1.2, 620 Hz, t60 1 ms fixed; gap/level/band live) --------
(def w-det (tensor @shape [1 2] @data [0.951 1.062]))
(def w-r1 (exp (/ -6.9077553 (* samplerate 0.001 1.0))))
(def w-r2 (exp (/ -6.9077553 (* samplerate 0.0004 1.0))))
(def w-wc1 (* twopi (/ (clip 620.0 20 (* 0.2 samplerate)) samplerate)))
(def w-wc2 (* twopi (/ (clip (* 2.41 620.0) 20 (* 0.3 samplerate)) samplerate)))
(def w-c1 (* 2 w-r1 (cos (* w-wc1 w-det))))
(def w-c2 (* 2 w-r2 (cos (* w-wc2 w-det))))
(def w-q1 (* w-r1 w-r1))
(def w-q2 (* w-r2 w-r2))
(def w-z (* (* 0.06 1.2) head))
(make-tensor-history wy1 @shape [1 2])
(make-tensor-history wy2 @shape [1 2])
(make-tensor-history wu1 @shape [1 2])
(make-tensor-history wu2 @shape [1 2])
(def w-y1 (read-tensor-history wy1))
(def w-y2 (read-tensor-history wy2))
(def w-u1 (read-tensor-history wu1))
(def w-u2 (read-tensor-history wu2))
(def w-m1 (- (* w-c1 w-y1) (* w-q1 w-y2)))
(def w-m2 (- (* w-c2 w-u1) (* w-q2 w-u2)))
(def w-ov (min (max (- w-z w-m1 (* 0.5 w-m2) g-wgv) 0) 0.02))
(def w-push (* w-ov (+ 1 (* 0.8 (min (* w-ov 1000) 1)))))
(write-tensor-history wy2 w-y1)
(write-tensor-history wy1 (+ w-m1 (* w-push 0.769)))
(write-tensor-history wu2 w-u1)
(write-tensor-history wu1 (+ w-m2 (* w-push 0.4615)))
(def w-app (sum w-ov))
(make-history wcm1)
(make-history wcm2)
(def w-cp1 (read-history wcm1))
(def w-cl1 (+ w-cp1 (* 0.1 (- w-app w-cp1))))
(write-history wcm1 w-cl1)
(def w-ca (- w-app w-cl1))
(def w-cp2 (read-history wcm2))
(def w-cl2 (+ w-cp2 (* 0.1 (- w-ca w-cp2))))
(write-history wcm2 w-cl2)
(def wire-raw (* g-wlv (- w-ca w-cl2)))
(def wire-out (wire-svf-lp (wire-svf-hp wire-raw g-wglo 1.414213562) g-wghi g-wk))

;; ---- drive, clip, length fade, declick, output ---------------------------------
(def dry (+ head noise-out wire-out))
(def driven0 (+ dry (* g-damt (- (* (tanh (* dry g-dv)) g-dnorm) dry))))
(def clip-u (* driven0 g-cinv))
(def clip-u2 (* clip-u clip-u))
(def driven (+ driven0 (* g-cmix (- (* g-cceil (/ clip-u (sqrt (sqrt (+ 1 (* clip-u2 clip-u2)))))) driven0))))
(def fade-x (clip (* (- age g-fstart) g-ifade) 0 1))
(def fade-1 (- 1 (* fade-x fade-x (- 3 (* 2 fade-x)))))
(def fade-2 (* fade-1 fade-1))
(def fade (+ (* fade-1 (- 1 g-fcurve)) (* fade-2 fade-2 g-fcurve)))
(def shaped (* driven fade g-gain g-gc))
;; On a retrigger the previous hit is cut; carry its last output sample and let it fall
;; over ~1 ms so the cut never steps.
(make-history last-out)
(make-history carry-h)
(def carry (gswitch hit (read-history last-out) (* (read-history carry-h) (exp (/ -1000 samplerate)))))
(write-history carry-h carry)
;; Velocity: the decoder was trained at velocities 0.4/0.7/1.0 only, so its input is clamped to
;; [0.4, 1] (timbre as trained) and a plain gain v/0.4 carries the level below 0.4 (VILLAIN's slope there).
(def vel-gain (event-hold (min 1 (/ (clip velocity 0 1) 0.4)) hit))
(def voice-out (* (+ shaped carry) level vel-gain))
(write-history last-out (+ shaped carry))
(out voice-out 1 @name audio)
(use-defmacro voice-amp)
(out (voice-amp 0 voice-out voice-out) 2 @name amp @amp true)""")
if a.headtap:
    emit("(out head 3 @name head_tap)")
    emit("(out wire-out 4 @name wire_tap)")

os.makedirs(a.out, exist_ok=True)
for f in ("w1.json", "b1.json", "w2.json", "b2.json"):
    shutil.copy(os.path.join(SRC, f), a.out)
W3, b3 = load("w3.json"), load("b3.json")
for f, arr in (("w3s.json", W3[:, s_cols]), ("b3s.json", b3[:, s_cols]), ("w3r.json", W3[:, r_cols]), ("b3r.json", b3[:, r_cols])):
    # same decimal text as the installed files (values are copied, not re-rounded)
    json.dump({"shape": list(arr.shape), "data": [float(x) for x in arr.reshape(-1)]}, open(os.path.join(a.out, f), "w"))
for f in ("w3.json", "b3.json"):
    if os.path.exists(os.path.join(a.out, f)): os.remove(os.path.join(a.out, f))
open(os.path.join(a.out, "dsp.lisp"), "w").write("\n".join(L) + "\n")
ij = os.path.join(a.out, "instrument.json")
if not os.path.exists(ij):
    open(ij, "w").write('{"version":1,"run_mode":"instrument"}\n')
print("wrote", os.path.join(a.out, "dsp.lisp"), f"hop {HOP}, strike cols {NS}, ring cols {NR}")
