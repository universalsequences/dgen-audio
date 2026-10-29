#!/usr/bin/env python3
"""Generate the latent-modal-snare realtime instrument (dsp.lisp), its
decoder_layout.json contract and, optionally, random placeholder weights.

    python3 gen_dsp.py [--out DIR]                 # dsp.lisp only (keeps weights)
    python3 gen_dsp.py --random-weights [--seed 7]

The voice model is the E1 ModalSnareAD model (dgen-modal-ad
Examples/ModalSnareAD: HeadRenderer + band noise + drive + length fade)
minus the VILLAIN-specific head/drum gates, with every per-snare constant
replaced by a decoder output. reference_render.py is the same model in
numpy and is checked against the compiled dylib.

Env switches (research variants, not for shipping):
    LATENT_MLP=matmul   natural matmul decoder (needs dgen 69f392b: matmul over frame-varying tensors)
    LATENT_MODZ=0       plain (non-@mod) z knobs (3x slower schedule)
    LATENT_PROBE=1      extra outputs tapping each decoded segment (verify_decoder.py)
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e3proto")

# E2 layout (2026-10-28): K=32 = VILLAIN's own head slots (blends interpolate slot-wise;
# any fixed 24-slot window drops up to 26-70% of some fits' head energy), H=128, two
# head gates. The E3 prototype layout (K=24, H=64, no gates) is the checked-in
# scripts/e3proto/decoder_layout.json. The trainer (dgen Examples/ModalSnareAD latent-*) reads
# the layout JSON this script writes, so the two cannot drift.
K = 32      # modes
B = 16      # noise bands
H = 128     # hidden width (two tanh layers)
LATENT_D = int(os.environ.get("LATENT_D", "4"))  # latent knobs z1..zD
D_IN = LATENT_D + 1  # z1..zD, velocity

# Output segments, in order. Each is (name, width, squash, lo, hi).
#   log : exp(ln lo + sigmoid(u) * (ln hi - ln lo))
#   lin : lo + sigmoid(u) * (hi - lo)
#   pi  : pi * tanh(u)                 (lo/hi unused)
# Ranges cover the 32 VILLAIN fits (data/villain_modes.json, live slots p0..p100
# except outliers) and E1's ModalSnareAD Ranges.
MODE_SEGS = [
    ("mode_freq",  K, "log", 4.0, 24000.0),    # f_k, Hz at tune 0 (heard pitch)
    ("mode_amp",   K, "log", 1e-6, 10.0),      # amp_k
    ("mode_decay", K, "log", 0.002, 12.0),     # 1/idec_k, s
    ("mode_rise",  K, "log", 1e-5, 0.3),       # 1/irise_k, s
    ("mode_t0",    K, "log", 1e-6, 0.6),       # t0_k, s, per-mode onset delay
    ("mode_phase", K, "pi",  0.0, 0.0),        # ph_k, rad
    ("mode_glide", K, "lin", -0.9, 2.8),       # g_k: sh_k = g_k f_k (Hz above f at onset)
    ("mode_gtau",  K, "log", 3e-4, 0.3),       # tau_k, s, glide time constant
]
NOISE_SEGS = [
    ("band_amp",    B, "log", 1e-5, 4.0),      # A_b
    ("band_d1",     B, "log", 0.002, 0.4),     # fast decay, s
    ("band_tail",   B, "lin", 0.0, 1.0),       # L_b: slow-stage share
    ("band_d2",     B, "log", 0.02, 2.0),      # slow decay, s
    ("band_couple", B, "log", 1e-5, 20.0),     # C_b: head-envelope (wire) drive
]
GLOBAL_SEGS = [
    ("noise_rise",  1, "log", 5e-5, 0.05),     # s, shared noise attack
    ("noise_onset", 1, "lin", 0.0, 0.05),      # s, shared noise onset delay
    ("pre",         1, "lin", 0.0, 0.035),     # s, head predelay
    ("drive",       1, "log", 1.0, 6.0),       # dv
    ("drive_amt",   1, "lin", 0.0, 1.0),       # dry/driven mix
    ("drive_norm",  1, "log", 0.1, 2.0),       # driven-branch gain
    ("gain",        1, "log", 0.05, 8.0),      # output gain G
    ("len",         1, "log", 0.08, 2.0),      # s, hit length (fade ends here)
    ("fade",        1, "log", 0.005, 0.08),    # s, fade length
    ("fade_curve",  1, "lin", 0.0, 1.0),       # 0 smoothstep .. 1 smoothstep^4
    # head gates (VILLAIN GH / GD without sag, all-mode masks): G = 1 - depth smooth01((T - time)/fall)
    ("gh_depth",    1, "lin", 0.0, 1.0),
    ("gh_time",     1, "log", 0.03, 2.0),      # s after the predelay
    ("gh_fall",     1, "log", 0.001, 0.2),     # s
    ("gd_depth",    1, "lin", 0.0, 1.0),
    ("gd_time",     1, "log", 0.03, 2.0),
    ("gd_fall",     1, "log", 0.001, 0.2),
    ("gate_noise",  1, "lin", 0.0, 1.0),       # share of the noise layer GD also gates
    # output soft clipper on the driven signal (VILLAIN clipper, k=4 shape):
    # x += clip_mix (ceil u / (1 + u^4)^(1/4) - x), u = x / ceil
    ("clip_ceil",   1, "log", 0.05, 4.0),
    ("clip_mix",    1, "lin", 0.0, 1.0),
]
# E5 physical wire layer knobs (scripts/e5/wire_splice.py WIRE_SEGS, copied so this script stays
# standalone). Decoded now so the W2 layer can be distilled later without changing P; the
# wire DSP itself (e5 wire_block) is not spliced in yet, so these outputs are unused here.
WIRE_SEGS = [
    ("wire_thr",   1, "log", 0.004, 0.3),     # contact threshold in head units (= gap/drive)
    ("wire_level", 1, "log", 1e-3, 40.0),     # strand output gain (VILLAIN c_rattle); 1e-3 = off
    ("wire_lo",    1, "log", 150.0, 5000.0),  # wire band-pass SVF high-pass, Hz
    ("wire_hi",    1, "log", 1200.0, 16000.0),# wire band-pass SVF low-pass, Hz
    ("wire_k",     1, "log", 0.12, 2.0),      # LP damping (1.414 = Butterworth, small = resonant)
]
SEGS = MODE_SEGS + NOISE_SEGS + GLOBAL_SEGS + WIRE_SEGS
P = sum(s[1] for s in SEGS)

# Noise filterbank: NOISE_BANDS.md (= e1/noise_bands/spec.json).
BAND_LO, BAND_HI, BAND_Q, BAND_STAGES = 100.0, 18000.0, 1.8, 2
BAND_FC = [BAND_LO * (BAND_HI / BAND_LO) ** (i / (B - 1)) for i in range(B)]
NORM_RATES = [44100.0, 48000.0, 88200.0, 96000.0]

WEIGHT_FILES = {  # tensor name -> (file, shape)
    "l1.W": ("w1.json", [D_IN, H]), "l1.b": ("b1.json", [1, H]),
    "l2.W": ("w2.json", [H, H]),    "l2.b": ("b2.json", [1, H]),
    "l3.W": ("w3.json", [H, P]),    "l3.b": ("b3.json", [1, P]),
}

MLP_FORM = os.environ.get("LATENT_MLP", "matmul")  # rows: workaround for compilers < the 69f392b fix
MOD_Z = os.environ.get("LATENT_MODZ", "1") == "1"
PROBE = bool(os.environ.get("LATENT_PROBE"))
HEADTAP = bool(os.environ.get("LATENT_HEADTAP"))  # research: out 3 = gated head (pre drive)


def band_coeffs(fc, sr, q=BAND_Q):
    w0 = 2 * math.pi * fc / sr
    al = math.sin(w0) / (2 * q)
    return al / (1 + al), -2 * math.cos(w0) / (1 + al), (1 - al) / (1 + al)


def band_norm(fc, sr, n=1 << 16):
    """1/sqrt(sum h^2) of the BAND_STAGES cascade (exact impulse response)."""
    b0, a1, a2 = band_coeffs(fc, sr)
    h = [1.0] + [0.0] * (n - 1)
    for _ in range(BAND_STAGES):
        y = [0.0] * n; x1 = x2 = y1 = y2 = 0.0
        for i, x in enumerate(h):
            v = b0 * (x - x2) - a1 * y1 - a2 * y2
            y[i] = v; x2, x1, y2, y1 = x1, x, y1, v
        h = y
    return 1.0 / math.sqrt(sum(v * v for v in h))


def offsets():
    o, out = 0, {}
    for name, w, sq, lo, hi in SEGS:
        out[name] = (o, w, sq, lo, hi)
        o += w
    return out


def fmt(x):
    return repr(float(x))


def squash_expr(u, sq, lo, hi):
    if sq == "log":
        a, b = math.log(lo), math.log(hi) - math.log(lo)
        return f"(exp (+ {fmt(a)} (* {fmt(b)} (sigmoid {u}))))"
    if sq == "lin":
        return f"(+ {fmt(lo)} (* {fmt(hi - lo)} (sigmoid {u})))"
    if sq == "pi":
        return f"(* {fmt(math.pi)} (tanh {u}))"
    raise ValueError(sq)


def tensor_lit(vals):
    return f"(tensor @shape [1 {len(vals)}] @data [{' '.join(f'{v:.9g}' for v in vals)}])"


def gen_dsp(norms):
    off = offsets()
    w_ = []
    w = w_.append
    if MOD_Z:
        mod_inputs = "\n".join(f"(def mod{i} (in {5 + i} @name mod{i} @modulator {i}))" for i in range(1, LATENT_D + 1))
        z_params = "\n".join(f"(param z{i} @default 0 @min -3 @max 3 @mod true @mod-mode additive)" for i in range(1, LATENT_D + 1))
        # the hit clock never reads (mod z); it only samples it (bd memory dgen-modulation-scheduling)
        z_holds = "\n".join(f"(def e-z{i} (event-hold (clip (mod z{i}) -3 3) hit))" for i in range(1, LATENT_D + 1))
    else:
        mod_inputs = ""
        z_params = "\n".join(f"(param z{i} @default 0 @min -3 @max 3)" for i in range(1, LATENT_D + 1))
        z_holds = "\n".join(f"(def e-z{i} (event-hold z{i} hit))" for i in range(1, LATENT_D + 1))
    w(f""";; Latent Modal Snare (E3 realtime prototype) -- GENERATED by gen_dsp.py, do not edit.
;;
;; A {D_IN}->{H}->{H}->{P} tanh MLP decoder maps (z1..z{LATENT_D}, velocity) to the physical
;; parameters of a {K}-mode closed-form modal bank, a {B}-band filtered-noise layer
;; and the output stage. The decoder runs ONLY on the onset frame of each hit:
;; its inputs are sampled with (event-hold x hit), so the matmuls and squashes
;; are scheduled on the hit event clock, and results come back to audio rate
;; through a final (latch .. hit) (tensors) or (event-latch .. hit) (scalars).
;; Per sample: the closed-form bank (3 exp + 1 sin per mode, NEON), 16 x 2 RBJ
;; band-pass biquads on one white-noise source, closed-form envelopes, drive, fade.
;; The numpy twin is reference_render.py; decoder contract: decoder_layout.json.
;;
;; Weights: w1/b1/w2/b2/w3/b3.json next to this file, {{"shape":[r,c],"data":[...]}}
;; ROW-MAJOR (numpy C order), y = x[1,in] @ W[in,out] + b (see export_weights.py).

(def gate (in 1 @name gate))
(def pitch (in 2 @name pitch))
(def velocity (in 3 @name velocity))
(def trigger (in 4 @name trigger))
(def clock (in 5 @name clock))
{mod_inputs}
{z_params}
(param tune @default 0 @min -24 @max 24 @unit st)
(param decay @default 1 @min 0.25 @max 2)
(param level @default 0.5 @min 0 @max 2)

(defmacro onset-of (trig)
  (make-history previous)
  (def high (gt trig 0.5))
  (def rising (* high (lte (read-history previous) 0.5)))
  (write-history previous high)
  rising)

;; Seconds since the last onset (0 on the onset frame). Returns 0 before the
;; first hit and after a 30 s cap; at T=0 every envelope's (1 - e^(-T/rise))
;; factor is 0, so a frozen T can never leave a DC offset.
(defmacro hit-age (onset)
  (make-history elapsed)
  (make-history armed)
  (def played (max onset (read-history armed)))
  (write-history armed played)
  (def samples (gswitch onset 0 (min (* 30 samplerate) (+ 1 (read-history elapsed)))))
  (write-history elapsed samples)
  (def alive (* played (lt samples (* 30 samplerate))))
  (* alive (/ samples samplerate)))

(def hit (onset-of trigger))
(def age (hit-age hit))

;; ---- decoder (event clock: runs on onset frames only) ----------------------
{z_holds}
(def e-vel (event-hold (clip velocity 0 1) hit))
(def e-tune (event-hold tune hit))
(def e-decay (event-hold decay hit))
""")
    for name, (f, shp) in WEIGHT_FILES.items():
        w(f'(def {name.replace(".", "-")} (tensor @shape [{shp[0]} {shp[1]}] @file "{f}"))')
    xs = [f"e-z{i}" for i in range(1, LATENT_D + 1)] + ["e-vel"]

    def scalars(src, n, name):
        for k in range(n):
            w(f"(def {name}{k} (sum (shrink {src} @ranges [0:1,{k}:{k + 1}])))")
        return [f"{name}{k}" for k in range(n)]

    if MLP_FORM == "matmul":
        # Natural form. Needs a DGenLisp with dgen 69f392b (v0.1.40 miscompiles it).
        for i in range(D_IN):
            oh = " ".join("1" if j == i else "0" for j in range(D_IN))
            w(f"(def onehot{i} (tensor @shape [1 {D_IN}] @data [{oh}]))")
        w("(def dec-x (+ " + " ".join(f"(* {x} onehot{i})" for i, x in enumerate(xs)) + "))")
        w("(def dec-h1 (tanh (+ (matmul dec-x l1-W) l1-b)))")
        w("(def dec-h2 (tanh (+ (matmul dec-h1 l2-W) l2-b)))")
        w("(def dec-u (+ (matmul dec-h2 l3-W) l3-b))")
    else:
        # Workaround for pre-69f392b compilers: x @ W as a sum of scalar-scaled STATIC rows,
        # h = b + sum_k x_k W[k,:]. The varying operand is never broadcast; each
        # x_k is read out of the previous layer with a one-element shrink.
        def layer(out, inputs, W, b, n_out, act):
            for k in range(len(inputs)):
                w(f"(def {W}-r{k} (shrink {W} @ranges [{k}:{k + 1},0:{n_out}]))")
            body = f"(+ {b} " + " ".join(f"(* {x} {W}-r{k})" for k, x in enumerate(inputs)) + ")"
            w(f"(def {out} {'(tanh ' + body + ')' if act else body})")
        layer("dec-h1", xs, "l1-W", "l1-b", H, True)
        layer("dec-h2", scalars("dec-h1", H, "h1s"), "l2-W", "l2-b", H, True)
        layer("dec-u", scalars("dec-h2", H, "h2s"), "l3-W", "l3-b", P, False)
    w("")
    w(";; squash each output segment into its bounded physical range")
    for name, (o, wd, sq, lo, hi) in off.items():
        u = f"(shrink dec-u @ranges [0:1,{o}:{o + wd}])"
        if wd == 1:
            u = f"(sum {u})"
        w(f"(def raw-{name} {squash_expr(u, sq, lo, hi)})")
    nt = {sr: tensor_lit(norms[sr]) for sr in NORM_RATES}
    w(f"""
;; post-decoder performance controls (still on the event clock)
(def e-ratio (exp (* e-tune {fmt(math.log(2) / 12)})))
(def ev-f (* raw-mode_freq e-ratio))
(def ev-sh (* ev-f raw-mode_glide))
(def ev-nyq (clip (/ (- (* 0.48 samplerate) (+ ev-f (max ev-sh 0))) (* 0.08 samplerate)) 0 1))
(def ev-amp (* raw-mode_amp ev-nyq))
(def ev-idec (/ 1 (* raw-mode_decay e-decay)))
(def ev-irise (/ 1 raw-mode_rise))
(def ev-phc (* raw-mode_phase {fmt(1 / (2 * math.pi))}))
(def ev-itau (/ 1 raw-mode_gtau))
(def ev-shtau (* ev-sh raw-mode_gtau))
(def ev-bid1 (/ 1 (* raw-band_d1 e-decay)))
(def ev-bid2 (/ 1 (* raw-band_d2 e-decay)))
(def ev-bfast (* raw-band_amp (- 1 raw-band_tail)))
(def ev-bslow (* raw-band_amp raw-band_tail))

;; ---- back to audio rate ----------------------------------------------------
(def m-f (latch ev-f hit))
(def m-amp (latch ev-amp hit))
(def m-idec (latch ev-idec hit))
(def m-irise (latch ev-irise hit))
(def m-t0 (latch raw-mode_t0 hit))
(def m-phc (latch ev-phc hit))
(def m-itau (latch ev-itau hit))
(def m-shtau (latch ev-shtau hit))
(def b-id1 (latch ev-bid1 hit))
(def b-id2 (latch ev-bid2 hit))
(def b-fast (latch ev-bfast hit))
(def b-slow (latch ev-bslow hit))
(def b-couple (latch raw-band_couple hit))
(def g-nirise (event-latch (/ 1 raw-noise_rise) hit))
(def g-nonset (event-latch raw-noise_onset hit))
(def g-pre (event-latch raw-pre hit))
(def g-dv (event-latch raw-drive hit))
(def g-damt (event-latch raw-drive_amt hit))
(def g-dnorm (event-latch raw-drive_norm hit))
(def g-gain (event-latch raw-gain hit))
(def g-fstart (event-latch (- raw-len raw-fade) hit))
(def g-ifade (event-latch (/ 1 raw-fade) hit))
(def g-fcurve (event-latch raw-fade_curve hit))
(def g-ghd (event-latch raw-gh_depth hit))
(def g-ght (event-latch raw-gh_time hit))
(def g-ghif (event-latch (/ 1 raw-gh_fall) hit))
(def g-gdd (event-latch raw-gd_depth hit))
(def g-gdt (event-latch raw-gd_time hit))
(def g-gdif (event-latch (/ 1 raw-gd_fall) hit))
(def g-gnoise (event-latch raw-gate_noise hit))
(def g-cceil (event-latch raw-clip_ceil hit))
(def g-cinv (event-latch (/ 1 raw-clip_ceil) hit))
(def g-cmix (event-latch raw-clip_mix hit))

;; ---- modal bank (closed form, per sample) ----------------------------------
;; t_k = max(0, max(0, T - pre) - t0_k)
;; m_k = amp_k e^(-t idec_k) (1 - e^(-t irise_k)) sin(2 pi (f_k t + sh_k tau_k (1 - e^(-t/tau_k))) + ph_k)
(def head-T (max 0 (- age g-pre)))
;; gates G = 1 - depth * smoothstep((T - time) / fall); GH*GD scale every mode, GD also
;; scales a gate_noise share of the noise layer
(def gh-x (clip (* (- head-T g-ght) g-ghif) 0 1))
(def gate-h (- 1 (* g-ghd gh-x gh-x (- 3 (* 2 gh-x)))))
(def gd-x (clip (* (- head-T g-gdt) g-gdif) 0 1))
(def gate-d (- 1 (* g-gdd gd-x gd-x (- 3 (* 2 gd-x)))))
(def mode-t (max 0 (- head-T m-t0)))
(def mode-env (* m-amp (* gate-h gate-d) (exp (* -1 mode-t m-idec)) (- 1 (exp (* -1 mode-t m-irise)))))
(def mode-cyc (+ (* m-f mode-t) (* m-shtau (- 1 (exp (* -1 mode-t m-itau)))) m-phc))
(def head (sum (* mode-env (sin (* twopi mode-cyc)))))
(def head-e (sqrt (+ (sum (* mode-env mode-env)) 1e-10)))

;; ---- 16-band noise (NOISE_BANDS.md) ----------------------------------------
;; N_b = norm_b * BP_b(BP_b(white)), RBJ constant-0dB-peak band-pass, Q {BAND_Q}, DF1
(def band-fc {tensor_lit(BAND_FC)})
(def band-w0 (/ (* twopi band-fc) samplerate))
(def band-alpha (/ (sin band-w0) {fmt(2 * BAND_Q)}))
(def band-b0 (/ band-alpha (+ 1 band-alpha)))
(def band-a1 (/ (* -2 (cos band-w0)) (+ 1 band-alpha)))
(def band-a2 (/ (- 1 band-alpha) (+ 1 band-alpha)))
;; unit-RMS normalisation of the cascade, exact at the tabulated rates
(def band-norm (gswitch (lt samplerate 46050) {nt[44100.0]}
                 (gswitch (lt samplerate 68100) {nt[48000.0]}
                   (gswitch (lt samplerate 92100) {nt[88200.0]} {nt[96000.0]}))))
;; zero-mean, unit-variance white: dgen (noise) is xorshift32 uniform [0,1)
(def white (* 3.4641016 (- (noise) 0.5)))
(make-history nx1)
(make-history nx2)
(def white-x2 (read-history nx2))
(write-history nx2 (read-history nx1))
(write-history nx1 white)
(make-tensor-history s1y1 @shape [1 {B}])
(make-tensor-history s1y2 @shape [1 {B}])
(make-tensor-history s2y1 @shape [1 {B}])
(make-tensor-history s2y2 @shape [1 {B}])
(def s1-1 (read-tensor-history s1y1))
(def s1-2 (read-tensor-history s1y2))
(def s2-1 (read-tensor-history s2y1))
(def s2-2 (read-tensor-history s2y2))
(def stage1 (- (* band-b0 (- white white-x2)) (+ (* band-a1 s1-1) (* band-a2 s1-2))))
(def stage2 (- (* band-b0 (- stage1 s1-2)) (+ (* band-a1 s2-1) (* band-a2 s2-2))))
(write-tensor-history s1y2 s1-1)
(write-tensor-history s1y1 stage1)
(write-tensor-history s2y2 s2-1)
(write-tensor-history s2y1 stage2)
(def bands (* band-norm stage2))
;; env_b = (1 - e^(-t/rise)) (A_b ((1-L_b) e^(-t/d1_b) + L_b e^(-t/d2_b))) + C_b E(t),
;; t = max(0, T - onset), E = head amplitude-envelope RMS (wire coupling)
(def noise-t (max 0 (- age g-nonset)))
(def noise-att (- 1 (exp (* -1 noise-t g-nirise))))
(def band-env (+ (* noise-att (+ (* b-fast (exp (* -1 noise-t b-id1))) (* b-slow (exp (* -1 noise-t b-id2)))))
                 (* b-couple head-e)))
(def noise-out (* (- 1 (* g-gnoise (- 1 gate-d))) (sum (* band-env bands))))

;; ---- drive, length fade, declick, output -----------------------------------
(def dry (+ head noise-out))
(def driven0 (+ dry (* g-damt (- (* (tanh (* dry g-dv)) g-dnorm) dry))))
(def clip-u (* driven0 g-cinv))
(def clip-u2 (* clip-u clip-u))
(def driven (+ driven0 (* g-cmix (- (* g-cceil (/ clip-u (sqrt (sqrt (+ 1 (* clip-u2 clip-u2)))))) driven0))))
(def fade-x (clip (* (- age g-fstart) g-ifade) 0 1))
(def fade-1 (- 1 (* fade-x fade-x (- 3 (* 2 fade-x)))))
(def fade-2 (* fade-1 fade-1))
(def fade (+ (* fade-1 (- 1 g-fcurve)) (* fade-2 fade-2 g-fcurve)))
(def shaped (* driven fade g-gain))
;; On a retrigger the previous hit is cut; carry its last output sample and let
;; it fall over ~1 ms so the cut never steps.
(make-history last-out)
(make-history carry-h)
(def carry (gswitch hit (read-history last-out) (* (read-history carry-h) (exp (/ -1000 samplerate)))))
(write-history carry-h carry)
(def voice-out (* (+ shaped carry) level))
(write-history last-out (+ shaped carry))
(out voice-out 1 @name audio)
(use-defmacro voice-amp)
(out (voice-amp 0 voice-out voice-out) 2 @name amp @amp true)
""")
    if HEADTAP:
        w("(out head 3 @name head_tap)")
    if PROBE:
        ch = 3
        for name, (o, wd, sq, lo, hi) in off.items():
            if wd > 1:
                c = [(k + 1) / wd for k in range(wd)]
                w(f"(out (event-latch (sum (* raw-{name} {tensor_lit(c)})) hit) {ch} @name p_{name})")
            else:
                w(f"(out (event-latch raw-{name} hit) {ch} @name p_{name})")
            ch += 1
    return "\n".join(w_)


def layout_json(norms):
    off = offsets()
    return {
        "doc": "Latent Snare decoder contract; see dgen Examples/ModalSnareAD (scripts/e3proto/gen_dsp.py, scripts/e2/template.py)",
        "inputs": [f"z{i}" for i in range(1, LATENT_D + 1)] + ["velocity"], "version": "e2",
        "gates": "G = 1 - depth smooth01((T - time)/fall), T = max(0, age - pre); modes *= GH GD (E from gated env); noise *= 1 - gate_noise (1 - GD)",
        "input_ranges": {"z": [-3, 3], "velocity": [0, 1]},
        "hidden": [H, H], "activation": "tanh", "P": P, "K": K, "B": B,
        "weights": {k: {"file": f, "shape": s} for k, (f, s) in WEIGHT_FILES.items()},
        "matmul": "y[1,out] = x[1,in] @ W[in,out] + b[1,out]; W stored row-major (C order)",
        "segments": [{"name": n, "offset": off[n][0], "width": wd, "squash": sq, "lo": lo, "hi": hi}
                     for n, wd, sq, lo, hi in SEGS],
        "noise_bands": {"fc": BAND_FC, "Q": BAND_Q, "stages": BAND_STAGES,
                        "form": "RBJ BPF constant 0 dB peak, DF1, a0-normalised; N_b = norm_b * cascade(white)",
                        "norm": {str(int(sr)): norms[sr] for sr in NORM_RATES}},
    }


def random_weights(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    off = offsets()
    W1 = rng.normal(0, 1.0 / math.sqrt(D_IN), (D_IN, H)) * 1.2
    W2 = rng.normal(0, 1.0 / math.sqrt(H), (H, H)) * 1.2
    W3 = rng.normal(0, 1.0 / math.sqrt(H), (H, P)) * 0.8
    b1 = rng.normal(0, 0.1, (1, H)); b2 = rng.normal(0, 0.1, (1, H))
    b3 = np.zeros((1, P))
    logit = lambda p: math.log(p / (1 - p))

    def setv(name, vals):
        o, wd, sq, lo, hi = off[name]
        for i, v in enumerate(vals if isinstance(vals, (list, tuple)) else [vals] * wd):
            if sq == "log":
                p = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
            elif sq == "lin":
                p = (v - lo) / (hi - lo)
            else:
                p = 0.5 * (v / math.pi + 1)
            b3[0, o + i] = logit(min(max(p, 1e-3), 1 - 1e-3))
    # Placeholder biases that land a vaguely VILLAIN-like default (medians of
    # data/villain_modes.json): modes 150 Hz..4 kHz, short; noise centred ~3 kHz.
    setv("mode_freq", [150 * (4000 / 150) ** (k / (K - 1)) for k in range(K)])
    setv("mode_amp", [0.6 * 0.8 ** k for k in range(K)])
    setv("mode_decay", [0.12 * 0.9 ** k for k in range(K)])
    setv("mode_rise", 2.5e-4); setv("mode_t0", 2e-4); setv("mode_phase", 0.0)
    setv("mode_glide", 0.05); setv("mode_gtau", 0.014)
    setv("band_amp", [0.003 + 0.06 * math.exp(-((math.log2(f / 3000)) ** 2) / 3) for f in BAND_FC])
    setv("band_d1", 0.03); setv("band_tail", 0.3); setv("band_d2", 0.15); setv("band_couple", 2e-3)
    setv("noise_rise", 3e-4); setv("noise_onset", 1e-3); setv("pre", 1e-3)
    setv("drive", 1.5); setv("drive_amt", 0.3); setv("drive_norm", 0.9); setv("gain", 0.2)
    setv("len", 0.5); setv("fade", 0.02); setv("fade_curve", 0.3)
    setv("gh_depth", 0.0); setv("gh_time", 0.3); setv("gh_fall", 0.005)
    setv("gd_depth", 0.0); setv("gd_time", 0.3); setv("gd_fall", 0.005); setv("gate_noise", 0.0)
    setv("clip_ceil", 1.0); setv("clip_mix", 0.0)
    setv("wire_thr", 0.042); setv("wire_level", 2e-3); setv("wire_lo", 1300); setv("wire_hi", 7000); setv("wire_k", 1.3)
    return {"l1.W": W1, "l1.b": b1, "l2.W": W2, "l2.b": b2, "l3.W": W3, "l3.b": b3}


def write_eseq_weights(params, outdir):
    """{name: array} -> one {"shape","data"} JSON per tensor, row-major."""
    for name, (f, shp) in WEIGHT_FILES.items():
        a = params[name]
        assert list(a.shape) == shp, (name, a.shape, shp)
        with open(os.path.join(outdir, f), "w") as fh:
            json.dump({"shape": shp, "data": [round(float(x), 7) for x in a.reshape(-1)]}, fh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "Latent Snare"))
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layout", help="also write decoder_layout.json here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(HERE, ".band_norms.json")
    key = f"{BAND_LO}|{BAND_HI}|{BAND_Q}|{BAND_STAGES}|{B}"
    c = json.load(open(cache)) if os.path.exists(cache) else {}
    if c.get("key") != key:
        c = {"key": key, "norms": {str(sr): [band_norm(f, sr) for f in BAND_FC] for sr in NORM_RATES}}
        json.dump(c, open(cache, "w"))
    norms = {float(k): v for k, v in c["norms"].items()}
    open(os.path.join(a.out, "dsp.lisp"), "w").write(gen_dsp(norms))
    # The layout travels with the instrument (and to --layout, the trainer's copy).
    # scripts/e3proto/decoder_layout.json is the frozen E3 prototype contract.
    for lp in [os.path.join(a.out, "decoder_layout.json")] + ([a.layout] if a.layout else []):
        json.dump(layout_json(norms), open(lp, "w"), indent=1)
    if a.random_weights:
        write_eseq_weights(random_weights(a.seed), a.out)
    print(f"K={K} B={B} H={H} P={P} -> {a.out}")
