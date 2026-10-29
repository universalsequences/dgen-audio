#!/usr/bin/env python3
"""Generate the standalone E5 wire test instrument for one VILLAIN snare:
closed-form table head (constants from data/villain_modes.json, same tensor form as
the e3proto Latent Snare bank) + the E5 snare-wire contact layer (WIRE_SNIPPET, the
exact text to splice into scripts/e3proto/gen_dsp.py).
    python3 gen_test.py snare_00_v100 [--nstr 6] [--out DIR]
Outputs: ch1 wires (pre band-pass; comparable to stems/<id>.strands.wav),
ch2 wires through the wire band-pass (what goes into the drum bus), ch3 head."""
import argparse, json, math, os
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e5")
MODES = json.load(open(os.path.join(HERE, "..", "data/villain_modes.json")))["renders"]
TUNING = 2 * 261.6255653 / 440
VDET = [0.906, 1.062, 0.951, 1.114, 0.874, 1.147]
DET = {6: VDET, 4: [0.906, 1.062, 0.951, 1.114], 3: [0.906, 1.062, 0.951], 2: [0.951, 1.062], 1: [1.0]}

def tl(vals):
    return f"(tensor @shape [1 {len(vals)}] @data [{' '.join(f'{v:.9g}' for v in vals)}])"

# ---------------------------------------------------------------------------------
# The splice. Inputs it expects in scope: `head` (audio-rate head displacement, the
# summed modal bank), `hit` (onset pulse), and the latched knobs w-drive, w-gap, w-hz,
# w-t60 (ms), w-level, w-lo, w-hi, w-k. Defines `wire-raw` (contact signal, pre BP) and
# `wire-out` (band-passed; add it to `dry`).
def wire_snippet(nstr, reset=False):
    """Inputs in scope: `head` (audio-rate summed modal bank), `hit` (onset pulse) and EIGHT
    EVENT-CLOCK scalars (decoder outputs or (event-hold knob hit)): ev-w-drive, ev-w-gap,
    ev-w-hz, ev-w-t60 (ms), ev-w-level, ev-w-lo, ev-w-hi, ev-w-k. All coefficient math runs on
    the hit's event clock; per sample only the strand recursions, the contact and 2 SVFs run.
    Defines `wire-raw` (contact signal, pre band-pass) and `wire-out` (add it to `dry`).
    reset=True zeroes the wire state on each hit (VILLAIN); False lets wires keep moving."""
    det = DET[nstr]
    rz = (lambda s: f"(gswitch hit 0 {s})") if reset else (lambda s: s)
    return f""";; ---- snare wires (E5): {nstr} detuned 2-mode strands, one-sided contact (VILLAIN / Modal Snare) ----
;; z = 0.06 drive head; per strand w = w1 + w2/2 (damped 2-poles at hz and 2.41 hz, T60 t60 / 0.4 t60 ms),
;; ov = clip(z - w - 0.06 gap, 0, 0.02); push = ov (1 + 0.8 min(1000 ov, 1)) kicks both poles;
;; contact = sum ov -> two leaky DC blockers -> x60 level -> SVF HP lo -> SVF LP hi (damping k).
(def w-det {tl(det)})
(def ev-w-r1 (exp (/ -6.9077553 (* samplerate 0.001 ev-w-t60))))
(def ev-w-r2 (exp (/ -6.9077553 (* samplerate 0.0004 ev-w-t60))))
(def ev-w-wc1 (* twopi (/ (clip ev-w-hz 20 (* 0.2 samplerate)) samplerate)))
(def ev-w-wc2 (* twopi (/ (clip (* 2.41 ev-w-hz) 20 (* 0.3 samplerate)) samplerate)))
(def w-c1 (latch (* 2 ev-w-r1 (cos (* ev-w-wc1 w-det))) hit))
(def w-c2 (latch (* 2 ev-w-r2 (cos (* ev-w-wc2 w-det))) hit))
(def w-q1 (event-latch (* ev-w-r1 ev-w-r1) hit))
(def w-q2 (event-latch (* ev-w-r2 ev-w-r2) hit))
(def w-zk (event-latch (* 0.06 ev-w-drive) hit))
(def w-gv (event-latch (* 0.06 ev-w-gap) hit))
(def w-lv (event-latch (* 60 ev-w-level) hit))
(def w-glo (event-latch (wire-svf-g ev-w-lo) hit))
(def w-ghi (event-latch (wire-svf-g ev-w-hi) hit))
(def w-k (event-latch ev-w-k hit))
(def w-z (* w-zk head))
(make-tensor-history wy1 @shape [1 {nstr}])
(make-tensor-history wy2 @shape [1 {nstr}])
(make-tensor-history wu1 @shape [1 {nstr}])
(make-tensor-history wu2 @shape [1 {nstr}])
(def w-y1 {rz("(read-tensor-history wy1)")})
(def w-y2 {rz("(read-tensor-history wy2)")})
(def w-u1 {rz("(read-tensor-history wu1)")})
(def w-u2 {rz("(read-tensor-history wu2)")})
(def w-m1 (- (* w-c1 w-y1) (* w-q1 w-y2)))
(def w-m2 (- (* w-c2 w-u1) (* w-q2 w-u2)))
(def w-ov (min (max (- w-z w-m1 (* 0.5 w-m2) w-gv) 0) 0.02))
(def w-push (* w-ov (+ 1 (* 0.8 (min (* w-ov 1000) 1)))))
(write-tensor-history wy2 w-y1)
(write-tensor-history wy1 (+ w-m1 (* w-push 0.769)))
(write-tensor-history wu2 w-u1)
(write-tensor-history wu1 (+ w-m2 (* w-push 0.4615)))
(def w-app (sum w-ov))
(make-history wcm1)
(make-history wcm2)
(def w-cp1 {rz("(read-history wcm1)")})
(def w-cl1 (+ w-cp1 (* 0.1 (- w-app w-cp1))))
(write-history wcm1 w-cl1)
(def w-ca (- w-app w-cl1))
(def w-cp2 {rz("(read-history wcm2)")})
(def w-cl2 (+ w-cp2 (* 0.1 (- w-ca w-cp2))))
(write-history wcm2 w-cl2)
(def wire-raw (* w-lv (- w-ca w-cl2)))
(def wire-out (wire-svf-lp (wire-svf-hp wire-raw w-glo 1.414213562) w-ghi w-k))
"""

SVF_MACROS = """;; TPT state-variable filter steps (VILLAIN svf-lp / svf-hp, no onset reset) + Pade prewarp
(defmacro wire-svf-g (hz)
  (def x (/ (* pi (clip hz 10 (* 0.33 samplerate))) samplerate))
  (/ (* x (- 15 (* x x))) (- 15 (* 6 x x))))
(defmacro wire-svf-lp (x gk kk)
  (make-history s1)
  (make-history s2)
  (def ic1 (read-history s1))
  (def ic2 (read-history s2))
  (def a1 (/ 1 (+ 1 (* gk (+ gk kk)))))
  (def a2 (* gk a1))
  (def a3 (* gk a2))
  (def v3 (- x ic2))
  (def v1 (+ (* a1 ic1) (* a2 v3)))
  (def v2 (+ ic2 (* a2 ic1) (* a3 v3)))
  (write-history s1 (- (* 2 v1) ic1))
  (write-history s2 (- (* 2 v2) ic2))
  v2)
(defmacro wire-svf-hp (x gk kk)
  (make-history s1)
  (make-history s2)
  (def ic1 (read-history s1))
  (def ic2 (read-history s2))
  (def a1 (/ 1 (+ 1 (* gk (+ gk kk)))))
  (def a2 (* gk a1))
  (def a3 (* gk a2))
  (def v3 (- x ic2))
  (def v1 (+ (* a1 ic1) (* a2 v3)))
  (def v2 (+ ic2 (* a2 ic1) (* a3 v3)))
  (write-history s1 (- (* 2 v1) ic1))
  (write-history s2 (- (* 2 v2) ic2))
  (- x (* kk v1) v2))
"""

# knob: (default-key or value, min, max, doc)
KNOBS = [("wire_drive", 0.01, 30.0), ("wire_gap", 0.0, 1.0), ("wire_hz", 100.0, 3000.0),
         ("wire_t60", 0.2, 20.0), ("wire_level", 0.0, 5000.0), ("wire_lo", 20.0, 16000.0),
         ("wire_hi", 100.0, 16000.0), ("wire_k", 0.05, 2.0)]

def defaults(rid):
    c = MODES[rid]["chain"]
    return dict(wire_drive=c["c_zs"], wire_gap=c["c_gap"], wire_hz=c["c_wire_hz"],
                wire_t60=max(1.0, c["c_wire_ms"]), wire_level=c["c_rattle"],
                wire_lo=c["c_buzz_lo"], wire_hi=c["c_buzz_hi"], wire_k=c["c_buzz_k"])

def gen(rid, nstr):
    r = MODES[rid]; c = r["chain"]; S = r["slots"]
    f = [TUNING * s["f"] for s in S]
    sh = [TUNING * max(-0.9 * s["f"], s["g0"] * s["f"]) for s in S]
    amp = [s["amp"] for s in S]
    nyq = [min(max((0.48 * 48000 - fi - max(si, 0)) / (0.08 * 48000), 0), 1) for fi, si in zip(f, sh)]
    amp = [a * n for a, n in zip(amp, nyq)]
    dfl = defaults(rid)
    params = "\n".join(f"(param {k} @default {dfl[k]:.9g} @min {lo} @max {hi})" for k, lo, hi in KNOBS)
    return f""";; E5 wire test instrument for {rid} -- GENERATED by gen_test.py, research only.
(def gate (in 1 @name gate))
(def pitch (in 2 @name pitch))
(def velocity (in 3 @name velocity))
(def trigger (in 4 @name trigger))
{params}
(defmacro onset-of (trig)
  (make-history previous)
  (def high (gt trig 0.5))
  (def rising (* high (lte (read-history previous) 0.5)))
  (write-history previous high)
  rising)
(defmacro hit-age (onset)
  (make-history elapsed)
  (make-history armed)
  (def played (max onset (read-history armed)))
  (write-history armed played)
  (def samples (gswitch onset 0 (min (* 30 samplerate) (+ 1 (read-history elapsed)))))
  (write-history elapsed samples)
  (def alive (* played (lt samples (* 30 samplerate))))
  (* alive (/ samples samplerate)))
{SVF_MACROS}
(def hit (onset-of trigger))
(def age (hit-age hit))
(def vel (latch (clip velocity 0 1) hit))
;; ---- table head (VILLAIN head_modes = head_v * head_sum, ungated), closed form ----
(def m-f {tl(f)})
(def m-amp {tl(amp)})
(def m-idec {tl([1 / s['dec0'] for s in S])})
(def m-irise {tl([1 / s['rise0'] for s in S])})
(def m-t0 {tl([s['t0'] for s in S])})
(def m-phc {tl([s['ph'] / (2 * math.pi) for s in S])})
(def m-itau {tl([1 / s['tau'] for s in S])})
(def m-shtau {tl([a * s['tau'] for a, s in zip(sh, S)])})
(def head-T (max 0 (- age {c['c_pre']:.9g})))
(def head-v (* {c['c_head']:.9g} (exp (* {c['c_hv']:.9g} (log (max vel 0.0001)))) (- 1 (exp (/ (- 0 head-T) {c['c_hr']:.9g})))))
(def mode-t (max 0 (- head-T m-t0)))
(def mode-env (* m-amp (exp (* -1 mode-t m-idec)) (- 1 (exp (* -1 mode-t m-irise)))))
(def mode-cyc (+ (* m-f mode-t) (* m-shtau (- 1 (exp (* -1 mode-t m-itau)))) m-phc))
(def head (* head-v (sum (* mode-env (sin (* twopi mode-cyc))))))
;; ---- knobs, sampled on the hit's event clock (as decoder outputs would be) ----
(def ev-w-drive (event-hold wire_drive hit))
(def ev-w-gap (event-hold wire_gap hit))
(def ev-w-hz (event-hold wire_hz hit))
(def ev-w-t60 (event-hold wire_t60 hit))
(def ev-w-level (event-hold wire_level hit))
(def ev-w-lo (event-hold wire_lo hit))
(def ev-w-hi (event-hold wire_hi hit))
(def ev-w-k (event-hold wire_k hit))
{wire_snippet(nstr, reset=True)}
(out wire-raw 1 @name wires)
(out wire-out 2 @name wires_bp)
(out head 3 @name head)
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("rid"); ap.add_argument("--nstr", type=int, default=6)
    ap.add_argument("--out"); a = ap.parse_args()
    d = a.out or os.path.join(HERE, "WireTest", f"{a.rid}_n{a.nstr}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "dsp.lisp"), "w").write(gen(a.rid, a.nstr))
    json.dump({"version": 1, "run_mode": "instrument", "current": 1,
               "releases": {"1": {"path": ".", "name": f"Research/E5 Wire {a.rid}"}}},
              open(os.path.join(d, "instrument.json"), "w"), indent=1)
    print(d)
