#!/usr/bin/env python3
"""Research-only copy of VILLAIN Snare whose snare-hit also returns stem taps
(ch1 output, ch2 head_g, ch4 source, ch5 strands; ch3 amp) -> $MODAL_AD_ROOT/villain_stems/.
The main output stays bit-identical to the shipped instrument."""
import os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402

src_dir = os.path.join(MAD.eseq_root(), "content/instruments/Drums/VILLAIN Snare")
dst = MAD.stage("villain_stems")
src = open(os.path.join(src_dir, "dsp.lisp")).read()
old_end = "  output)\n\n(def release_v"
assert src.count(old_end) == 1
src = src.replace(old_end, "  (tuple output head_g source strands))\n\n(def release_v")
old = """(def hit_a (snare-hit trigger_a pitch velocity))
(def hit_b (snare-hit trigger_b pitch velocity))
(def mixed (clip (+ (* hit_a gain_a) (* hit_b gain_b)) (- 0 c_mix_clip) c_mix_clip))
(out mixed 1 @name left)
(out mixed 2 @name right)"""
new = """(def (hit_a head_a src_a str_a) (snare-hit trigger_a pitch velocity))
(def (hit_b head_b src_b str_b) (snare-hit trigger_b pitch velocity))
(def mixed (clip (+ (* hit_a gain_a) (* hit_b gain_b)) (- 0 c_mix_clip) c_mix_clip))
(out mixed 1 @name left)
(out (+ (* head_a gain_a) (* head_b gain_b)) 2 @name head)
(out (+ (* src_a gain_a) (* src_b gain_b)) 4 @name source)
(out (+ (* str_a gain_a) (* str_b gain_b)) 5 @name strands)"""
assert src.count(old) == 1
src = src.replace(old, new)
src = src.replace(";; VILLAIN Snare:", ";; RESEARCH COPY (stem taps: ch1 output, ch2 head_g, ch4 source, ch5 strands; ch3 amp). Not shipped.\n;; VILLAIN Snare:", 1)
open(os.path.join(dst, "dsp.lisp"), "w").write(src)
shutil.copy(os.path.join(src_dir, "instrument.json"), dst)
print("wrote", dst)
