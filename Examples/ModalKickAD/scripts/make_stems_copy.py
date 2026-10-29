#!/usr/bin/env python3
"""Research-only copy of VILLAIN Kick whose kick-hit also returns stem taps.
Main output (channel 0) must stay bit-identical to the shipped instrument."""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
KICK = MAD.stage("kick")
REPO = MAD.eseq_root()
os.makedirs(os.path.join(KICK, "villain_stems"), exist_ok=True)
SRC = os.path.join(REPO, "content/instruments/Drums/VILLAIN Kick/dsp.lisp")
DST = os.path.join(KICK, "villain_stems", "dsp.lisp")
s = open(SRC).read()
TAPS = ["modal", "bank_all", "body_sig", "source", "texture", "limited"]
old_ret = "  (clip output (- c_out_clip) c_out_clip))\n"
assert s.count(old_ret) == 1
new_ret = ("  (def modal (* onset_ramp bank_sum))\n"
           "  (tuple (clip output (- c_out_clip) c_out_clip) " + " ".join(TAPS) + "))\n")
s = s.replace(old_ret, new_ret)
old_hits = "(def hit_a (kick-hit trigger_a pitch velocity))\n(def hit_b (kick-hit trigger_b pitch velocity))\n"
assert s.count(old_hits) == 1
s = s.replace(old_hits,
    "(def (hit_a " + " ".join(f"{t}_a" for t in TAPS) + ") (kick-hit trigger_a pitch velocity))\n"
    "(def (hit_b " + " ".join(f"{t}_b" for t in TAPS) + ") (kick-hit trigger_b pitch velocity))\n")
old_out = "(out mixed 2 @name right)\n"
assert s.count(old_out) == 1
outs = "".join(f"(out (+ (* {t}_a gain_a) (* {t}_b gain_b)) {ch} @name {t})\n" for t, ch in zip(TAPS, [2, 4, 5, 6, 7, 8]))
s = s.replace(old_out, outs)
open(DST, "w").write(s)
json.dump({"version": 1, "run_mode": "instrument"}, open(os.path.join(KICK, "villain_stems", "instrument.json"), "w"))
print("wrote", DST, "taps (channel order after out):", TAPS)
