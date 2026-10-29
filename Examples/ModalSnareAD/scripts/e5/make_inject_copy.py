#!/usr/bin/env python3
"""E5 research copy of the VILLAIN Snare stem-tap copy whose `strands` signal comes from
input 6 (ext_strands) -> $MODAL_AD_ROOT/e5/villain_inject/ (inject.py runs VILLAIN's real chain
with a model's strand signal). Needs $MODAL_AD_ROOT/villain_stems (dataset/make_stems_copy.py)."""
import os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402

src_dir = os.path.join(MAD.ROOT, "villain_stems")
dst = os.path.join(MAD.stage("e5"), "villain_inject")
os.makedirs(dst, exist_ok=True)
src = open(os.path.join(src_dir, "dsp.lisp")).read()
old = "(def strands (* c_rattle (clip rattle 0 4) (vpow c_rattle_ve) strands_f))"
assert src.count(old) == 1
src = src.replace(old, "(def strands ext_strands)")
a = "(def clock (in 5 @name clock))"
assert src.count(a) == 1
src = src.replace(a, a + "\n;; E5: the strand signal is injected from input 6 (replaces VILLAIN's own strands)\n(def ext_strands (in 6 @name ext_strands))")
src = ";; E5 RESEARCH COPY of villain_stems/dsp.lisp: `strands` comes from input 6 (ext_strands).\n" + src
open(os.path.join(dst, "dsp.lisp"), "w").write(src)
shutil.copy(os.path.join(src_dir, "instrument.json"), dst)
print("wrote", dst)
