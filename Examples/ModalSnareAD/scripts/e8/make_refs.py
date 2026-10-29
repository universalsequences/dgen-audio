#!/usr/bin/env python3
"""Reference builds for the E8 checks, in $MODAL_AD_ROOT/e8:
  ref/headtap   the A/B Latent Snare (e6/stage) + out 3 = head, out 4 = wire-out   (t_parity.py)
  ref/nowire    the same without the chaotic wire layer
  build/h128tap, build/h256tap  the morph at hop 128 / 256 with the same taps     (t_parity.py, t_click.py)
  build/h1tap   the morph decoding every sample (no hop boundaries)               (t_click.py)
  build/h256    the morph at hop 256                                              (bench8.py)"""
import glob, os, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402

src = os.path.join(MAD.ROOT, "e6", "stage", "Latent Snare")
os.chdir(MAD.stage("e8"))
TAP = "(out (voice-amp 0 voice-out voice-out) 2 @name amp @amp true)"
dsp = open(os.path.join(src, "dsp.lisp")).read()
assert dsp.count(TAP) == 1
dsp = dsp.replace(TAP, TAP + "\n(out head 3 @name head_tap)\n(out wire-out 4 @name wire_tap)")
for d, text in (("ref/headtap", dsp), ("ref/nowire", dsp.replace("(def dry (+ head noise-out wire-out))", "(def dry (+ head noise-out))"))):
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.json")):
        if os.path.basename(f) != "instrument.json":
            shutil.copy(f, d)
    open(os.path.join(d, "dsp.lisp"), "w").write(text)
    open(os.path.join(d, "instrument.json"), "w").write('{"version":1,"run_mode":"instrument"}\n')
gm = os.path.join(MAD.code("e8"), "gen_morph.py")
for args in (["build/h128tap", "--hop", "128", "--headtap"], ["build/h256tap", "--hop", "256", "--headtap"],
             ["build/h1tap", "--hop", "1", "--headtap", "--always"],
             ["build/h256", "--hop", "256"]):
    subprocess.run([sys.executable, gm] + args, check=True)
print("ok")
