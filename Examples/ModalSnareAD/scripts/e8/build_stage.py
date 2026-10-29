"""Stage Latent Snare Morph in $MODAL_AD_ROOT/e8/Latent Snare Morph (+ presets from presets_renders.py) and
optionally install it. usage: build_stage.py [--hop 128] [--install DIR]
DIR = an eseq instrument library folder, e.g. <eseq user-instruments>/Research (`eseq paths`)."""
import argparse, os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e8")
SRC = os.path.join(MAD.ROOT, "e6", "stage", "Latent Snare")  # the A/B Latent Snare (scripts/e6/build_instrument.py)
STAGE = os.path.join(HERE, "Latent Snare Morph")
ap = argparse.ArgumentParser(); ap.add_argument("--hop", type=int, default=128); ap.add_argument("--install", metavar="DIR")
a = ap.parse_args()
subprocess.run([sys.executable, os.path.join(MAD.code("e8"), "gen_morph.py"), STAGE, "--hop", str(a.hop)], check=True)
os.makedirs(os.path.join(STAGE, "data"), exist_ok=True)
shutil.copy(os.path.join(SRC, "data", "family.json"), os.path.join(STAGE, "data", "family.json"))
for f in ("codes_std.json", "decoder_layout.json"):
    shutil.copy(os.path.join(SRC, f), STAGE)
ui = open(os.path.join(SRC, "ui.lisp")).read()
codes = ui[ui.index(";; BEGIN GENERATED LATENT CODES"):ui.index(";; END GENERATED LATENT CODES") + len(";; END GENERATED LATENT CODES")]
codes = codes.replace("instruments/Research/Latent Snare/data/family.json", "instruments/Research/Latent Snare Morph/data/family.json")
codes = "\n".join(l for l in codes.splitlines() if not l.startswith(("(def ls-x-param", "(def ls-y-param", "(def ls-knob-params")))
open(os.path.join(STAGE, "ui.lisp"), "w").write(open(os.path.join(MAD.code("e8"), "ui.tmpl.lisp")).read().replace("@@CODES@@", codes))
open(os.path.join(STAGE, "instrument.json"), "w").write('{"version":1,"run_mode":"instrument"}\n')
if a.install:
    DEST = os.path.join(a.install, "Latent Snare Morph")
    if os.path.exists(DEST): shutil.rmtree(DEST)
    shutil.copytree(STAGE, DEST)
    pres = os.path.join(HERE, "Latent Snare Morph.presets")
    if os.path.exists(pres): shutil.copy(pres, os.path.dirname(DEST))
    print("installed", DEST)
