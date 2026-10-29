#!/usr/bin/env python3
"""Check builds of the morph, in $MODAL_AD_ROOT/kick/morph/build:
  h128     hop 128, as shipped                          (t_parity.py)
  h128tap  hop 128 + taps (4 modal, 5 noise)            (t_click.py)
  h256tap  hop 256 + taps                               (t_click.py)
  h1tap    decodes every sample (no hop boundaries)     (t_click.py reference)"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402

os.chdir(MAD.stage("kick/morph"))
gm = os.path.join(MAD.code("kick/morph"), "gen_morph.py")
for args in (["build/h128"], ["build/h128tap", "--tap"], ["build/h256tap", "--hop", "256", "--tap"],
             ["build/h1tap", "--hop", "1", "--always", "--tap"]):
    subprocess.run([sys.executable, gm] + args, check=True)
print("ok")
