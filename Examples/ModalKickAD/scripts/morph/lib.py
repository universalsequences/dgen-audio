"""Latent Kick Morph helpers: instruments, codes, render, metrics."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("kick/morph"); KICK = MAD.stage("kick")
MAD.use("kick")
os.environ["ESEQ_DGENLISP_TOOL"] = MAD.dgenlisp()
os.chdir(HERE)  # build/, renders/ and the result JSONs live in $MODAL_AD_ROOT/kick/morph
from kick_common import Instrument, C4, write_wav_f32, relrms  # noqa
SR = 48000
INST = os.path.join(KICK, "stage", "Latent Kick")  # the A/B Latent Kick (scripts/build_instrument.py)
STD = json.load(open(os.path.join(INST, "codes_std.json")))
NAMES, CODES = STD["names"], STD["codes"]
CODE = {n: [float(f"{v:.6g}") for v in c] for n, c in zip(NAMES, CODES)}   # as compiled into LK-CODE

def zp(code, pre="z"):
    return {f"{pre}{i+1}": float(code[i]) for i in range(4)}

def load(d):
    return Instrument(d, sample_rate=48000.0, max_frames=128)

def render(inst, seconds=1.0, vel=1.0, params=None, pitch=C4):
    y, _ = inst.render(seconds, pitch=pitch, vel=vel, params=params or {})
    return np.asarray(y, np.float64)
