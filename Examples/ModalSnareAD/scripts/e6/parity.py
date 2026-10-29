#!/usr/bin/env python3
"""Compiled E6 instrument vs the numpy twin (trainer-space decoder, e7lib wire layer, the instrument's own
xorshift band noise): single snares (A = B), warped blends, offsets, velocity, note tracking (C5 = 2x).
Band-envelope dB (the wire layer is chaotic, so waveforms diverge after ~1000 samples) + early-window rel RMS.
usage: parity.py RUN WARP.json INST_DIR"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
C.MAD.use("e7")
import e7lib
from warp import tau_of
run, warp_path, inst_dir = sys.argv[1:4]
C.tp.env_setup()
from audition import Instrument
W, Z, names = C.load_run(run); mu, M, Zk = C.knob_transform(Z)
T = np.array(json.load(open(warp_path))["table"])
inst = Instrument(inst_dir)
n = 48000; bands = C.tp.bandpass_bank(C.tp.xorshift_white(n))
C4 = 261.6255653


def twin(a, b, t, off=(0, 0, 0, 0), vel=1.0, even=1, pitch=C4):
    tau = tau_of(T[a, b], t) if even else t
    zk = np.clip((1 - tau) * Zk[a] + tau * Zk[b] + np.array(off), -3, 3)
    p = C.decode(W, mu + zk @ M, vel)
    p = dict(p); p["mode_freq"] = np.asarray(p["mode_freq"]) * min(max(pitch / C4, 0.25), 4)
    knobs = {k: float(np.ravel(p[k])[0]) for k in e7lib.WIRE}
    head, noise, chain, _ = e7lib.parts(p, n, True, bands=bands)
    return np.asarray(chain(head + noise + e7lib.wire_out(head, knobs), 1.0), float)


cases = [(k, k, 0.0, (0, 0, 0, 0), 1.0, C4) for k in (0, 5, 12, 19, 26)]
cases += [(3, 21, 0.3, (0, 0, 0, 0), 1.0, C4), (8, 30, 0.5, (0, 0, 0, 0), 1.0, C4), (14, 2, 0.8, (0, 0, 0, 0), 0.7, C4),
          (6, 6, 0.0, (0.3, -0.2, 0.1, 0), 1.0, C4), (10, 25, 0.5, (0, 0, 0, 0), 1.0, 2 * C4)]
rows = []
for a, b, t, off, vel, pitch in cases:
    y, _ = inst.render(1.0, vel=vel, pitch=pitch, params={"snare_a": a, "snare_b": b, "blend": t, "even": 1,
                                                          "z1": off[0], "z2": off[1], "z3": off[2], "z4": off[3], "tune": 0, "decay": 1, "level": 1.0})
    y = (y if y.ndim == 1 else y[:, 0]).astype(float); r = twin(a, b, t, off, vel, pitch=pitch)
    e = int(0.02 * 48000)
    rows.append({"a": names[a], "b": names[b], "t": t, "off": off, "vel": vel, "pitch": pitch, "band_db": C.band(y, r),
                 "early_rel_rms": float(np.sqrt(np.mean((y[:e] - r[:e]) ** 2) / np.mean(r[:e] ** 2))), "peak": float(np.abs(y).max())})
    print(rows[-1])
print("median band dB", np.median([r["band_db"] for r in rows]), "max", max(r["band_db"] for r in rows))
C.json_dump(rows, os.path.join(C.HERE, "parity.json"))
