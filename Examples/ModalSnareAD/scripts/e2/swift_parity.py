"""Swift trainer render (latent-render) vs numpy twin, same weights, same E1 noise bank."""
import json, os, subprocess, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); MAD.use("e2")
MAD.use("dataset")
import template as tp, latent_np as ln
from villain_common import read_wav_f32
run = sys.argv[1]; outd = os.path.join(run, "swift_parity")
W, Z, names = ln.load_run(run)
rng = np.random.default_rng(1)
zs = [Z[0], Z[12], Z[20], rng.normal(0, 1, Z.shape[1])]; vels = [1.0, 0.7, 0.4, 0.85]
nm = ["c0", "c12", "c20", "rand"]
spec = os.path.join(run, "render_spec.json")
json.dump({"names": nm, "vel": vels, "z": [list(map(float, z)) for z in zs]}, open(spec, "w"))
B = MAD.binary("ModalSnareAD")
subprocess.run([B, "latent-render", "--ckpt", os.path.join(run, "checkpoint.json"), "--spec", spec, "--out", outd + "/"], check=True)
bank = ln.e1_bank()
for n, z, v in zip(nm, zs, vels):
    y, h, _ = tp.render(W, z, v, level=1.0, bands=bank, parts=True)
    ys = read_wav_f32(os.path.join(outd, n + ".swift.wav")); hs = read_wav_f32(os.path.join(outd, n + ".swift_head.wav"))
    e = lambda a, b: np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2))
    print(f"{n}: out rel {e(ys, y):.2e}  head rel {e(hs, h):.2e}  peak {np.abs(y).max():.3f}")
