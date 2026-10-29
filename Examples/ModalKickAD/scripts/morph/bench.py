"""CPU: ModalSnareAD/scripts/e3proto/bench.py method (128-frame blocks, 48 k, hit every 0.5 s), median of 5 x 20 s.
usage: bench.py [MORPH_DIR]   (default: the staged Latent Kick Morph)"""
import sys
from lib import *
MAD.use("e3proto")
import bench
drv = bench.driver(); drv.set_hit_period(0.5)
SNM = os.path.join(MAD.ROOT, "e8", "Latent Snare Morph")
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Latent Kick Morph")
A, B = zp(CODE["K55"]), zp(CODE["K62"], "b")
cases = [("Latent Kick (installed)", INST, {"kick_a": 3, "kick_b": 3}),
         ("Morph static (START=END)", d, {**A, **zp(CODE["K55"], "b")}),
         ("Morph 400 ms", d, {**A, **B, "morph": 400.0}),
         ("Morph 4000 ms (ticks always)", d, {**A, **B, "morph": 4000.0}),
         ("Latent Snare Morph static", SNM, {}),
         ("Latent Snare Morph 400 ms", SNM, {"z1": -0.48858, "z2": 0.218, "z3": 0.63605, "z4": 0.19747, "b1": -0.9, "b2": 0.5, "morph": 400.0})]
for name, dd, p in cases:
    inst = load(dd)
    xs = sorted(bench.measure(inst, 20.0, params=p, run=drv.run) for _ in range(5))
    print(f"{name:32s} median {xs[2]:7.2f} us/block (min {xs[0]:.2f}, max {xs[-1]:.2f})", flush=True)
