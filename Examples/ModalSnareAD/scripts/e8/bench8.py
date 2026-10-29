"""CPU: scripts/e3proto/bench.py method (128-frame blocks, 48k, hit every 0.5 s), per param set."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
MAD.use("e3proto")
import bench
from lib8 import *
drv = bench.driver(); drv.set_hit_period(0.5)
A = zp(CODES["SP01"]); B = zp(CODES["ZV"], "b")
cases = [("installed", INST, {**A}), ]
for h in (128, 256):
    d = os.path.join(HERE, "Latent Snare Morph") if h == 128 else os.path.join(HERE, "build", f"h{h}")
    cases += [(f"h{h} static (A=B)", d, {**A, **zp(CODES['SP01'], 'b')}),
              (f"h{h} morph 400ms", d, {**A, **B, "morph": 400.0}),
              (f"h{h} morph 4000ms (ticks always)", d, {**A, **B, "morph": 4000.0})]
for name, d, p in cases:
    inst = Instrument(d)
    xs = sorted(bench.measure(inst, 20.0, params=p, run=drv.run) for _ in range(5))
    print(f"{name:34s} median {xs[2]:7.2f} us/block (min {xs[0]:.2f}, max {xs[-1]:.2f})", flush=True)
