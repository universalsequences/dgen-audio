#!/usr/bin/env python3
"""Native per-voice CPU timing of an eseq instrument dylib (method of
tools/pm-gamelan/performance.py): 128-frame blocks, 2 hits/s, CPU time of the
process calls only. Usage: bench.py DIR [DIR...] [--seconds 20] [--reps 5]"""
import argparse, ctypes as C, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e3proto")
MAD.audition()
from audition import Instrument

def driver():
    lib = os.path.join(HERE, "bench_driver.dylib")
    src = os.path.join(MAD.code("e3proto"), "bench_driver.c")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        subprocess.run(["cc", "-O2", "-dynamiclib", "-fPIC", "-I", MAD.code("e3proto"), src, "-o", lib], check=True)
    d = C.CDLL(lib); d.run.restype = C.c_double
    d.set_hit_period.argtypes = [C.c_double]
    d.run.argtypes = [C.c_void_p] * 3 + [C.c_uint32] + [C.c_void_p] * 3 + [C.c_uint32]
    return d

def measure(inst, seconds=20.0, params=None, vel=1.0, run=None):
    frames = inst.max_frames
    ins = np.zeros((inst.n_in, frames), np.float32); outs = np.zeros((inst.n_out, frames), np.float32)
    st = inst.fresh_memory()
    for k, v in (params or {}).items():
        st[inst.params[k]["cellId"]] = v
    for k, v in [("pitch", 261.6256), ("velocity", vel), ("gate", 1.0)]:
        if k in inst.inputs: ins[inst.inputs[k]] = v
    ptr = lambda arrs: (C.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
    blocks = round(seconds * inst.sample_rate / frames)
    el = run(C.cast(inst.process_fn, C.c_void_p), ptr(ins), ptr(outs), frames, st.ctypes.data,
             C.byref(inst.context), ins[inst.inputs["trigger"]].ctypes.data, blocks)
    assert np.isfinite(st).all() and np.isfinite(outs).all()
    return el / blocks * 1e6

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seconds", type=float, default=20); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--period", type=float, default=0.5, help="seconds between hits")
    a = ap.parse_args(); drv = driver(); drv.set_hit_period(a.period); run = drv.run
    for d in a.dirs:
        inst = Instrument(d)
        xs = sorted(measure(inst, a.seconds, run=run) for _ in range(a.reps))
        budget = inst.max_frames / inst.sample_rate * 1e6
        print(f"{os.path.basename(d.rstrip('/')):28s} median {xs[len(xs)//2]:7.2f} us/block (min {xs[0]:.2f}, max {xs[-1]:.2f})"
              f"  = {100*xs[len(xs)//2]/budget:.2f}% of one core @128fr/48k   [{inst.build_dir}]")
