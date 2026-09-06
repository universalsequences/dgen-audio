#!/usr/bin/env python3
"""Compile each bench/perf/*.lisp with the local DGenLisp and time dgen_process_v1.

Usage: python3 bench/perf/bench.py [--compiler PATH] [--toolchain-root DIR] [--ref DIR] [names...]
Prints us/block and, with --ref, a bit-exactness check of the first blocks
against reference output (.npy) saved from a previous run via --save DIR.
"""
import argparse, ctypes, json, os, subprocess, sys, time, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SR, BLK, SECONDS = 48000, 128, 20

class Ctx(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32), ("sample_rate", ctypes.c_float)]

def compile_one(src, compiler, toolchain, out):
    cmd = [compiler, str(src), "-o", str(out), "--name", "patch", "--sample-rate", str(SR), "--max-frames", str(BLK), "--skip-inline-audit"]
    if toolchain: cmd += ["--toolchain-root", toolchain]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=src.parent)
    if r.returncode: sys.exit(f"compile failed for {src.name}:\n{r.stderr[-3000:]}")
    return json.load(open(out/"patch.json"))

def run(out, manifest):
    lib = ctypes.CDLL(str(out/"patch.dylib")); fn = lib.dgen_process_v1
    P = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = (ctypes.POINTER(P), ctypes.POINTER(P), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p); fn.restype = None
    mem = np.zeros(max(1024, manifest["totalMemorySlots"] + 16), np.float32)
    for p in manifest["params"]: mem[p["cellId"]] = p["default"]
    for t in manifest.get("tensorInitData", []): mem[t["offset"]:t["offset"]+len(t["data"])] = t["data"]
    ins = {i["name"]: np.zeros(BLK, np.float32) for i in manifest["inputs"]}
    inl = [ins[i["name"]] for i in sorted(manifest["inputs"], key=lambda i: i["channel"])]
    outs = [np.zeros(BLK, np.float32) for _ in manifest["outputs"]]
    for k, v in (("pitch", 220), ("velocity", 1), ("gate", 1)):
        if k in ins: ins[k][:] = v
    ip = (P*len(inl))(*[x.ctypes.data_as(P) for x in inl]); op = (P*len(outs))(*[x.ctypes.data_as(P) for x in outs])
    ctx = Ctx(1, ctypes.sizeof(Ctx), SR); mp = mem.ctypes.data_as(ctypes.c_void_p); cx = ctypes.byref(ctx)
    for k in ("note_on", "trigger"):
        if k in ins: ins[k][0] = 1
    trace = []
    fn(ip, op, BLK, mp, cx, None); trace.append(outs[0].copy())
    for k in ("note_on", "trigger"):
        if k in ins: ins[k][0] = 0
    for _ in range(7): fn(ip, op, BLK, mp, cx, None); trace.append(outs[0].copy())
    n = SR*SECONDS//BLK; best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n): fn(ip, op, BLK, mp, cx, None)
        best = min(best, time.perf_counter()-t)
    c = open(out/"patch.c").read()
    return best/n*1e6, np.concatenate(trace), c.count("for (int i = 0;"), c.count("[i] =")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--compiler", default=os.environ.get("ESEQ_DGENLISP_TOOL", str(ROOT/".build/release/DGenLisp")))
    ap.add_argument("--toolchain-root", default=os.environ.get("ESEQ_DGEN_TOOLCHAIN_ROOT"))
    ap.add_argument("--save", help="dir to write reference traces"); ap.add_argument("--ref", help="dir with reference traces to compare")
    ap.add_argument("--keep", help="dir to keep build outputs"); ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    files = sorted(HERE.glob("*.lisp")); files = [f for f in files if not a.names or f.stem in a.names]
    for f in files:
        with tempfile.TemporaryDirectory() as d:
            out = Path(a.keep)/f.stem if a.keep else Path(d)
            out.mkdir(parents=True, exist_ok=True)
            m = compile_one(f, a.compiler, a.toolchain_root, out)
            us, tr, loops, stores = run(out, m)
        line = f"{f.stem:20s} {us:8.2f} us/block  {us/1e6/(BLK/SR)*100:5.2f}% RT  loops={loops:4d} stores={stores:5d}"
        if a.save: Path(a.save).mkdir(parents=True, exist_ok=True); np.save(Path(a.save)/f"{f.stem}.npy", tr)
        if a.ref:
            r = np.load(Path(a.ref)/f"{f.stem}.npy"); d = float(np.max(np.abs(r-tr)))
            line += f"  maxdiff={d:.3g} {'EXACT' if d == 0 else ('ok' if d < 1e-5 else 'MISMATCH')}"
        print(line, flush=True)
main()
