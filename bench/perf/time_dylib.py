#!/usr/bin/env python3
"""Time dgen_process_v1 in prebuilt dylibs that share one patch.json: time_dylib.py <dir> a.dylib b.dylib ..."""
import sys, json, ctypes, time, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); m = json.load(open(d/"patch.json")); SR, BLK = 48000, 128
class Ctx(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32), ("sample_rate", ctypes.c_float)]
P = ctypes.POINTER(ctypes.c_float)
for name in sys.argv[2:]:
    fn = ctypes.CDLL(str(d/name)).dgen_process_v1
    fn.argtypes = (ctypes.POINTER(P), ctypes.POINTER(P), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p); fn.restype = None
    mem = np.zeros(max(1024, m["totalMemorySlots"]+16), np.float32)
    for p in m["params"]: mem[p["cellId"]] = p["default"]
    ins = {i["name"]: np.zeros(BLK, np.float32) for i in m["inputs"]}
    inl = [ins[i["name"]] for i in sorted(m["inputs"], key=lambda i: i["channel"])]
    outs = [np.zeros(BLK, np.float32) for _ in m["outputs"]]
    for k, v in (("pitch", 220), ("velocity", 1), ("gate", 1)):
        if k in ins: ins[k][:] = v
    ip = (P*len(inl))(*[x.ctypes.data_as(P) for x in inl]); op = (P*len(outs))(*[x.ctypes.data_as(P) for x in outs])
    ctx = Ctx(1, ctypes.sizeof(Ctx), SR); mp = mem.ctypes.data_as(ctypes.c_void_p); cx = ctypes.byref(ctx)
    for k in ("note_on", "trigger"):
        if k in ins: ins[k][0] = 1
    fn(ip, op, BLK, mp, cx, None)
    for k in ("note_on", "trigger"):
        if k in ins: ins[k][0] = 0
    n = SR*20//BLK; best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n): fn(ip, op, BLK, mp, cx, None)
        best = min(best, time.perf_counter()-t)
    print(f"{name:20s} {best/n*1e6:7.2f} us/block  peak={float(np.max(np.abs(outs[0]))):.4f}")
