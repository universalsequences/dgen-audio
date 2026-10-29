"""Run VILLAIN's full chain with an injected strand signal (villain_inject/)."""
import ctypes, numpy as np
from e5lib import *
from villain_common import load as _l, C4
from audition import Instrument
_I = {}
def inst():
    if "v" not in _I: _I["v"] = Instrument(os.path.join(HERE, "villain_inject"), sample_rate=48000.0, max_frames=128)
    return _I["v"]
def render_inject(params, ext, vel=1.0):
    I = inst(); n = len(ext); blk = I.max_frames
    mem = I.fresh_memory()
    for k, v in params.items(): mem[I.params[k]["cellId"]] = v
    ins = [np.zeros(blk, np.float32) for _ in range(I.n_in)]; outs = [np.zeros(blk, np.float32) for _ in range(I.n_out)]
    ins[I.inputs["pitch"]][:] = C4; ins[I.inputs["velocity"]][:] = vel; ins[I.inputs["gate"]][:] = 1
    ip = (ctypes.POINTER(ctypes.c_float) * I.n_in)(*[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in ins])
    op = (ctypes.POINTER(ctypes.c_float) * I.n_out)(*[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in outs])
    e = np.zeros(n + blk, np.float32); e[:n] = ext
    y = np.zeros(n, np.float32); tch = I.inputs["trigger"]; xch = I.inputs["ext_strands"]
    for b in range(0, n, blk):
        fr = min(blk, n - b)
        ins[tch][:] = 0
        if b == 0: ins[tch][0] = 1
        ins[xch][:] = e[b:b + blk]
        I.process_fn(ip, op, fr, mem.ctypes.data_as(ctypes.c_void_p), ctypes.byref(I.context), None)
        y[b:b + fr] = outs[0][:fr]
    return y.astype(np.float64)
