#!/usr/bin/env python3
"""Evaluate VILLAIN Kick's block-rate section (top-level defs between the param block
and the ';; ---- helpers' section) in Python, exactly as the DSP computes them.

The per-hit copy inside kick-hit (the __h defs) is the same arithmetic with the table
selectors latched at the onset; with params constant over a render the two agree.
Transform knobs drop_time/tilt/spread/beat are NOT latched in VILLAIN (live block-rate),
tune/decay/bend/attack/body/knock/tick are latched at the hit (handled in kick_np).

    from kick_table import BlockRate
    env = BlockRate().eval(kick_a=3, kick_b=3)
"""
import math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402

REPO = MAD.eseq_root()
DSP = os.path.join(REPO, "content/instruments/Drums/VILLAIN Kick/dsp.lisp")
NK, NCOL = 33, 591

def tokenize(s):
    return re.findall(r"\(|\)|\[|\]|[^\s()\[\]]+", s)

def parse(tokens, i=0):
    out = []
    while i < len(tokens):
        t = tokens[i]
        if t in "([":
            sub, i = parse(tokens, i + 1)
            out.append(sub)
        elif t in ")]":
            return out, i + 1
        else:
            try:
                out.append(float(t))
            except ValueError:
                out.append(t)
            i += 1
    return out, i

def mix(a, b, t):
    return a + (b - a) * t

class BlockRate:
    def __init__(self, path=DSP):
        src = open(path).read()
        self.defaults = {}
        for m in re.finditer(r"\(param (\w+) @default ([-\d.e]+)", src):
            self.defaults[m.group(1)] = float(m.group(2))
        lines = src.split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith("(def kiA"))
        end = next(i for i, l in enumerate(lines) if l.startswith(";; ---- helpers"))
        self.defs = []
        for l in lines[start:end]:
            if not l.startswith("(def "):
                continue
            if l.startswith("(def KT "):
                nums = l[l.index("@data [") + 7:l.rindex("]")].split()
                flat = [float(x) for x in nums]
                assert len(flat) == NK * NCOL
                # dgen stores @shape [33 591] column-major: data[col*33 + row]
                self.KT = [[flat[c * NK + r] for c in range(NCOL)] for r in range(NK)]
                continue
            tree, _ = parse(tokenize(l))
            node = tree[0]
            self.defs.append((node[1], node[2]))
        self.names = [n for n, _ in self.defs]

    def ev(self, e, env):
        if isinstance(e, float):
            return e
        if isinstance(e, str):
            if e == "pi": return math.pi
            if e == "samplerate": return 48000.0
            return env[e]
        op, args = e[0], e[1:]
        if op == "peek":
            assert args[0] == "KT"
            return self.KT[int(self.ev(args[1], env))][int(self.ev(args[2], env))]
        v = [self.ev(a, env) for a in args]
        if op == "+": return sum(v)
        if op == "*":
            p = 1.0
            for x in v: p *= x
            return p
        if op == "-": return -v[0] if len(v) == 1 else v[0] - sum(v[1:])
        if op == "/": return v[0] / v[1]
        if op == "mix": return mix(*v)
        if op == "clip": return min(max(v[0], v[1]), v[2])
        if op == "exp": return math.exp(v[0])
        if op == "log": return math.log(v[0])
        if op == "min": return min(v)
        if op == "max": return max(v)
        if op == "abs": return abs(v[0])
        if op == "floor": return math.floor(v[0])
        if op == "sqrt": return math.sqrt(v[0])
        if op == "tanh": return math.tanh(v[0])
        if op == "gswitch": return v[1] if v[0] > 0 else v[2]
        if op == "gte": return float(v[0] >= v[1])
        if op == "gt": return float(v[0] > v[1])
        if op == "lte": return float(v[0] <= v[1])
        if op == "lt": return float(v[0] < v[1])
        raise ValueError(op)

    def eval(self, **params):
        env = dict(self.defaults)
        env.update({k: float(v) for k, v in params.items()})
        for name, expr in self.defs:
            env[name] = self.ev(expr, env)
        return env

if __name__ == "__main__":
    br = BlockRate()
    e = br.eval(kick_a=0, kick_b=0)
    print(len(br.defs), "defs")
    for i in range(24):
        print(i, *(f"{k}={e.get(f'b{i}_{k}', float('nan')):.5g}" for k in ("f", "amp", "idec", "tau", "g", "irise", "t0", "ph", "MGA", "MGB", "MSM")))
