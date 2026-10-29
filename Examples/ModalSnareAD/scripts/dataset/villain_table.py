#!/usr/bin/env python3
"""Evaluate VILLAIN Snare's block-rate section (dsp.lisp top-level defs between
the param block and the per-sample macros) in Python, exactly as the DSP
computes them, for any (snare_a, snare_b, blend, exaggerate, pc1..3, knobs).

    from villain_table import BlockRate
    br = BlockRate()            # parses dsp.lisp once
    env = br.eval(snare_a=3)    # dict of every block-rate def -> float
"""
import math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402

REPO = MAD.eseq_root()
DSP = os.path.join(REPO, "content/instruments/Drums/VILLAIN Snare/dsp.lisp")

PARAM_DEFAULTS = {}

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
        for m in re.finditer(r"\(param (\w+) @default ([-\d.e]+)", src):
            PARAM_DEFAULTS[m.group(1)] = float(m.group(2))
        lines = src.split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith("(def kiA"))
        end = next(i for i, l in enumerate(lines) if l.startswith(";; TPT state-variable"))
        self.defs = []
        for l in lines[start:end]:
            if not l.startswith("(def "):
                continue
            if l.startswith("(def ST "):
                nums = l[l.index("@data [") + 7:l.rindex("]")].split()
                flat = [float(x) for x in nums]
                # dgen stores @shape [32 546] column-major: data[col*32 + row]
                self.ST = [[flat[c * 32 + r] for c in range(546)] for r in range(32)]
                continue
            tree, _ = parse(tokenize(l))
            node = tree[0]
            self.defs.append((node[1], node[2]))
        self.names = [n for n, _ in self.defs]

    def ev(self, e, env):
        if isinstance(e, float):
            return e
        if isinstance(e, str):
            if e == "pi":
                return math.pi
            if e == "samplerate":
                return 48000.0
            return env[e]
        op, args = e[0], e[1:]
        if op == "peek":
            assert args[0] == "ST"
            return self.ST[int(self.ev(args[1], env))][int(self.ev(args[2], env))]
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
        raise ValueError(op)

    def eval(self, **params):
        env = dict(PARAM_DEFAULTS)
        env.update({k: float(v) for k, v in params.items()})
        for name, expr in self.defs:
            env[name] = self.ev(expr, env)
        return env
