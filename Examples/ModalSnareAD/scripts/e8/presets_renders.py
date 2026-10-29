"""Write Latent Snare Morph.presets (12 A->B morphs) and demo renders e8/renders/<id>.wav:
[A alone | B alone | morph at the preset time | morph at a second time], 0.25 s gaps, each 1.2 s."""
import json, os, sys
from lib8 import *
MAD.use("dataset")
from villain_common import write_wav_f32
P = [  # name, A, B, morph ms, curve, amount, follow, alt ms
    ("Drop Ring", "SP13", "SP04", 600, 0.0, 1.0, 1.0, 150),
    ("Ring Bend Up", "XCF", "SP07", 500, 0.0, 1.0, 1.0, 120),
    ("Buzz Out", "SP01", "YYUUU", 250, 0.0, 1.0, 1.0, 1000),
    ("Buzz In", "XDXDFGHS", "SP12", 300, -0.3, 1.0, 1.0, 80),
    ("Bright To Dark", "SP20", "XCF", 150, 0.0, 1.0, 1.0, 600),
    ("Long To Short", "SP03", "SP11", 1200, 0.0, 1.0, 1.0, 300),
    ("Pitch Dive", "SP07", "SP04", 80, -0.4, 1.0, 1.0, 300),
    ("Tom Slide", "SP03", "SP19", 900, -0.5, 1.0, 1.0, 250),
    ("Ring To Wires", "SP13", "ZDSZS", 400, 0.0, 1.0, 1.0, 1500),
    ("Wire Swell", "ZFG", "SP01", 150, 0.0, 1.0, 1.0, 500),
    ("Half Way", "SP06", "SP02", 300, 0.0, 0.5, 1.0, 1000),
    ("Ring Only", "SP01", "ZV", 400, 0.0, 1.0, 0.3, 100),
]
inst = Instrument(os.path.join(HERE, "Latent Snare Morph"))
gap = np.zeros(int(0.25 * SR)); pres = []; summary = {}
def r(p): return render(inst, 1.2, params={**p, "level": 0.8})[:, 0] if False else render(inst, 1.2, params=p)
for name, A, B, mt, cv, am, fo, alt in P:
    pid = "latent-morph-" + name.lower().replace(" ", "-")
    base = {**zp(CODES[A]), **zp(CODES[B], "b"), "curve": cv, "amount": am, "follow": fo, "tune": 0.0, "decay": 1.0, "level": 0.8}
    pres.append({"id": pid, "name": name, "base_note_offset": 0,
                 "params": {k: round(v, 4) if isinstance(v, float) else v for k, v in {**base, "morph": float(mt)}.items()}})
    ya = r({**base, **zp(CODES[A], "b")}); yb = r({**base, **zp(CODES[B]), **zp(CODES[B], "b")})
    y1 = r({**base, "morph": float(mt)}); y2 = r({**base, "morph": float(alt)})
    write_wav_f32(os.path.join(HERE, "renders", pid[13:] + ".wav"), np.concatenate([ya, gap, yb, gap, y1, gap, y2]).astype(np.float32))
    summary[name] = dict(A=A, B=B, morph=mt, alt=alt, peaks=[round(float(np.abs(x).max()), 3) for x in (ya, yb, y1, y2)],
                         tail_rms=[round(float(np.sqrt(np.mean(x[24000:] ** 2))), 4) for x in (ya, yb, y1, y2)])
    print(name, summary[name], flush=True)
json.dump({"version": 1, "engine_name": "Research/Latent Snare Morph",
           "source_file": "instruments/Research/Latent Snare Morph/dsp.lisp", "presets": pres},
          open(os.path.join(HERE, "Latent Snare Morph.presets"), "w"), indent=2)
json.dump(summary, open(os.path.join(HERE, "renders", "summary.json"), "w"), indent=1)
