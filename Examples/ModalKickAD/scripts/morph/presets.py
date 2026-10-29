#!/usr/bin/env python3
"""Pick Latent Kick Morph presets (START -> END pairs by musical category, from survey_pairs.json),
write "Latent Kick Morph.presets" and demo renders renders/<id>.wav:
[START alone | END alone | morph at the preset time | morph at an alt time], 0.25 s gaps, 1.0 s each.
A pair qualifies when its 300 ms morph never dips more than MAXC dB below both ends' envelopes and
its peak stays within 1.1x of the louder end. Each kick is used at most twice."""
import json, os, sys
from lib import *
S = json.load(open(os.path.join(HERE, "survey_singles.json")))
P = json.load(open(os.path.join(HERE, "survey_pairs.json")))
m = load(os.path.join(HERE, "Latent Kick Morph"))
MAXC = 4.0


def body_feats(n):
    y = render(m, 1.0, params={**zp(CODE[n]), **zp(CODE[n], "b")})[:, 0]
    seg = y[int(0.05 * SR):int(0.2 * SR)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); f = np.fft.rfftfreq(len(seg), 1 / SR)
    cen = float((X * f).sum() / X.sum())
    hp = np.abs(np.fft.rfft(y[:int(0.3 * SR)])); f2 = np.fft.rfftfreq(int(0.3 * SR), 1 / SR)
    air = float(np.sqrt((hp[f2 > 3000] ** 2).sum() / (hp ** 2).sum()))
    return cen, air


BF = {n: body_feats(n) for n in NAMES}
good = [r for r in P if r["collapse"] < MAXC and r["over"] < 1.1]
print(len(good), "qualifying pairs of", len(P))
L_ = np.log
CATS = [  # name, score(r), morph ms, curve, amount, follow, alt ms
    ("Pitch Dive", lambda r: L_(S[r["A"]]["sub"] / S[r["B"]]["sub"]), 150, -0.3, 1.0, 1.0, 600),
    ("Pitch Rise", lambda r: L_(S[r["B"]]["sub"] / S[r["A"]]["sub"]), 250, 0.0, 1.0, 1.0, 80),
    ("Long To Short", lambda r: L_(S[r["A"]]["t20"] / S[r["B"]]["t20"]), 250, 0.0, 1.0, 1.0, 900),
    ("Bright To Dark", lambda r: L_(BF[r["A"]][0] / BF[r["B"]][0]), 120, 0.0, 1.0, 1.0, 500),
    ("Dark To Bright", lambda r: L_(BF[r["B"]][0] / BF[r["A"]][0]), 300, 0.3, 1.0, 1.0, 100),
    ("Air Out", lambda r: L_(BF[r["A"]][1] / max(BF[r["B"]][1], 1e-4)), 100, 0.0, 1.0, 1.0, 400),
    ("Air In", lambda r: L_(BF[r["B"]][1] / max(BF[r["A"]][1], 1e-4)), 200, 0.0, 1.0, 1.0, 60),
    ("Sub Slide", lambda r: abs(L_(S[r["A"]]["sub"] / S[r["B"]]["sub"])) + 0.5 * abs(L_(S[r["A"]]["t20"] / S[r["B"]]["t20"])), 900, -0.5, 1.0, 1.0, 300),
    ("Half Way", lambda r: abs(L_(BF[r["A"]][0] / BF[r["B"]][0])) + abs(L_(S[r["A"]]["sub"] / S[r["B"]]["sub"])), 300, 0.0, 0.5, 1.0, 1200),
    ("Strike Levels", lambda r: L_(S[r["A"]]["t20"] / S[r["B"]]["t20"]) * 0 + abs(L_(BF[r["A"]][0] / BF[r["B"]][0])), 400, 0.0, 1.0, 0.3, 100),
]
used = {}
picks = []
for name, score, mt, cv, am, fo, alt in CATS:
    cands = sorted(good, key=score, reverse=True)
    for r in cands:
        if used.get(r["A"], 0) >= 2 or used.get(r["B"], 0) >= 2 or any({r["A"], r["B"]} == {p[1], p[2]} for p in picks):
            continue
        picks.append((name, r["A"], r["B"], mt, cv, am, fo, alt, score(r), r))
        used[r["A"]] = used.get(r["A"], 0) + 1; used[r["B"]] = used.get(r["B"], 0) + 1
        break
os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
gap = np.zeros(int(0.25 * SR)); pres = []; summary = {}
for name, A, B, mt, cv, am, fo, alt, sc, r in picks:
    pid = "latent-kick-morph-" + name.lower().replace(" ", "-")
    base = {**zp(CODE[A]), **zp(CODE[B], "b"), "curve": cv, "amount": am, "follow": fo, "tune": 1.0, "decay": 1.0, "drive": 1.0, "level": 1.0}
    pres.append({"id": pid, "name": name, "base_note_offset": 0,
                 "params": {k: round(v, 5) if isinstance(v, float) else v for k, v in {**base, "morph": float(mt)}.items()}})
    rr = lambda p: render(m, 1.0, params=p)[:, 0]
    ya = rr({**base, **zp(CODE[A], "b")}); yb = rr({**base, **zp(CODE[B]), **zp(CODE[B], "b")})
    y1 = rr({**base, "morph": float(mt)}); y2 = rr({**base, "morph": float(alt)})
    write_wav_f32(os.path.join(HERE, "renders", pid[len("latent-kick-morph-"):] + ".wav"), np.concatenate([ya, gap, yb, gap, y1, gap, y2]))
    summary[name] = dict(A=A, B=B, morph=mt, alt=alt, curve=cv, amount=am, follow=fo, score=round(float(sc), 3),
                         collapse_300ms=round(r["collapse"], 2), peaks=[round(float(np.abs(x).max()), 3) for x in (ya, yb, y1, y2)],
                         sub=[round(S[A]["sub"], 1), round(S[B]["sub"], 1)], t20=[S[A]["t20"], S[B]["t20"]],
                         body_centroid=[round(BF[A][0]), round(BF[B][0])])
    print(name, summary[name], flush=True)
json.dump({"version": 1, "engine_name": "Research/Latent Kick Morph",
           "source_file": "instruments/Research/Latent Kick Morph/dsp.lisp", "presets": pres},
          open(os.path.join(HERE, "Latent Kick Morph.presets"), "w"), indent=2)
json.dump(summary, open(os.path.join(HERE, "renders", "summary.json"), "w"), indent=1)
