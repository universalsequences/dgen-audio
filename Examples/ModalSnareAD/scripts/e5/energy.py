"""Strand energy accounting on VILLAIN: out vs out(rattle=0), all 32 fits at vel 1.
Writes $MODAL_AD_ROOT/e5/energy.json and the E7 wire targets e5/targets/<id>.{strand_out,out_r0}.wav."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
os.chdir(MAD.stage("e5"))
MAD.use("dataset")
from villain_common import load, render, read_wav_f32, write_wav_f32
D = "../data/villain"
man = {it["id"]: it for it in json.load(open(f"{D}/manifest.json"))["items"]}
modes = json.load(open("../data/villain_modes.json"))["renders"]
inst = load(stems=True)
os.makedirs("targets", exist_ok=True)
E = lambda x: float(np.sum(np.asarray(x, np.float64) ** 2))
rows = []
for k in range(32):
    rid = f"snare_{k:02d}_v100"
    p = man[rid]["params"]
    r = render(inst, p); r0 = render(inst, {**p, "rattle": 0.0})
    out = read_wav_f32(f"{D}/{rid}.wav")
    assert np.abs(out - r["out"]).max() < 1e-6
    d = r["out"] - r0["out"]
    write_wav_f32(f"targets/{rid}.out_r0.wav", r0["out"])
    write_wav_f32(f"targets/{rid}.strand_out.wav", d)
    st = read_wav_f32(f"{D}/stems/{rid}.strands.wav")
    ho = read_wav_f32(f"{D}/stems/{rid}.out_headonly.wav"); no = read_wav_f32(f"{D}/stems/{rid}.out_noiseonly.wav")
    c = modes[rid]["chain"]
    rows.append(dict(id=rid, name=man[rid]["name"], strand_out_frac=E(d) / E(out),
                     resid_frac=E(out - ho - no) / E(out), strands_stem_rel_source=E(st) / E(r["source"]),
                     c_rattle=c["c_rattle"], c_zs=c["c_zs"], c_gap=c["c_gap"], c_wire_hz=c["c_wire_hz"],
                     c_wire_ms=c["c_wire_ms"], c_sthi_on=c["c_sthi_on"], c_sthi=c["c_sthi"],
                     c_buzz_lo=c["c_buzz_lo"], c_buzz_hi=c["c_buzz_hi"], c_buzz_k=c["c_buzz_k"], c_buzz_steep=c["c_buzz_steep"],
                     c_buzz_gd=c["c_buzz_gd"], c_st_head=c["c_st_head"], strokes=sum(c[f"c_s{i}"] for i in range(1,5))))
json.dump(rows, open("energy.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: -r["strand_out_frac"]):
    print(f"{r['id']} {r['name']:6s} strand/out {r['strand_out_frac']:.3f} resid {r['resid_frac']:.3f} stem/src {r['strands_stem_rel_source']:.3f} "
          f"rat {r['c_rattle']:.2f} zs {r['c_zs']:.2f} gap {r['c_gap']:.3f} hz {r['c_wire_hz']:.0f} ms {r['c_wire_ms']:.2f} sthi {r['c_sthi_on']:.0f}/{r['c_sthi']:.0f} "
          f"bp {r['c_buzz_lo']:.0f}-{r['c_buzz_hi']:.0f} k{r['c_buzz_k']:.2f} st{r['c_buzz_steep']:.1f} strokes {r['strokes']:.2f}")
