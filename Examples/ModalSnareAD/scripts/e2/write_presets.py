"""One preset per VILLAIN snare: its standardised z code as knob values (eseq .presets format,
like content/instruments/Drums/VILLAIN Snare.presets). usage: write_presets.py INST_DIR"""
import json, os, sys
inst = sys.argv[1]; c = json.load(open(os.path.join(inst, "codes_std.json")))
rel = json.load(open(os.path.join(inst, "instrument.json")))["releases"]["1"]["name"]
pre = []
for i, n in enumerate(c["order"]):
    p = {f"z{k+1}": round(float(v), 4) for k, v in enumerate(c["codes"][n])}
    p.update({"tune": 0, "decay": 1, "level": 1.0})
    pre.append({"id": f"latent-snare-{n.lower()}", "name": f"Latent {n}", "base_note_offset": 0, "params": p})
doc = {"version": 1, "engine_name": rel, "source_file": f"instruments/{rel}/dsp.lisp", "presets": pre}
dst = inst.rstrip("/") + ".presets"; json.dump(doc, open(dst, "w"), indent=2)
print("wrote", dst, len(pre), "presets")
