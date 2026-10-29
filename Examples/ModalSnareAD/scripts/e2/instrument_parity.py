"""Close the loop: compiled instrument (this checkout's DGenLisp) vs the Swift trainer's render of
the same z. Head: instrument head tap (LATENT_HEADTAP=1 build) vs trainer head, exact
(rel RMS). Full output: noise realisations differ (instrument: live xorshift + biquads;
trainer: E1's fixed band table), so band-envelope / MR metrics, plus both vs VILLAIN.
usage: instrument_parity.py RUN INST_DIR -> INST_DIR/parity.json, renders in e2/renders/parity"""
import json, os, shutil, subprocess, sys
import numpy as np
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_ad as MAD  # noqa: E402
HERE = MAD.stage("e2"); R = MAD.ROOT
MAD.use("e2"); MAD.use("e1"); MAD.use("dataset")
import template as tp
tp.env_setup()
from audition import Instrument
from metric import score, band_score
from villain_common import read_wav_f32, write_wav_f32
run, inst_dir = sys.argv[1], sys.argv[2]
codes = json.load(open(os.path.join(inst_dir, "codes_std.json")))
ck = json.load(open(os.path.join(run, "codes.json")))
names = codes["order"]; D = codes["D"]
pick = [0, 5, 11, 12, 19, 20, 26, 29]   # SP01 SP06 SP12 SP13 SP20 XCF ZDSZS ZSFG
out = os.path.join(HERE, "renders", "parity"); os.makedirs(out, exist_ok=True)
# head-tap build
tap = inst_dir.rstrip("/") + " headtap"
if os.path.exists(tap): shutil.rmtree(tap)
shutil.copytree(inst_dir, tap)
subprocess.run([sys.executable, os.path.join(MAD.code("e3proto"), "gen_dsp.py"), "--out", tap], check=True,
               env=dict(os.environ, LATENT_D=str(D), LATENT_HEADTAP="1"), stdout=subprocess.DEVNULL)
inst = Instrument(tap)
spec = {"names": [], "vel": [], "z": []}
for s in pick:
    for v in (1.0, 0.7, 0.4):
        spec["names"].append(f"{names[s]}_v{int(v*100):03d}"); spec["vel"].append(v); spec["z"].append(ck["codes"][names[s]])
sp = os.path.join(out, "spec.json"); json.dump(spec, open(sp, "w"))
B = MAD.binary("ModalSnareAD")
subprocess.run([B, "latent-render", "--ckpt", os.path.join(run, "checkpoint.json"), "--spec", sp, "--out", out + "/"], check=True)
rows = []
for n, v in zip(spec["names"], spec["vel"]):
    s = names.index(n.rsplit("_v", 1)[0])
    zs = codes["codes"][names[s]]
    y, _ = inst.render(1.0, pitch=261.6256, vel=v, params={**{f"z{i+1}": zs[i] for i in range(D)}, "level": 1.0})
    yo, yh = y[:, 0].astype(float), y[:, 1].astype(float) if y.ndim > 1 and y.shape[1] > 1 else None
    if y.ndim > 1 and y.shape[1] > 2: yh = y[:, 2].astype(float)
    sw = read_wav_f32(os.path.join(out, n + ".swift.wav")).astype(float)
    swh = read_wav_f32(os.path.join(out, n + ".swift_head.wav")).astype(float)
    vid = f"snare_{s:02d}_v{int(round(v*100)):03d}"
    t = read_wav_f32(os.path.join(R, "data/villain", vid + ".wav")).astype(float)
    rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))
    r = {"id": n, "head_rel_rms": rel(yh, swh) if yh is not None else None,
         "out_band_inst_vs_swift": band_score(yo, sw), "out_mr_inst_vs_swift": score(yo, sw),
         "inst_vs_villain_mr": score(yo, t), "swift_vs_villain_mr": score(sw, t),
         "inst_vs_villain_band": band_score(yo, t), "swift_vs_villain_band": band_score(sw, t),
         "peak_inst": float(np.abs(yo).max()), "peak_swift": float(np.abs(sw).max()), "peak_villain": float(np.abs(t).max())}
    rows.append(r)
    write_wav_f32(os.path.join(out, n + ".instrument.wav"), yo); write_wav_f32(os.path.join(out, n + ".villain_target.wav"), t)
    print(f"{n:16s} head rel {r['head_rel_rms']:.2e}  out inst~swift band {r['out_band_inst_vs_swift']:.2f} dB MR {r['out_mr_inst_vs_swift']:.3f} | vs VILLAIN MR inst {r['inst_vs_villain_mr']:.3f} swift {r['swift_vs_villain_mr']:.3f} band {r['inst_vs_villain_band']:.2f}/{r['swift_vs_villain_band']:.2f} | peak {r['peak_inst']:.2f}/{r['peak_swift']:.2f}/{r['peak_villain']:.2f}")
json.dump(rows, open(os.path.join(inst_dir, "parity.json"), "w"), indent=1)
print("max head rel", max(r["head_rel_rms"] for r in rows), "median out band inst~swift", np.median([r["out_band_inst_vs_swift"] for r in rows]))
