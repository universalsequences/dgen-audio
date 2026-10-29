"""E7: compiled instrument (headtap variant: out1 audio, out3 head, out4 wire-out) vs numpy twin
(weights_wires, live-xorshift noise). Wires are chaotic: sample-exact only early, judge by band."""
import json, os
from e7lib import *
import template as tp
from audition import Instrument
cs = json.load(open(os.path.join(HERE, "Latent Snare", "codes_std.json")))
inst = Instrument(os.path.join(HERE, "variants", "Latent Snare headtap"))
import eval_distill as ed
Ww = ed.loadw(os.path.join(HERE, "weights_wires.json"))
XB = tp.bandpass_bank(tp.xorshift_white(SR))
rows = {}
for nm in ("SP01", "SP12", "ZDSZS", "ZV", "SP06", "SP11", "SP13"):
    for v in (1.0, 0.4):
        zk = cs["codes"][nm]
        y, _ = inst.render(1.0, vel=v, params={"z1": zk[0], "z2": zk[1], "z3": zk[2], "z4": zk[3], "tune": 0.0, "decay": 1.0, "level": 1.0})
        a, hd, wt = y[:, 0].astype(np.float64), y[:, 1].astype(np.float64), y[:, 2].astype(np.float64)
        p = decode(code(nm), v, Ww); head, noise, chain, lin = parts(p, bands=XB)
        kn = {k: float(np.ravel(p[k])[0]) for k in WIRE}; wo = wire_out(head, kn); tw = chain(head + noise + wo)
        rel = lambda x, t: float(np.sqrt(E(x - t) / max(E(t), 1e-30)))
        first = np.argmax(np.abs(wt - wo) > 1e-4 * max(np.abs(wo).max(), 1e-9)) if np.any(np.abs(wt - wo) > 1e-4 * max(np.abs(wo).max(), 1e-9)) else SR
        r = dict(head_rel=rel(hd, head), wire_first_diverge=int(first), wire_band=band_score(wt + 1e-12, wo + 1e-12) if E(wo) > 1e-12 else 0.0,
                 wire_E_ratio=E(wt) / max(E(wo), 1e-30), out_band=band_score(a, tw), out_rel=rel(a, tw))
        rows[f"{nm}_v{v}"] = r
        print(nm, v, {k: round(x, 5) if isinstance(x, float) else x for k, x in r.items()}, flush=True)
json.dump(rows, open(os.path.join(HERE, "parity.json"), "w"), indent=1)
