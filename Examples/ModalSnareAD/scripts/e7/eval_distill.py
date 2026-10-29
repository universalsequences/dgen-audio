"""E7 steps 3+5: the distilled decoders (twin, E1 band-noise table) on all 32 snares x 3 velocities
vs VILLAIN out, and the strand contribution at v0.4/0.7 vs VILLAIN's (out - out(rattle=0)).
-> eval_distill.json"""
import json, sys
from multiprocessing import Pool
from e7lib import *
os.chdir(HERE)  # writes go to $MODAL_AD_ROOT/e7
import latent_np as ln_
from villain_common import load, render as vrender


def loadw(fn):
    w = json.load(open(fn)); a = ln_.arr
    return {**{f"{k}.W": a(w[k]["W"]) for k in ("l1", "l2", "l3")}, **{f"{k}.b": a(w[k]["b"]) for k in ("l1", "l2", "l3")}}


WS = {"E2 base": (W, False), "weights_wires": (loadw("weights_wires.json"), True),
      "weights_wires_rebal": (loadw("weights_wires_rebal.json"), True)}
FIT = json.load(open("fit_wires.json"))
_V = {}


def villain_d(rid):
    """VILLAIN strand contribution at any velocity: out - out(rattle=0)."""
    if "i" not in _V: _V["i"] = load(stems=True)
    it = MAN[rid]; p = it["params"]
    a = vrender(_V["i"], p, vel=it["vel"])["out"]; b = vrender(_V["i"], {**p, "rattle": 0.0}, vel=it["vel"])["out"]
    return np.asarray(a, np.float64) - np.asarray(b, np.float64)


def one(args):
    nm, v = args; rid = rid_of(nm, v); row = {"name": nm, "vel": v, "id": rid}
    d = villain_d(rid); o = v_out(rid); row["villain_strand_share"] = E(d) / E(o)
    for tag, (Wt, wired) in WS.items():
        p = decode(code(nm), v, Wt); head, noise, chain, lin = parts(p)
        kn = {k: float(np.ravel(p[k])[0]) for k in WIRE}
        wo = wire_out(head, kn) if wired else 0 * head
        y = chain(head + noise + wo); s = scores(y, rid)
        if wired:
            dw = y - chain(head + noise)
            s.update(model_strand_share=E(dw) / E(y), strand_band=band_score(dw + 1e-12, d), knobs=kn)
        s["peak"] = float(np.abs(y).max()); row[tag] = s
    # per-snare fitted knobs (not distilled) for reference at v1.0
    if v == 1.0:
        p = decode(code(nm), v); head, noise, chain, lin = parts(p)
        row["per_snare_knobs"] = scores(chain(head + noise + wire_out(head, FIT[nm]["knobs"])), rid)
    row["villain_peak"] = float(np.abs(o).max())
    return row


if __name__ == "__main__":
    jobs = [(nm, v) for nm in NAMES for v in (0.4, 0.7, 1.0)]
    with Pool(8) as pool: rows = pool.map(one, jobs)
    json.dump(rows, open("eval_distill.json", "w"), indent=1)
    heavy = [nm for nm in NAMES if FIT[nm]["strand_share"] > 0.15]
    for sub, lab in ((NAMES, "all 32"), (heavy, "strand-heavy " + ",".join(heavy))):
        print(f"== {lab}")
        for v in (0.4, 0.7, 1.0):
            rs = [x for x in rows if x["name"] in sub and x["vel"] == v]
            line = f"  v{v}: "
            for tag in WS:
                line += f"{tag}: band {np.median([x[tag]['band'] for x in rs]):.3f} (n {np.median([x[tag]['band_norm'] for x in rs]):.3f}) MR {np.median([x[tag]['mr'] for x in rs]):.3f} (n {np.median([x[tag]['mr_norm'] for x in rs]):.3f}) | "
            print(line)
    print("== velocity: strand share model vs VILLAIN, strand-heavy")
    for nm in heavy:
        for v in (0.4, 0.7, 1.0):
            x = next(r for r in rows if r["name"] == nm and r["vel"] == v); a = x["weights_wires"]; b = x["weights_wires_rebal"]
            print(f"  {nm:6s} v{v}: VILLAIN share {x['villain_strand_share']:.3f} model {a['model_strand_share']:.3f} strand-band {a['strand_band']:.2f}"
                  f" | out band base {x['E2 base']['band']:.3f} wires {a['band']:.3f} rebal {b['band']:.3f} | MR {x['E2 base']['mr']:.3f} {a['mr']:.3f} {b['mr']:.3f}"
                  f" | peak {x['E2 base']['peak']:.2f} {a['peak']:.2f} {b['peak']:.2f} (VILLAIN {x['villain_peak']:.2f})")
