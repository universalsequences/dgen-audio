"""E7 step 1+2: fit the 5 decoded W2 wire knobs on the DECODER's own head (v1.0) against
e5/targets/<id>.strand_out.wav through the latent's (linearised) output chain; then score
(A) E5-style through VILLAIN's real chain (villain_inject) and (B) the full latent output,
with and without band_couple. -> fit_wires.json"""
import json, math, os, sys, time
from multiprocessing import Pool
from e7lib import *
os.chdir(HERE)  # writes go to $MODAL_AD_ROOT/e7
SPACE = {k: (math.log(SEG[k]["lo"]), math.log(SEG[k]["hi"])) for k in WIRE}


def clampu(k, u):
    lo, hi = SPACE[k]
    return min(max(u, lo), hi)


def pattern_search(evalf, init, free, step0=0.7, tol=0.02, max_evals=400):
    kn = {k: math.log(v) for k, v in init.items()}; cache = {}

    def ev(kk):
        key = tuple(round(kk[k], 5) for k in sorted(kk))
        if key not in cache:
            cache[key] = evalf({k: math.exp(v) for k, v in kk.items()})
        return cache[key]
    best = ev(kn)[0]; step = {k: step0 for k in free}
    while len(cache) < max_evals and max(step.values()) > tol:
        imp = False
        for k in free:
            for sg in (1, -1):
                tr = dict(kn); tr[k] = clampu(k, kn[k] + sg * step[k])
                if tr[k] == kn[k]:
                    continue
                l = ev(tr)[0]
                if l < best - 1e-4:
                    best, kn, imp = l, tr, True
                    break
        if not imp:
            for k in step:
                step[k] *= 0.5
    l, extra = ev(kn)
    return {k: math.exp(v) for k, v in kn.items()}, l, extra, len(cache)


def head_scale(h, hv, n=int(0.2 * SR)):
    return math.sqrt(E(hv[:n]) / max(E(h[:n]), 1e-30))


def inside(kk):
    return {k: min(max(v, math.exp(SPACE[k][0]) * 1.0001), math.exp(SPACE[k][1]) * 0.9999) for k, v in kk.items()}


OFF = dict(wire_thr=0.042, wire_level=1e-3, wire_lo=1300.0, wire_hi=7000.0, wire_k=1.3)


def fit_knobs(head, lin, d, c, s):
    """twin search of thr/lo/hi/k, level = energy match (clipped to its range)."""
    lvlo, lvhi = SEG["wire_level"]["lo"], SEG["wire_level"]["hi"]

    def evB(kk):
        y = lin * wire_out(head, dict(kk, wire_level=1.0))
        g = math.sqrt(E(d) / E(y)) if E(y) > 0 else lvhi
        lvl = min(max(g, lvlo), lvhi); y = y * lvl
        return band_score(y + 1e-12, d) + mr_score(y, d), lvl
    vthr = (c["c_gap"] / c["c_zs"]) / s  # VILLAIN contact threshold mapped to latent head units
    inits = [dict(wire_thr=vthr, wire_lo=c["c_buzz_lo"], wire_hi=c["c_buzz_hi"], wire_k=c["c_buzz_k"]),
             dict(wire_thr=0.042, wire_lo=1300.0, wire_hi=7000.0, wire_k=1.3)]
    best = None; nev = 0
    for ini in inits:
        kk, l, lvl, n = pattern_search(evB, inside(ini), ["wire_thr", "wire_lo", "wire_hi", "wire_k"]); nev += n
        if best is None or l < best[1]:
            best = (dict(kk, wire_level=lvl), l)
    return best[0], best[1], nev, vthr


def rebalance(p, head, wire, couple, chain, rid, max_evals=260):
    """per-band multipliers m_b on band_amp, searched on the full-output score (band + MR vs
    VILLAIN out): lets the noise layer give back the strand energy E2 training put into it."""
    Nb, Nc = noise_basis(p); Nc = Nc if couple else 0.0
    o = v_out(rid); base = head + wire + Nc
    def ev(lm):
        y = chain(base + Nb @ np.exp(lm)); return band_score(y, o) + mr_score(y, o)
    lm = np.zeros(16); best = ev(lm); step = 0.6; n = 1
    while step > 0.04 and n < max_evals:
        imp = False
        for b in range(16):
            for sg in (1, -1):
                tr = lm.copy(); tr[b] = np.clip(tr[b] + sg * step, -4, 1.5); n += 1
                l = ev(tr)
                if l < best - 1e-4: best, lm, imp = l, tr, True; break
        if not imp: step *= 0.5
    m = np.exp(lm)
    return m, chain(base + Nb @ m)


def fit_one(k):
    from inject import render_inject
    t0 = time.time(); rid = f"snare_{k:02d}_v100"; name = MAN[rid]["name"]; prm = MAN[rid]["params"]
    p = decode(code(name), 1.0); head, noise, chain, lin = parts(p)
    noise_nc = parts(p, couple=False)[1]
    d = strand_out_target(rid); o = v_out(rid); share = E(d) / E(o)
    c = MODES[rid]["chain"]; hv = head_modes(rid); s = head_scale(head, hv)
    row = {"id": rid, "name": name, "strand_share": share, "head_scale_villain_over_latent": s}
    if share < 1e-4:
        kn, lossB, nev, vthr = dict(OFF), None, 0, None
    else:
        kn, lossB, nev, vthr = fit_knobs(head, lin, d, c, s)
    row["knobs"] = kn; row["fit_loss"] = lossB; row["fit_evals"] = nev; row["fit_s"] = time.time() - t0
    row["villain_ref"] = dict(thr_latent_units=vthr, buzz_lo=c["c_buzz_lo"], buzz_hi=c["c_buzz_hi"], buzz_k=c["c_buzz_k"], rattle=c["c_rattle"])
    yB = lin * wire_out(head, kn)
    row["strand_out_fit"] = dict(band=band_score(yB + 1e-12, d), mr=mr_score(yB, d), eratio=E(yB) / max(E(d), 1e-30))
    # ---- (B) full latent output, 4 variants
    full = {}
    for tag, kk, nz in (("base", None, noise), ("wires", kn, noise), ("wires_nocouple", kn, noise_nc), ("nocouple", None, noise_nc)):
        full[tag] = scores(chain(head + nz + (wire_out(head, kk) if kk else 0)), rid)
    wo = wire_out(head, kn); row["rebal"] = {}
    for tag, cp, wv in (("rebal_nowires", True, 0 * wo), ("wires_rebal", True, wo), ("wires_nocouple_rebal", False, wo)):
        m, y = rebalance(p, head, wv, cp, chain, rid); full[tag] = scores(y, rid); row["rebal"][tag] = m.tolist()
    fl = dict(mr=CAL[rid]["mr_floor"], band=CAL[rid]["band_floor"])
    for tag in [t for t in full if t != "base"]:
        for m in ("mr", "band"):
            full[tag][f"gap_closed_{m}"] = (full["base"][m] - full[tag][m]) / max(full["base"][m] - fl[m], 1e-9)
    row["full"] = full
    # ---- (A) E5-style: wires on the decoder head, injected into VILLAIN's real chain
    tgt = stem(rid, "strands"); r0 = out_r0(rid)
    S0 = dict(band=band_score(r0, o), mr=mr_score(r0, o))
    rng = np.random.default_rng(100 + k); v = villain_wire_params(rid)
    yf = render_inject(prm, wires(hv * (1 + 1e-6 * rng.standard_normal(SR)), **v).astype(np.float64))
    flA = dict(band=band_score(yf, o), mr=mr_score(yf, o))
    inj = {}
    if share >= 1e-4:
        def evA(kk):
            y = wire_raw(head, kk["wire_thr"], 1.0); g = math.sqrt(E(tgt) / E(y)) if E(y) > 0 else 0.0
            y = y * g
            return band_score(y + 1e-12, tgt) + mr_score(y, tgt), g
        best = None
        for ini in (dict(wire_thr=kn["wire_thr"]), dict(wire_thr=vthr)):
            kk, l, g, _ = pattern_search(evA, inside(ini), ["wire_thr"])
            if best is None or l < best[1]:
                best = (kk, l, g)
        cands = {"W2_decoder_head": wire_raw(head, best[0]["wire_thr"], 1.0) * best[2],
                 "W2_decoder_head_Bthr": wire_raw(head, kn["wire_thr"], 1.0) * evA(dict(wire_thr=kn["wire_thr"]))[1],
                 "villain_stage_decoder_head": wires(head * s, **v).astype(np.float64)}
        row["stem_fit_thr"] = best[0]["wire_thr"]
        for tag, x in cands.items():
            y = render_inject(prm, x); sc = dict(band=band_score(y, o), mr=mr_score(y, o))
            for m in ("band", "mr"):
                sc[f"gap_closed_{m}"] = (S0[m] - sc[m]) / max(S0[m] - flA[m], 1e-9)
            inj[tag] = sc
    row["inject"] = dict(out_r0=S0, floor=flA, **inj); row["t_total"] = time.time() - t0
    f = full
    print(f"{rid} {name:7s} share {share:.3f} fit {row['fit_s']:.1f}s ({nev} ev) " +
          " ".join(f"{a[5:]}={b:.4g}" for a, b in kn.items()) +
          f" | full band {f['base']['band']:.3f}->{f['wires']['band']:.3f} (nc {f['wires_nocouple']['band']:.3f})"
          f" MR {f['base']['mr']:.3f}->{f['wires']['mr']:.3f} (nc {f['wires_nocouple']['mr']:.3f})"
          f" | rebal band nw {f['rebal_nowires']['band']:.3f} w {f['wires_rebal']['band']:.3f} wnc {f['wires_nocouple_rebal']['band']:.3f} MR {f['rebal_nowires']['mr']:.3f} {f['wires_rebal']['mr']:.3f} {f['wires_nocouple_rebal']['mr']:.3f}" +
          ("" if not inj else f" | inj gap band W2 {inj['W2_decoder_head']['gap_closed_band']:.2f} vil {inj['villain_stage_decoder_head']['gap_closed_band']:.2f}"),
          flush=True)
    return row


if __name__ == "__main__":
    ks = [int(a) for a in sys.argv[1:]] or list(range(32))
    os.makedirs("rows", exist_ok=True)
    todo = [k for k in ks if not os.path.exists(f"rows/{k:02d}.json")]
    with Pool(min(8, max(1, len(todo)))) as pool:
        for r in pool.imap_unordered(fit_one, todo):
            json.dump(r, open(f"rows/{int(r['id'][6:8]):02d}.json", "w"), indent=1)
    rows = [json.load(open(f"rows/{k:02d}.json")) for k in ks]
    json.dump({r["name"]: r for r in rows}, open("fit_wires.json" if len(ks) == 32 else "fit_wires_partial.json", "w"), indent=1)
