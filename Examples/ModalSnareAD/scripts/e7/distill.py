"""E7 step 3: distil the per-snare W2 knobs into the decoder.
 - wire_targets.json: per-snare natural (squashed-output) knob values, velocity-invariant
   (E5: v100 knobs hold at v40/v70) -> supervised targets for E6.
 - weights_wires.json: E2 outbox_d4 weights (trainer space, weights.json format) with ONLY
   W3/b3 columns 355-359 refit by ridge least squares on the last hidden layer (32 codes x 3 vel).
 - weights_wires_rebal.json (optional extra): additionally refits the 16 band_amp columns to the
   noise rebalance found in fit_wires.py (wires_rebal), as a CPU stand-in for E6's fine-tune.
Ridge lambda by leave-one-snare-out CV; sanity on 20 random prior z's."""
import json, math, os
import numpy as np
from e7lib import *
os.chdir(HERE)  # writes go to $MODAL_AD_ROOT/e7
r = json.load(open("fit_wires.json"))
VELS = (0.4, 0.7, 1.0)
OFF = {s["name"]: s["offset"] for s in tp.LAY["segments"]}
WCOLS = [OFF[k] for k in WIRE]; assert WCOLS == list(range(355, 360)), WCOLS
BCOLS = list(range(OFF["band_amp"], OFF["band_amp"] + 16))


def h2(Wt, z, vel):
    x = np.array([[*z, min(max(vel, 0.0), 1.0)]])
    h = np.tanh(x @ Wt["l1.W"] + Wt["l1.b"]); return np.tanh(h @ Wt["l2.W"] + Wt["l2.b"])[0]


def logit(p): p = np.clip(p, 0.01, 0.99); return np.log(p / (1 - p))


# ---- targets
targets = {nm: {k: float(r[nm]["knobs"][k]) for k in WIRE} for nm in NAMES}
# gate: a snare whose per-snare wires make the full latent output worse on BOTH metrics (band
# > +2%, MR not better) gets the off knobs (wire_level at its floor). Only SP13 trips it: its
# decoder head is the long 1.65 kHz ring and E2's noise layer already sits near the floor.
OFFK = dict(wire_thr=0.042, wire_level=1e-3, wire_lo=1300.0, wire_hi=7000.0, wire_k=1.3)
GATED = {}
for nm in NAMES:
    f = r[nm]["full"]
    if f["wires"]["band"] > 1.02 * f["base"]["band"] and f["wires"]["mr"] >= f["base"]["mr"] and r[nm]["strand_share"] > 0.05:
        GATED[nm] = dict(fitted=targets[nm], base=f["base"], wires=f["wires"]); targets[nm] = dict(OFFK)
print("gated off:", list(GATED))
json.dump(targets, open("wire_targets.json", "w"), indent=1)
json.dump({"doc": "wire_targets.json: natural (post-squash) values of decoder outputs 355-359 per VILLAIN snare; "
           "velocity-invariant (use the same values at v0.4/0.7/1.0). Fit by twin pattern search on the E2 "
           "outbox_d4 decoder's own head at v1.0 vs e5/targets/<id>.strand_out.wav; wire_drive/hz/t60 fixed 1.2/620/1.0, "
           "2 strands det [0.951,1.062]. Positions p = (ln x - ln lo)/(ln hi - ln lo); trainer loss suggestion: "
           "squared error in p (or logit p), weight ~ strand share.",
           "ranges": {k: [SEG[k]["lo"], SEG[k]["hi"]] for k in WIRE},
           "positions": {nm: {k: pos(k, v) for k, v in t.items()} for nm, t in targets.items()},
           "strand_share": {nm: r[nm]["strand_share"] for nm in NAMES}, "gated_off": GATED,
           "band_amp_rebal_log_mult": {nm: np.log(r[nm]["rebal"]["wires_rebal" if nm not in GATED else "rebal_nowires"]).tolist() for nm in NAMES}},
          open("wire_targets.meta.json", "w"), indent=1)

H = np.array([h2(W, code(nm), v) for nm in NAMES for v in VELS])           # [96,128]
grp = np.array([i for i in range(32) for _ in VELS])
U_w = np.array([[logit(pos(k, targets[nm][k])) for k in WIRE] for nm in NAMES for v in VELS])
U_now = np.array([decode_u_ for decode_u_ in (tp.decode_u(W, code(nm), v) for nm in NAMES for v in VELS)])
dpos_b = np.array([np.log(r[nm]["rebal"]["wires_rebal" if nm not in GATED else "rebal_nowires"]) / (math.log(SEG["band_amp"]["hi"]) - math.log(SEG["band_amp"]["lo"]))
                   for nm in NAMES for v in VELS])
P_now_b = 1 / (1 + np.exp(-U_now[:, BCOLS]))
U_b = logit(P_now_b + dpos_b)


def ridge(Hm, U, lam, prior_W=None, prior_b=None):
    """min ||[H 1][W;b] - U||^2 + lam ||W - prior_W||^2 (bias unpenalised)."""
    mu = Hm.mean(0); Hc = Hm - mu; Um = U.mean(0)
    pW = np.zeros((Hm.shape[1], U.shape[1])) if prior_W is None else prior_W
    A = Hc.T @ Hc + lam * np.eye(Hm.shape[1])
    Wc = np.linalg.solve(A, Hc.T @ (U - Um) + lam * pW)
    return Wc, Um - mu @ Wc


def cv(U, lams, prior_W=None):
    out = {}
    for lam in lams:
        err = []
        for g in range(32):
            tr = grp != g
            Wc, b = ridge(H[tr], U[tr], lam, prior_W)
            P = 1 / (1 + np.exp(-(H[~tr] @ Wc + b))); T = 1 / (1 + np.exp(-U[~tr]))
            err.append(np.mean((P - T) ** 2))
        Wc, b = ridge(H, U, lam, prior_W)
        P = 1 / (1 + np.exp(-(H @ Wc + b))); T = 1 / (1 + np.exp(-U))
        out[lam] = (float(np.sqrt(np.mean(err))), float(np.sqrt(np.mean((P - T) ** 2))))
    return out


LAMS = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
rep = {}
cvw = cv(U_w, LAMS); rep["wire_cv"] = {str(k): v for k, v in cvw.items()}
print("wire ridge: lambda -> (LOSO rms pos err, train rms pos err)"); [print(f"  {k:g}: {v[0]:.4f} {v[1]:.4f}") for k, v in cvw.items()]
# held-out CV never beats the mean predictor (the 4-D latent does not generalise, E2), so CV
# cannot pick lambda: take the most regularised fit that still holds the training snares
# (train rms position error <= 0.01 = ~1% of each log range)
lam_w = max(k for k, v in cvw.items() if v[1] <= 0.01)
print("chosen lambda", lam_w); rep["lam_wire"] = lam_w
Wc, bc = ridge(H, U_w, lam_w)
W_new = {k: v.copy() for k, v in W.items()}
W_new["l3.W"][:, WCOLS] = Wc; W_new["l3.b"][0, WCOLS] = bc
# baseline position error: each snare's per-snare fit vs its 32-snare mean
Tpos = 1 / (1 + np.exp(-U_w)); rep["mean_predictor_rms_pos"] = float(np.sqrt(np.mean((Tpos - Tpos.mean(0)) ** 2)))
Pd = 1 / (1 + np.exp(-(H @ Wc + bc)))
rep["train_pos_err_per_knob"] = dict(zip(WIRE, np.sqrt(np.mean((Pd - Tpos) ** 2, 0)).tolist()))
rep["train_log_err_per_knob"] = {k: float(np.sqrt(np.mean(((Pd[:, i] - Tpos[:, i]) * (math.log(SEG[k]["hi"]) - math.log(SEG[k]["lo"]))) ** 2))) for i, k in enumerate(WIRE)}
print("mean-predictor rms pos err", rep["mean_predictor_rms_pos"]); print("train log err per knob", rep["train_log_err_per_knob"])
# rebal variant: band_amp columns too (ridge toward the existing columns: prior = current W3)
cvb = cv(U_b, LAMS, prior_W=W["l3.W"][:, BCOLS]); rep["band_cv"] = {str(k): v for k, v in cvb.items()}
lam_b = max(k for k, v in cvb.items() if v[1] <= 0.01); rep["lam_band"] = lam_b
Wb, bb = ridge(H, U_b, lam_b, prior_W=W["l3.W"][:, BCOLS])
W_rb = {k: v.copy() for k, v in W_new.items()}
W_rb["l3.W"][:, BCOLS] = Wb; W_rb["l3.b"][0, BCOLS] = bb
print("band ridge lambda", lam_b, cvb[lam_b])


def save(Wt, fn):
    src = json.load(open(os.path.join(RUN, "weights.json")))
    out = {}
    for l in ("l1", "l2", "l3"):
        out[l] = {"W": {"shape": list(Wt[f"{l}.W"].shape), "data": Wt[f"{l}.W"].ravel().tolist()},
                  "b": {"shape": list(Wt[f"{l}.b"].shape), "data": Wt[f"{l}.b"].ravel().tolist()}}
    for k in src:
        if k not in out: out[k] = src[k]
    json.dump(out, open(fn, "w"))


save(W_new, "weights_wires.json"); save(W_rb, "weights_wires_rebal.json")
# exactness: only the intended columns changed
for Wt, cols in ((W_new, WCOLS), (W_rb, WCOLS + BCOLS)):
    keep = [c for c in range(W["l3.W"].shape[1]) if c not in cols]
    assert np.array_equal(Wt["l3.W"][:, keep], W["l3.W"][:, keep]) and np.array_equal(Wt["l1.W"], W["l1.W"])
# ---- sanity at 20 random z ~ N(mu, Sigma) of the codes (E2 prior), at v 0.4..1
rng = np.random.default_rng(7); mu = Z.mean(0); Sg = np.cov(Z.T)
tmin = {k: min(t[k] for t in targets.values()) for k in WIRE}; tmax = {k: max(t[k] for t in targets.values()) for k in WIRE}
rs = []
for i in range(20):
    z = rng.multivariate_normal(mu, Sg * (1.0 if i < 10 else 4.0)); v = float(rng.uniform(0.4, 1.0))
    p = decode(z, v, W_new); kn = {k: float(np.ravel(p[k])[0]) for k in WIRE}
    inr = all(tmin[k] / 1.5 <= kn[k] <= tmax[k] * 1.5 for k in WIRE)
    rs.append(dict(spread="1x" if i < 10 else "2x", vel=v, knobs=kn, within_corpus_x1p5=inr))
rep["random_z"] = rs; rep["corpus_range"] = {k: [tmin[k], tmax[k]] for k in WIRE}
print("random z within 1.5x of corpus knob range:", sum(x["within_corpus_x1p5"] for x in rs[:10]), "/10 at 1x,",
      sum(x["within_corpus_x1p5"] for x in rs[10:]), "/10 at 2x")
for x in rs: print(" ", x["spread"], f"v{x['vel']:.2f}", " ".join(f"{k[5:]}={x['knobs'][k]:.3g}" for k in WIRE), x["within_corpus_x1p5"])
json.dump(rep, open("distill.json", "w"), indent=1)
