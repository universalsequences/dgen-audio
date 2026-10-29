#!/usr/bin/env python3
"""Stage 1: param-space autodecoder pretrain of the Latent Kick decoder (E2 recipe, numpy).

z_k (D per row, free) -> tanh MLP D->128->128->P -> u ; loss = weighted squared error of
sigmoid(u) against the teacher's sigmoid(u) positions (kick/data/teacher.json), measured in
natural units (log segments as log ratio, mode_f x30, mode params x sqrt(slot energy share)),
normalised so the mean-row predictor scores 1; + z L2 + E2's range box on random z.

    python3 train_np.py --out runs/pre_d4 [--steps 20000] [--D 4] [--rows fit|fit+blend]   (from $MODAL_AD_ROOT/kick)
"""
import argparse, json, math, os, sys, time
import numpy as np
import kick_layout as L

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ModalSnareAD", "scripts"))
import modal_ad as MAD  # noqa: E402
DATA = os.path.join(MAD.stage("kick"), "data")


def col_weights(rows_P):
    off = L.offsets()
    w = np.zeros(L.P)
    for name, (o, wd, sq, lo, hi) in off.items():
        scale = math.log(hi / lo) if sq == "log" else (math.pi if sq == "pi" else 1.0)
        w[o:o + wd] = scale
    # slot importance: sqrt(energy share) of the slot across rows (+ floor)
    share = np.mean([np.asarray(P["mode_amp"]) ** 2 * np.asarray(P["mode_dec"]) for P in rows_P], 0)
    share = share / share.sum()
    imp = np.sqrt(share) + 0.05
    for name in ("mode_f", "mode_dec", "mode_amp", "mode_glide", "mode_tau", "mode_rise", "mode_t0",
                 "mode_phase", "mode_mga", "mode_mgb", "mode_msm"):
        o, wd = off[name][0], off[name][1]
        w[o:o + wd] *= imp / imp.max()
    o, wd = off["mode_f"][0], off["mode_f"][1]
    w[o:o + wd] *= 30.0
    for name in ("mode_glide2", "mode_tau2"):
        o, wd = off[name][0], off[name][1]
        w[o:o + wd] *= 3.0
    for name in ("len", "hold", "rel", "fd_s", "fd_frac", "oA_e", "gA_hC", "gB_hC"):
        o = off[name][0]
        w[o] *= 5.0
    for name in ("band_amp", "band_d1", "band_tail", "band_d2", "band_couple"):
        o, wd = off[name][0], off[name][1]
        w[o:o + wd] *= 0.5
    return w


class Adam:
    def __init__(self, params, lr):
        self.p = params; self.lr = lr
        self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]; self.t = 0

    def step(self, grads, mask=None):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, (x, g) in enumerate(zip(self.p, grads)):
            if g is None:
                continue
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t); vh = self.v[i] / (1 - b2 ** self.t)
            x -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def forward(W, Z):
    h1 = np.tanh(Z @ W["W1"] + W["b1"]); h2 = np.tanh(h1 @ W["W2"] + W["b2"])
    return h1, h2, h2 @ W["W3"] + W["b3"]


def train(rows, D=4, H=128, steps=20000, lr=3e-3, seed=7, box=3.0, zprior=1e-4, log=None,
          row_w=None, tie=None, tie_w=0.0):
    """rows: list of (u, P). tie: list of (ia, ib, t, u_target, P) blend rows decoded at
    (1-t) z_a + t z_b (z tied to the endpoints)."""
    rng = np.random.default_rng(seed)
    U = np.array([u for u, _ in rows]); N = len(U)
    S = 1 / (1 + np.exp(-U))                             # target positions
    w = col_weights([P for _, P in rows])
    base = np.mean((S - S.mean(0)) ** 2 * w ** 2)        # mean-row predictor = 1
    rw = np.ones(N) if row_w is None else np.asarray(row_w)
    # z init: PCA of positions
    C = (S - S.mean(0)) * w
    _, sv, Vt = np.linalg.svd(C, full_matrices=False)
    Z = C @ Vt[:D].T
    Z = Z / Z.std(0) * 0.5
    W = {"W1": rng.normal(0, 1 / math.sqrt(D), (D, H)), "b1": np.zeros(H),
         "W2": rng.normal(0, 1 / math.sqrt(H), (H, H)), "b2": np.zeros(H),
         "W3": rng.normal(0, 0.01 / math.sqrt(H), (H, L.P)), "b3": np.log(S.mean(0) / (1 - S.mean(0)))}
    smin, smax = S.min(0), S.max(0)
    keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
    opt = Adam([W[k] for k in keys] + [Z], lr)
    if tie:
        TA = np.array([a for a, b, t, u, P in tie]); TB = np.array([b for a, b, t, u, P in tie])
        TT = np.array([t for a, b, t, u, P in tie])[:, None]
        TS = 1 / (1 + np.exp(-np.array([u for a, b, t, u, P in tie])))
    t0 = time.time()
    for it in range(steps):
        lr_t = lr * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.95 + 0.05)
        opt.lr = lr_t
        grads = {k: np.zeros_like(W[k]) for k in keys}
        gZ = np.zeros_like(Z)

        def backprop(Zin, dU, accumulate_z=True):
            h1, h2, u = forward(W, Zin)
            grads["W3"] += h2.T @ dU; grads["b3"] += dU.sum(0)
            dh2 = dU @ W["W3"].T * (1 - h2 ** 2)
            grads["W2"] += h1.T @ dh2; grads["b2"] += dh2.sum(0)
            dh1 = dh2 @ W["W2"].T * (1 - h1 ** 2)
            grads["W1"] += Zin.T @ dh1; grads["b1"] += dh1.sum(0)
            return dh1 @ W["W1"].T

        # reconstruction
        _, _, u = forward(W, Z)
        s = 1 / (1 + np.exp(-u))
        r = (s - S) * w
        loss = np.mean(rw[:, None] * r ** 2) / base
        dU = 2 * rw[:, None] * r * w * s * (1 - s) / (N * L.P * base)
        gZ += backprop(Z, dU)
        # tied blend rows
        lt = 0.0
        if tie:
            Zt = (1 - TT) * Z[TA] + TT * Z[TB]
            _, _, ut = forward(W, Zt)
            st = 1 / (1 + np.exp(-ut)); rt = (st - TS) * w
            lt = tie_w * np.mean(rt ** 2) / base
            dUt = tie_w * 2 * rt * w * st * (1 - st) / (len(TT) * L.P * base)
            dZt = backprop(Zt, dUt)
            np.add.at(gZ, TA, (1 - TT) * dZt); np.add.at(gZ, TB, TT * dZt)
        # range box on random z ~ N(mu, Sigma) at spreads 1, 1.5, 2
        lb = 0.0
        if box > 0:
            spread = [1.0, 1.5, 2.0][it % 3]
            mu = Z.mean(0); Lc = np.linalg.cholesky(np.cov(Z.T) + 1e-6 * np.eye(D))
            Zr = mu + spread * rng.normal(0, 1, (64, D)) @ Lc.T
            _, _, ur = forward(W, Zr)
            sr = 1 / (1 + np.exp(-ur))
            over = np.maximum(sr - smax, 0) + np.minimum(sr - smin, 0)
            lb = box * np.mean((over * w) ** 2) / base
            dUr = box * 2 * over * w ** 2 * sr * (1 - sr) / (64 * L.P * base)
            backprop(Zr, dUr)
        gZ += 2 * zprior * Z / N
        opt.step([grads[k] for k in keys] + [gZ])
        if it % 1000 == 0 or it == steps - 1:
            msg = f"step {it} loss {loss:.5f} tie {lt:.5f} box {lb:.5f} lr {lr_t:.2e} ({time.time() - t0:.0f}s)"
            print(msg, flush=True)
            if log:
                log.write(msg + "\n")
    return W, Z


def cents_report(W, Z, rows):
    off = L.offsets()
    _, _, u = forward(W, Z)
    out = []
    for k, (ut, P) in enumerate(rows):
        D = L.decode_u(u[k]); T = L.decode_u(np.asarray(ut))
        f, ft = np.asarray(D["mode_f"]), np.asarray(T["mode_f"])
        e = np.asarray(T["mode_amp"]) ** 2 * np.asarray(T["mode_dec"])
        c = 1200 * np.abs(np.log2(f / ft))
        out.append((float(np.sum(c * e) / e.sum()), float(c[e > 0.01 * e.max()].max())))
    return out


def save(W, Z, ids, outdir, meta):
    os.makedirs(outdir, exist_ok=True)
    mat = lambda a: {"shape": list(a.shape), "data": [float(x) for x in a.reshape(-1)]}
    json.dump({"l1": {"W": mat(W["W1"]), "b": mat(W["b1"][None])}, "l2": {"W": mat(W["W2"]), "b": mat(W["b2"][None])},
               "l3": {"W": mat(W["W3"]), "b": mat(W["b3"][None])}}, open(os.path.join(outdir, "weights.json"), "w"))
    json.dump({"order": ids, "codes": {i: [float(x) for x in z] for i, z in zip(ids, Z)}, "meta": meta},
              open(os.path.join(outdir, "codes.json"), "w"), indent=1)


def load(outdir):
    w = json.load(open(os.path.join(outdir, "weights.json")))
    a = lambda v: np.asarray(v["data"], float).reshape(v["shape"])
    W = {"W1": a(w["l1"]["W"]), "b1": a(w["l1"]["b"])[0], "W2": a(w["l2"]["W"]), "b2": a(w["l2"]["b"])[0],
         "W3": a(w["l3"]["W"]), "b3": a(w["l3"]["b"])[0]}
    c = json.load(open(os.path.join(outdir, "codes.json")))
    return W, np.array([c["codes"][i] for i in c["order"]]), c["order"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--box", type=float, default=3.0)
    ap.add_argument("--tie", default="", help="path to tied blend rows json (blend_rows.py)")
    ap.add_argument("--tie-w", type=float, default=0.03)
    a = ap.parse_args()
    T = json.load(open(os.path.join(DATA, "teacher.json")))["rows"]
    ids = [f"kick_{k:02d}_v100" for k in range(33)]
    rows = [(np.array(T[i]["u"]), T[i]["P"]) for i in ids]
    tie = None
    if a.tie:
        tie = [(r["a"], r["b"], r["t"], np.array(r["u"]), None) for r in json.load(open(a.tie))["rows"]]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.log"), "w") as lg:
        W, Z = train(rows, D=a.D, steps=a.steps, lr=a.lr, box=a.box, log=lg, tie=tie, tie_w=a.tie_w)
    c = cents_report(W, Z, rows)
    print("cents energy-weighted median %.3f max %.3f; worst live-mode max %.2f" %
          (np.median([x[0] for x in c]), max(x[0] for x in c), max(x[1] for x in c)))
    save(W, Z, ids, a.out, {"D": a.D, "steps": a.steps, "lr": a.lr, "box": a.box, "tie": a.tie, "tie_w": a.tie_w,
                            "cents": c})
