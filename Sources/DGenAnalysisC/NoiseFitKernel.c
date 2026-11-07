// NoiseFitKernel.c — NoiseFit's joint-refine residuals for several parameter vectors at once.
//
// LM's finite-difference Jacobian (DGEN_NOISEFIT=fd) evaluates one residual per parameter, all of
// the same problem. Those evaluations share every constant (frames, kernel, targets) and run identical
// control flow, so they map onto SIMD lanes: here 4 lanes per vector (clang ext_vector_type; two
// NEON registers on arm64). The 16x16 quadratic form per analysis band and frame — nearly all of
// the arithmetic — runs once for 4 lanes. exp / log10 stay scalar libm calls per lane, and the
// target is built with -ffp-contract=off, so every lane reproduces the Swift reference exactly.

#include "DGenAnalysisC.h"
#include "Internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LANES 4
typedef double v4 __attribute__((ext_vector_type(LANES)));

int dgen_nf_residual_count(const DGenNFProblem *pr, int np) {
    int nev = (np - pr->ev0) / pr->evW;
    return pr->nfit * pr->na + pr->ps * (pr->nb - 1) + nev * (pr->nb - 1);
}

/// exp of the first `active` lanes (the rest are padding: 0, never read back).
static inline v4 vexp(v4 x, int active) {
    v4 r = 0.0;
    for (int l = 0; l < active; l++) r[l] = exp(x[l]);
    return r;
}

/// One group of up to LANES parameter vectors (pv: np vectors, lane l = parameter vector l; only the
/// first `active` lanes are evaluated through libm, so a single vector costs one lane of exp / log).
static void group(const DGenNFProblem *pr, const v4 *pv, int np, int active, v4 *outv, v4 *N, v4 *scratch) {
    const int nb = pr->nb, na = pr->na, ps = pr->ps, nfit = pr->nfit;
    const int nev = (np - pr->ev0) / pr->evW;
    const v4 rnv = pv[ps * nb];
    // per band: L, h (t-independent)
    v4 *Lb = scratch, *hb = scratch + nb;
    // per event: t, L, d, r, colour[nb]
    v4 *ev = scratch + 2 * nb;  // nev * (4 + nb)
    for (int b = 0; b < nb; b++) {
        Lb[b] = vexp(pv[ps * b], active);
        hb[b] = ps == 5 ? vexp(pv[ps * b + 4], active) : (v4)0.0;
    }
    for (int k = 0; k < nev; k++) {
        const int o = pr->ev0 + k * pr->evW;
        v4 *e = ev + k * (4 + nb);
        e[0] = pv[o];
        e[1] = vexp(pv[o + 1], active);
        e[2] = vexp(pv[o + 2], active);
        e[3] = vexp(pv[o + 3], active);
        for (int b = 0; b < nb; b++) e[4 + b] = vexp(pv[o + 4 + b], active);
    }
    int o = 0;
    for (int ii = 0; ii < nfit; ii++) {
        const double t = pr->ft[ii];
        const v4 rise = 1.0 - vexp(-t / rnv, active);
        for (int b = 0; b < nb; b++) {
            const int q = ps * b;
            const v4 m = pv[q + 1];
            const v4 e1 = vexp(-t / pv[q + 2], active), e2 = vexp(-t / pv[q + 3], active);
            N[b] = (Lb[b] * (m * e1 + (1.0 - m) * e2) + hb[b]) * rise;
        }
        for (int k = 0; k < nev; k++) {
            const v4 *e = ev + k * (4 + nb);
            for (int l = 0; l < active; l++) {
                if (!(t > e[0][l])) continue;
                const double d = t - e[0][l];
                const double g = e[1][l] * (1 - exp(-d / e[3][l])) * exp(-d / e[2][l]);
                for (int b = 0; b < nb; b++) N[b][l] += g * e[4 + b][l];
            }
        }
        for (int a = 0; a < na; a++) {
            v4 s = 0.0;
            if (pr->sharedWhite) {
                const double *Ka = pr->K + (size_t)a * nb * nb;
                // four rows at a time: independent accumulators break the add-latency chain;
                // each row's sum and the sum over rows keep the reference order
                int b = 0;
                for (; b + 4 <= nb; b += 4) {
                    const double *r0 = Ka + b * nb, *r1 = r0 + nb, *r2 = r1 + nb, *r3 = r2 + nb;
                    v4 t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
                    for (int c = 0; c < nb; c++) {
                        const v4 nc = N[c];
                        t0 += r0[c] * nc; t1 += r1[c] * nc; t2 += r2[c] * nc; t3 += r3[c] * nc;
                    }
                    s += N[b] * t0; s += N[b + 1] * t1; s += N[b + 2] * t2; s += N[b + 3] * t3;
                }
                for (; b < nb; b++) {
                    const double *row = Ka + b * nb;
                    v4 tt = 0.0;
                    for (int c = 0; c < nb; c++) tt += row[c] * N[c];
                    s += N[b] * tt;
                }
            } else {
                const double *Ma = pr->M + (size_t)a * nb;
                for (int b = 0; b < nb; b++) s += Ma[b] * N[b] * N[b];
            }
            const double target = pr->eDB[(size_t)a * nfit + ii];
            v4 r = 0.0;
            for (int l = 0; l < active; l++) {
                const double sl = (0.0 >= s[l]) ? 0.0 : s[l];  // Swift max(s, 0)
                r[l] = 10 * log10(sl + pr->floorE) - target;
            }
            outv[o++] = r;
        }
    }
    const double w = pr->w;
    for (int b = 0; b < nb - 1; b++) {
        const int p0 = ps * b, p1 = ps * b + ps;
        outv[o++] = w * 8.686 * (pv[p0] - pv[p1]);
        outv[o++] = w * 4 * (pv[p0 + 1] - pv[p1 + 1]);
        v4 r2 = 0.0, r3 = 0.0;
        for (int l = 0; l < active; l++) {
            r2[l] = w * 2 * log(pv[p0 + 2][l] / pv[p1 + 2][l]);
            r3[l] = w * 2 * log(pv[p0 + 3][l] / pv[p1 + 3][l]);
        }
        outv[o++] = r2;
        outv[o++] = r3;
        if (ps == 5) outv[o++] = w * 8.686 * (pv[p0 + 4] - pv[p1 + 4]);
    }
    for (int k = 0; k < nev; k++) {
        const int c0 = pr->ev0 + k * pr->evW + 4;
        for (int b = 0; b < nb - 1; b++) outv[o++] = w * 8.686 * (pv[c0 + b] - pv[c0 + b + 1]);
    }
}

void dgen_nf_residual_batch(const DGenNFProblem *pr, const double *P, int L, int np, double *out) {
    const int nout = dgen_nf_residual_count(pr, np);
    const int nev = (np - pr->ev0) / pr->evW;
    v4 *pv = dgen_xmalloc(sizeof(v4) * np);
    v4 *outv = dgen_xmalloc(sizeof(v4) * nout);
    v4 *N = dgen_xmalloc(sizeof(v4) * pr->nb);
    v4 *scratch = dgen_xmalloc(sizeof(v4) * (2 * pr->nb + nev * (4 + pr->nb) + 1));
    for (int g0 = 0; g0 < L; g0 += LANES) {
        // transpose up to LANES rows into lane vectors (padding lanes repeat the last row)
        const int active = L - g0 < LANES ? L - g0 : LANES;
        for (int j = 0; j < np; j++)
            for (int l = 0; l < LANES; l++) {
                const int row = g0 + l < L ? g0 + l : L - 1;
                pv[j][l] = P[(size_t)row * np + j];
            }
        group(pr, pv, np, active, outv, N, scratch);
        for (int l = 0; l < LANES && g0 + l < L; l++) {
            double *dst = out + (size_t)(g0 + l) * nout;
            for (int i = 0; i < nout; i++) dst[i] = outv[i][l];
        }
    }
    free(pv); free(outv); free(N); free(scratch);
}

// MARK: analytic Jacobian
//
// N_b(t) = (L_b (m_b e1 + (1 - m_b) e2) + h_b) · rise  +  Σ_k g_k(t) c_kb,  rise = 1 − e^(−t/r_n),
// e1 = e^(−t/d1_b), e2 = e^(−t/d2_b), g_k = L_k (1 − e^(−d/r_k)) e^(−d/τ_k), d = t − t_k (t > t_k).
// s_a = Nᵀ K_a N,  r_a = 10 log10(max(s_a, 0) + floor) − target,
// ∂r_a/∂θ = 10/ln10 · Σ_b (K_a N + K_aᵀ N)_b ∂N_b/∂θ / (s_a + floor)  (0 where s_a ≤ 0).
void dgen_nf_jacobian(const DGenNFProblem *pr, const double *p, int np, double *r, double *J) {
    const int nb = pr->nb, na = pr->na, ps = pr->ps, nfit = pr->nfit;
    const int ev0 = pr->ev0, evW = pr->evW, nev = (np - ev0) / evW;
    const int irn = ps * nb;
    const double rnv = p[irn];
    double *buf = dgen_xmalloc(sizeof(double) * (size_t)(nb * 12 + nev * (5 + 2 * nb) + 8));
    double *Lb = buf, *hb = Lb + nb, *N = hb + nb, *core = N + nb, *u = core + nb, *v = u + nb, *G = v + nb;
    double *dN = G + nb;                 // [nb][5]: ∂N_b/∂(ln L, m, d1, d2, ln H)
    double *evp = dN + 5 * nb;           // per event: t, L, tau, rr, active, g, dg/dt, dg/dlnL, dg/dlntau, dg/dlnr
    double *evc = evp + nev * 10;        // per event: colour[nb]
    for (int b = 0; b < nb; b++) {
        Lb[b] = exp(p[ps * b]);
        hb[b] = ps == 5 ? exp(p[ps * b + 4]) : 0;
    }
    for (int k = 0; k < nev; k++) {
        const int o = ev0 + k * evW;
        evp[k * 10 + 0] = p[o];
        evp[k * 10 + 1] = exp(p[o + 1]);
        evp[k * 10 + 2] = exp(p[o + 2]);
        evp[k * 10 + 3] = exp(p[o + 3]);
        for (int b = 0; b < nb; b++) evc[k * nb + b] = exp(p[o + 4 + b]);
    }
    const double c10 = 10.0 / log(10.0);
    int row = 0;
    for (int ii = 0; ii < nfit; ii++) {
        const double t = pr->ft[ii];
        const double er = exp(-t / rnv), rise = 1 - er;
        const double drise = -er * t / (rnv * rnv);  // ∂rise/∂r_n
        for (int b = 0; b < nb; b++) {
            const int q = ps * b;
            const double m = p[q + 1], d1 = p[q + 2], d2 = p[q + 3];
            const double e1 = exp(-t / d1), e2 = exp(-t / d2);
            const double mix = m * e1 + (1 - m) * e2;
            core[b] = Lb[b] * mix + hb[b];
            N[b] = (Lb[b] * (m * e1 + (1 - m) * e2) + hb[b]) * rise;
            dN[b * 5 + 0] = Lb[b] * mix * rise;
            dN[b * 5 + 1] = Lb[b] * (e1 - e2) * rise;
            dN[b * 5 + 2] = Lb[b] * m * e1 * (t / (d1 * d1)) * rise;
            dN[b * 5 + 3] = Lb[b] * (1 - m) * e2 * (t / (d2 * d2)) * rise;
            dN[b * 5 + 4] = hb[b] * rise;
        }
        for (int k = 0; k < nev; k++) {
            double *e = evp + k * 10;
            e[4] = t > e[0];
            if (!e[4]) continue;
            const double d = t - e[0], Lk = e[1], tau = e[2], rr = e[3];
            const double ea = exp(-d / rr), ed = exp(-d / tau);
            const double g = Lk * (1 - ea) * ed;
            e[5] = g;
            e[6] = -Lk * ((ea / rr) * ed - (1 - ea) * ed / tau);  // ∂g/∂t_k = −∂g/∂d
            e[7] = g;
            e[8] = Lk * (1 - ea) * ed * (d / tau);
            e[9] = Lk * ed * (-ea * d / rr);
            for (int b = 0; b < nb; b++) N[b] += g * evc[k * nb + b];
        }
        for (int a = 0; a < na; a++, row++) {
            double s = 0.0;
            if (pr->sharedWhite) {
                const double *Ka = pr->K + (size_t)a * nb * nb;
                for (int b = 0; b < nb; b++) {
                    const double *rw = Ka + b * nb;
                    double tt = 0.0;
                    for (int c = 0; c < nb; c++) tt += rw[c] * N[c];
                    u[b] = tt;
                    s += N[b] * tt;
                }
                for (int c = 0; c < nb; c++) v[c] = 0;
                for (int b = 0; b < nb; b++) {
                    const double *rw = Ka + b * nb;
                    const double nbv = N[b];
                    for (int c = 0; c < nb; c++) v[c] += rw[c] * nbv;
                }
                for (int b = 0; b < nb; b++) G[b] = u[b] + v[b];
            } else {
                const double *Ma = pr->M + (size_t)a * nb;
                for (int b = 0; b < nb; b++) { s += Ma[b] * N[b] * N[b]; G[b] = 2 * Ma[b] * N[b]; }
            }
            const double sl = (0.0 >= s) ? 0.0 : s;
            r[row] = 10 * log10(sl + pr->floorE) - pr->eDB[(size_t)a * nfit + ii];
            const double coef = s > 0 ? c10 / (s + pr->floorE) : 0;
            double *Jr = J + (size_t)row * np;
            double grn = 0;
            for (int b = 0; b < nb; b++) {
                const double gb = coef * G[b];
                const int q = ps * b;
                for (int j = 0; j < ps; j++) Jr[q + j] = gb * dN[b * 5 + j];
                grn += gb * core[b] * drise;
            }
            Jr[irn] = grn;
            for (int k = 0; k < nev; k++) {
                const double *e = evp + k * 10;
                if (!e[4]) continue;
                const int o = ev0 + k * evW;
                double gc = 0;  // Σ_b coef G_b c_kb
                for (int b = 0; b < nb; b++) {
                    const double w = coef * G[b] * evc[k * nb + b];
                    gc += w;
                    Jr[o + 4 + b] = w * e[5];
                }
                Jr[o] = gc * e[6];
                Jr[o + 1] = gc * e[7];
                Jr[o + 2] = gc * e[8];
                Jr[o + 3] = gc * e[9];
            }
        }
    }
    // smoothness rows
    const double w = pr->w;
    for (int b = 0; b < nb - 1; b++) {
        const int p0 = ps * b, p1 = ps * b + ps;
        double *Jr;
        Jr = J + (size_t)row * np; r[row++] = w * 8.686 * (p[p0] - p[p1]); Jr[p0] = w * 8.686; Jr[p1] = -w * 8.686;
        Jr = J + (size_t)row * np; r[row++] = w * 4 * (p[p0 + 1] - p[p1 + 1]); Jr[p0 + 1] = w * 4; Jr[p1 + 1] = -w * 4;
        Jr = J + (size_t)row * np; r[row++] = w * 2 * log(p[p0 + 2] / p[p1 + 2]); Jr[p0 + 2] = w * 2 / p[p0 + 2]; Jr[p1 + 2] = -w * 2 / p[p1 + 2];
        Jr = J + (size_t)row * np; r[row++] = w * 2 * log(p[p0 + 3] / p[p1 + 3]); Jr[p0 + 3] = w * 2 / p[p0 + 3]; Jr[p1 + 3] = -w * 2 / p[p1 + 3];
        if (ps == 5) { Jr = J + (size_t)row * np; r[row++] = w * 8.686 * (p[p0 + 4] - p[p1 + 4]); Jr[p0 + 4] = w * 8.686; Jr[p1 + 4] = -w * 8.686; }
    }
    for (int k = 0; k < nev; k++) {
        const int c0 = ev0 + k * evW + 4;
        for (int b = 0; b < nb - 1; b++) {
            double *Jr = J + (size_t)row * np;
            r[row++] = w * 8.686 * (p[c0 + b] - p[c0 + b + 1]);
            Jr[c0 + b] = w * 8.686; Jr[c0 + b + 1] = -w * 8.686;
        }
    }
    free(buf);
}
