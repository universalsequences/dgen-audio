// DGenAnalysisC.h — C kernels for the kick analysis hot loops (DGenAnalysis).
#ifndef DGEN_ANALYSIS_C_H
#define DGEN_ANALYSIS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/// NoiseFit's joint-refine model (NoiseFit.fit, residualE): band envelopes + noise events mixed
/// through the analysis kernel, in dB against the measured band powers, plus the band smoothness
/// terms. All arrays are borrowed for the duration of a call.
typedef struct {
    int nb;            // synthesis bands
    int na;            // analysis bands
    int ps;            // per-band parameter stride (4 or 5: [ln L, m, d1, d2(, ln H)])
    int nfit;          // fitted frames
    int sharedWhite;   // 1: quadratic form through K[a][b][c]; 0: diagonal M[a][b]
    int ev0, evW;      // first event parameter, parameters per event
    const double *ft;  // [nfit] frame times
    const double *eDB; // [na * nfit] target band powers in dB
    const double *K;   // [na * nb * nb] (sharedWhite)
    const double *M;   // [na * nb] (diagonal)
    double floorE;     // power floor
    double w;          // smoothness weight
} DGenNFProblem;

/// Number of residuals for `np` parameters.
int dgen_nf_residual_count(const DGenNFProblem *pr, int np);

/// Evaluates the residuals of `L` parameter vectors at once (P: L rows of np, out: L rows of
/// dgen_nf_residual_count). Lanes are computed together in SIMD vectors; each lane's arithmetic
/// is the same IEEE operations in the same order as the Swift reference, so results match it bit
/// for bit.
void dgen_nf_residual_batch(const DGenNFProblem *pr, const double *P, int L, int np, double *out);

/// Residuals and their analytic Jacobian at one parameter vector p (np): r[nout], J[nout * np]
/// row-major; J must arrive zeroed (only the nonzero entries are written). r matches dgen_nf_residual_batch exactly; J is the exact derivative (the event-start
/// step at t = t_k is ignored), not a finite difference.
void dgen_nf_jacobian(const DGenNFProblem *pr, const double *p, int np, double *r, double *J);

/// Normal equations of a row-major m×n Jacobian: JtJ = JᵀJ (full symmetric n×n), Jtr = Jᵀr.
void dgen_normal_equations(const double *J, const double *r, int m, int n, double *JtJ, double *Jtr);

/// Solves A x = b for symmetric positive definite A (n×n) by Cholesky; x overwrites b.
/// Returns 0 on success, nonzero when A is not positive definite.
int dgen_cholesky_solve(const double *A, int n, double *b);

/// Linear least squares over basis rows B (m rows of n samples, row-major): G = B Bᵀ (full m×m),
/// rhs = B x.
void dgen_gram(const double *B, int m, int n, const double *x, double *G, double *rhs);

/// Least squares min ‖A x − b‖ (A m0×n row-major), optionally with sqrt(ridge)·I rows appended:
/// column scaling, LAPACK Householder QR, columns with |R_kk| ≤ 1e-13·|R_00| dropped (x_k = 0).
int dgen_lstsq(const double *A, int m0, int n, const double *b, double ridge, double *x);

/// Symmetric eigen-decomposition (LAPACK dsyevd): eigenvalues descending, vectors[i*n + k] =
/// component i of the k-th eigenvector.
int dgen_sym_eigen(const double *A, int n, double *values, double *vectors);

/// C = Hᵀ H for the Hankel matrix H[i][a] = x[i + a] (rows × cols); C full cols×cols row-major.
void dgen_hankel_gram(const double *x, int rows, int cols, double *C);

/// In-place complex FFT of length n (a power of two) by vDSP: forward e^{-i2πkn/N}, inverse
/// scaled by 1/N. Returns nonzero for an unsupported size.
int dgen_fft(double *re, double *im, int n, int inverse);

/// A mode's drop-law parameters (ModeParams without A, φ).
typedef struct { double F, G, T, D, R, G2, T2; } DGenModeParams;

/// Unit-amplitude columns of a mode on t = i/sr, i < n: c = env·cos(phase), s = env·sin(phase).
void dgen_mode_columns(const DGenModeParams *m, int n, double sr, double *c, double *s);

/// y[i] += A · env(t) · sin(phase(t) + φ), t = i/sr.
void dgen_mode_add(const DGenModeParams *m, double A, double phi, int n, double sr, double *y);

/// c[i] = e^(lr·i) cos(w·i), s[i] = e^(lr·i) sin(w·i).
void dgen_damped_columns(double lr, double w, int n, double *c, double *s);

/// Joint amplitudes of k damped sinusoids e^(lr_j·i)·(cos, sin)(w_j·i) on x[0..N) by least squares
/// (ridge as dgen_lstsq): coef[2j], coef[2j+1] = cos, sin weights; resid (optional) = x − fit.
int dgen_damped_fit(const double *x, int N, const double *lr, const double *w, int k, double ridge,
                    double *coef, double *resid);

/// Σ_i (x_i − Σ_c coef_c B_c,i)² for basis rows B (m×n row-major).
double dgen_lsq_sse(const double *B, int m, int n, const double *coef, const double *x);

#ifdef __cplusplus
}
#endif

#endif
