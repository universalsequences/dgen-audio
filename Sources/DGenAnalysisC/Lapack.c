// Lapack.c — least squares, symmetric eigen-decomposition and Hankel covariance for DGenAnalysis
// (LinAlg.leastSquares, LinAlg.symmetricEigen, MatrixPencil). LAPACK / BLAS from Accelerate on
// Apple platforms; elsewhere a portable Householder QR, loops, and (eigen) a nonzero return so the
// Swift Jacobi reference runs.
#include "DGenAnalysisC.h"
#include "Internal.h"
#include <math.h>
#include <string.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

/// Householder QR of the column-major m×n `a` (destroyed), applying Qᵀ to y; on return the upper
/// triangle of `a` holds R and y[0..<min(m,n)] holds (Qᵀy). Returns 0 on success.
static int qr_apply(double *a, double *y, int m, int n) {
#if defined(__APPLE__)
    __LAPACK_int M = m, N = n, K = m < n ? m : n, lda = m, info = 0, lwork = -1, one = 1;
    double *tau = dgen_xmalloc(sizeof(double) * (K > 0 ? K : 1)), wq;
    dgeqrf_(&M, &N, a, &lda, tau, &wq, &lwork, &info);
    lwork = (__LAPACK_int)wq;
    double *work = dgen_xmalloc(sizeof(double) * (lwork > 1 ? lwork : 1));
    dgeqrf_(&M, &N, a, &lda, tau, work, &lwork, &info);
    int status = (int)info;
    if (status == 0) {
        __LAPACK_int lw2 = -1;
        dormqr_("L", "T", &M, &one, &K, a, &lda, tau, y, &M, &wq, &lw2, &info);
        lw2 = (__LAPACK_int)wq;
        double *work2 = dgen_xmalloc(sizeof(double) * (lw2 > 1 ? lw2 : 1));
        dormqr_("L", "T", &M, &one, &K, a, &lda, tau, y, &M, work2, &lw2, &info);
        status = (int)info;
        free(work2);
    }
    free(tau); free(work);
    return status;
#else
    const int K = m < n ? m : n;
    double *v = dgen_xmalloc(sizeof(double) * (size_t)m);
    for (int k = 0; k < K; k++) {
        double *ck = a + (size_t)k * m, norm = 0;
        for (int i = k; i < m; i++) norm += ck[i] * ck[i];
        norm = sqrt(norm);
        if (norm == 0) continue;
        const double alpha = ck[k] > 0 ? -norm : norm;
        for (int i = k; i < m; i++) v[i - k] = ck[i];
        v[0] -= alpha;
        double vn = 0;
        for (int i = 0; i < m - k; i++) vn += v[i] * v[i];
        if (vn == 0) continue;
        for (int j = k; j < n; j++) {
            double *cj = a + (size_t)j * m, d = 0;
            for (int i = k; i < m; i++) d += v[i - k] * cj[i];
            const double f = 2 * d / vn;
            for (int i = k; i < m; i++) cj[i] -= f * v[i - k];
        }
        double d = 0;
        for (int i = k; i < m; i++) d += v[i - k] * y[i];
        const double f = 2 * d / vn;
        for (int i = k; i < m; i++) y[i] -= f * v[i - k];
    }
    free(v);
    return 0;
#endif
}

/// Least squares on a column-major copy `a` (m = m0 [+ n ridge rows] rows, filled by the caller and
/// destroyed) and y (length max(m, n), b in the first m0 entries, destroyed): ridge rows, column
/// scaling, QR, back substitution dropping columns with |R_kk| ≤ 1e-13·|R_00| (as the Swift solver).
static int lstsq_cm(double *a, double *y, int m0, int n, double ridge, double *x) {
    const int m = ridge > 0 ? m0 + n : m0;
    double *scale = dgen_xmalloc(sizeof(double) * n);
    if (ridge > 0) { const double r = sqrt(ridge); for (int j = 0; j < n; j++) a[(size_t)j * m + m0 + j] = r; }
    for (int j = 0; j < n; j++) {
        double *c = a + (size_t)j * m, s = 0;
        for (int i = 0; i < m; i++) s += c[i] * c[i];
        scale[j] = s > 0 ? 1 / sqrt(s) : 1;
        for (int i = 0; i < m; i++) c[i] *= scale[j];
    }
    const int status = qr_apply(a, y, m, n);
    const int K = m < n ? m : n;
    const double rmax = fabs(a[0]);
    for (int j = 0; j < n; j++) x[j] = 0;
    for (int k = K - 1; k >= 0; k--) {
        double s = y[k];
        for (int j = k + 1; j < n; j++) s -= a[(size_t)j * m + k] * x[j];
        const double rkk = a[(size_t)k * m + k];
        x[k] = fabs(rkk) > 1e-13 * rmax ? s / rkk : 0;
    }
    for (int j = 0; j < n; j++) x[j] *= scale[j];
    free(scale);
    return status;
}

int dgen_lstsq(const double *A, int m0, int n, const double *b, double ridge, double *x) {
    if (n == 0) return 0;
    const int m = ridge > 0 ? m0 + n : m0;
    double *a = dgen_xcalloc((size_t)m * n, sizeof(double));   // column-major
    double *y = dgen_xcalloc((size_t)(m > n ? m : n), sizeof(double));
    for (int i = 0; i < m0; i++)
        for (int j = 0; j < n; j++) a[(size_t)j * m + i] = A[(size_t)i * n + j];
    memcpy(y, b, sizeof(double) * m0);
    const int status = lstsq_cm(a, y, m0, n, ridge, x);
    free(a); free(y);
    return status;
}

int dgen_damped_fit(const double *x, int N, const double *lr, const double *w, int k, double ridge,
                    double *coef, double *resid) {
    const int n = 2 * k;
    if (n == 0) { if (resid) memcpy(resid, x, sizeof(double) * N); return 0; }
    const int m = ridge > 0 ? N + n : N;
    double *a = dgen_xcalloc((size_t)m * n, sizeof(double));
    double *basis = dgen_xmalloc(sizeof(double) * (size_t)N * n);  // column-major N×n, kept for the residual
    for (int j = 0; j < k; j++)
        dgen_damped_columns(lr[j], w[j], N, basis + (size_t)(2 * j) * N, basis + (size_t)(2 * j + 1) * N);
    for (int c = 0; c < n; c++) memcpy(a + (size_t)c * m, basis + (size_t)c * N, sizeof(double) * N);
    double *y = dgen_xcalloc((size_t)(m > n ? m : n), sizeof(double));
    memcpy(y, x, sizeof(double) * N);
    const int status = lstsq_cm(a, y, N, n, ridge, coef);
    if (resid) {
        memcpy(resid, x, sizeof(double) * N);
#if defined(__APPLE__)
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, n, -1.0, basis, N, coef, 1, 1.0, resid, 1);
#else
        for (int c = 0; c < n; c++) {
            const double *col = basis + (size_t)c * N;
            for (int i = 0; i < N; i++) resid[i] -= coef[c] * col[i];
        }
#endif
    }
    free(a); free(y); free(basis);
    return status;
}

int dgen_sym_eigen(const double *A, int n, double *values, double *vectors) {
#if defined(__APPLE__)
    double *z = dgen_xmalloc(sizeof(double) * (size_t)n * n);
    double *w = dgen_xmalloc(sizeof(double) * n);
    memcpy(z, A, sizeof(double) * (size_t)n * n);  // symmetric: row- and column-major agree
    __LAPACK_int N = n, lda = n, info = 0, lwork = -1, liwork = -1, iwq;
    double wq;
    dsyevd_("V", "U", &N, z, &lda, w, &wq, &lwork, &iwq, &liwork, &info);
    lwork = (__LAPACK_int)wq; liwork = iwq;
    double *work = dgen_xmalloc(sizeof(double) * (lwork > 1 ? lwork : 1));
    __LAPACK_int *iwork = dgen_xmalloc(sizeof(__LAPACK_int) * (liwork > 1 ? liwork : 1));
    dsyevd_("V", "U", &N, z, &lda, w, work, &lwork, iwork, &liwork, &info);
    // descending; vectors[i][k] = component i of the k-th largest (z column-major, ascending)
    for (int k = 0; k < n; k++) {
        const int src = n - 1 - k;
        values[k] = w[src];
        for (int i = 0; i < n; i++) vectors[(size_t)i * n + k] = z[(size_t)src * n + i];
    }
    free(z); free(w); free(work); free(iwork);
    return (int)info;
#else
    (void)A; (void)n; (void)values; (void)vectors;
    return 1;  // the Swift Jacobi reference runs instead
#endif
}

void dgen_hankel_gram(const double *x, int rows, int cols, double *C) {
#if defined(__APPLE__)
    // H[i][a] = x[i + a] (rows × cols, row-major); C = Hᵀ H
    double *H = dgen_xmalloc(sizeof(double) * (size_t)rows * cols);
    for (int i = 0; i < rows; i++) memcpy(H + (size_t)i * cols, x + i, sizeof(double) * cols);
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, cols, rows, 1.0, H, cols, 0.0, C, cols);
    free(H);
#else
    for (int a = 0; a < cols; a++)
        for (int b = a; b < cols; b++) {
            double s = 0;
            for (int i = 0; i < rows; i++) s += x[i + a] * x[i + b];
            C[a * cols + b] = s;
        }
#endif
    dgen_mirror_upper(C, cols);
}
