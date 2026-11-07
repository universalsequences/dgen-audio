// NormalEquations.c — LM's normal equations and the joint-amplitude Gram matrix (Accelerate BLAS on
// Apple platforms, plain loops elsewhere), and a small Cholesky solve.
#include "DGenAnalysisC.h"
#include "Internal.h"
#include <math.h>
#include <string.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

void dgen_normal_equations(const double *J, const double *r, int m, int n, double *JtJ, double *Jtr) {
#if defined(__APPLE__)
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n, m, 1.0, J, n, 0.0, JtJ, n);
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1.0, J, n, r, 1, 0.0, Jtr, 1);
#else
    memset(JtJ, 0, sizeof(double) * (size_t)n * n);
    memset(Jtr, 0, sizeof(double) * n);
    for (int i = 0; i < m; i++) {
        const double *row = J + (size_t)i * n;
        for (int a = 0; a < n; a++) {
            if (row[a] == 0) continue;
            Jtr[a] += row[a] * r[i];
            for (int b = a; b < n; b++) JtJ[a * n + b] += row[a] * row[b];
        }
    }
#endif
    dgen_mirror_upper(JtJ, n);
}

int dgen_cholesky_solve(const double *A, int n, double *b) {
    double *L = dgen_xmalloc(sizeof(double) * (size_t)n * n);
    memcpy(L, A, sizeof(double) * (size_t)n * n);
    for (int j = 0; j < n; j++) {
        double d = L[j * n + j];
        for (int k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        if (!(d > 0)) { free(L); return 1; }
        d = sqrt(d);
        L[j * n + j] = d;
        for (int i = j + 1; i < n; i++) {
            double s = L[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / d;
        }
    }
    for (int i = 0; i < n; i++) {            // L y = b
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i * n + k] * b[k];
        b[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {       // Lᵀ x = y
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= L[k * n + i] * b[k];
        b[i] = s / L[i * n + i];
    }
    free(L);
    return 0;
}

void dgen_gram(const double *B, int m, int n, const double *x, double *G, double *rhs) {
#if defined(__APPLE__)
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, m, n, 1.0, B, n, 0.0, G, m);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B, n, x, 1, 0.0, rhs, 1);
#else
    for (int a = 0; a < m; a++) {
        const double *pa = B + (size_t)a * n;
        double r = 0;
        for (int i = 0; i < n; i++) r += pa[i] * x[i];
        rhs[a] = r;
        for (int b = a; b < m; b++) {
            const double *pb = B + (size_t)b * n;
            double g = 0;
            for (int i = 0; i < n; i++) g += pa[i] * pb[i];
            G[a * m + b] = g;
        }
    }
#endif
    dgen_mirror_upper(G, m);
}

double dgen_lsq_sse(const double *B, int m, int n, const double *coef, const double *x) {
    double *y = dgen_xmalloc(sizeof(double) * (size_t)n);
    memcpy(y, x, sizeof(double) * (size_t)n);
#if defined(__APPLE__)
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, -1.0, B, n, coef, 1, 1.0, y, 1);  // y = x − Bᵀc
    const double e = cblas_ddot(n, y, 1, y, 1);
#else
    for (int c = 0; c < m; c++) {
        const double *row = B + (size_t)c * n;
        for (int i = 0; i < n; i++) y[i] -= coef[c] * row[i];
    }
    double e = 0;
    for (int i = 0; i < n; i++) e += y[i] * y[i];
#endif
    free(y);
    return e;
}
