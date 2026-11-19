// Internal.h — helpers shared by the DGenAnalysisC kernels (not part of the public header).
#ifndef DGEN_ANALYSIS_C_INTERNAL_H
#define DGEN_ANALYSIS_C_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>

/// malloc / calloc that abort on failure (the analysis has no way to continue without memory).
static inline void *dgen_xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "DGenAnalysisC: out of memory (%zu bytes)\n", bytes); abort(); }
    return p;
}
static inline void *dgen_xcalloc(size_t count, size_t size) {
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) { fprintf(stderr, "DGenAnalysisC: out of memory (%zu x %zu bytes)\n", count, size); abort(); }
    return p;
}

/// Copies the upper triangle of a row-major n×n matrix into its lower triangle.
static inline void dgen_mirror_upper(double *A, int n) {
    for (int a = 0; a < n; a++)
        for (int b = 0; b < a; b++) A[a * n + b] = A[b * n + a];
}

#endif
