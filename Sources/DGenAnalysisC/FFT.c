// FFT.c — in-place complex FFT by vDSP (DGenAnalysis FFT.transform): forward e^{-i2πkn/N},
// inverse scaled by 1/N. One setup per size, created once and shared read-only by all threads.
// Without Accelerate it returns nonzero and the Swift radix-2 reference runs.
#include "DGenAnalysisC.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <pthread.h>

#define MAX_LOG2 26
static FFTSetupD setups[MAX_LOG2 + 1];
static pthread_mutex_t setupLock = PTHREAD_MUTEX_INITIALIZER;

static FFTSetupD setupFor(int log2n) {
    FFTSetupD s = __atomic_load_n(&setups[log2n], __ATOMIC_ACQUIRE);
    if (s) return s;
    pthread_mutex_lock(&setupLock);
    s = setups[log2n];
    if (!s) {
        s = vDSP_create_fftsetupD(log2n, kFFTRadix2);
        __atomic_store_n(&setups[log2n], s, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&setupLock);
    return s;
}

int dgen_fft(double *re, double *im, int n, int inverse) {
    if (n <= 1) return 0;
    if (n & (n - 1)) return 1;
    const int log2n = __builtin_ctz((unsigned)n);
    if (log2n > MAX_LOG2) return 1;
    FFTSetupD s = setupFor(log2n);
    if (!s) return 1;
    DSPDoubleSplitComplex z = {re, im};
    vDSP_fft_zipD(s, &z, 1, log2n, inverse ? kFFTDirection_Inverse : kFFTDirection_Forward);
    if (inverse) {
        const double sc = 1.0 / n;
        vDSP_vsmulD(re, 1, &sc, re, 1, n);
        vDSP_vsmulD(im, 1, &sc, im, 1, n);
    }
    return 0;
}
#else
int dgen_fft(double *re, double *im, int n, int inverse) {
    (void)re; (void)im; (void)n; (void)inverse;
    return 1;
}
#endif
