// ModeSynth.c — damped (drop-law) sinusoids over whole arrays, replacing per-sample libm calls in
// the mode fits (KickAnalyzer.columns, KickVoice.renderModes, MatrixPencil). vForce (vvexp /
// vvsincos) on Apple platforms, libm loops elsewhere. Same formulas as ModeParams.envelope / phaseLaw.
#include "DGenAnalysisC.h"
#include "Internal.h"
#include <math.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

static void arrExp(double *out, const double *in, int n) {
#if defined(__APPLE__)
    vvexp(out, in, &n);
#else
    for (int i = 0; i < n; i++) out[i] = exp(in[i]);
#endif
}

static void arrSinCos(double *s, double *c, const double *in, int n) {
#if defined(__APPLE__)
    vvsincos(s, c, in, &n);
#else
    for (int i = 0; i < n; i++) { s[i] = sin(in[i]); c[i] = cos(in[i]); }
#endif
}

static void arrSin(double *out, const double *in, int n) {
#if defined(__APPLE__)
    vvsin(out, in, &n);
#else
    for (int i = 0; i < n; i++) out[i] = sin(in[i]);
#endif
}

/// env[i] = e^(−t/D) (1 − e^(−t/R)), phase[i] = 2π (F t + F G T (1 − e^(−t/T)) + F G2 T2 (1 − e^(−t/T2))),
/// t = i / sr. (A = 1, no φ.)
static void envelopeAndPhase(const DGenModeParams *m, int n, double sr, double *env, double *phase, double *tmp) {
    for (int i = 0; i < n; i++) tmp[i] = -((double)i / sr) / m->D;
    arrExp(env, tmp, n);
    if (m->R > 0) {
        for (int i = 0; i < n; i++) tmp[i] = -((double)i / sr) / m->R;
        arrExp(tmp, tmp, n);
        for (int i = 0; i < n; i++) env[i] *= 1 - tmp[i];
    }
    for (int i = 0; i < n; i++) phase[i] = m->F * ((double)i / sr);
    if (m->G != 0 && m->T > 0) {
        const double k = m->F * m->G * m->T;
        for (int i = 0; i < n; i++) tmp[i] = -((double)i / sr) / m->T;
        arrExp(tmp, tmp, n);
        for (int i = 0; i < n; i++) phase[i] += k * (1 - tmp[i]);
    }
    if (m->G2 != 0 && m->T2 > 0) {
        const double k = m->F * m->G2 * m->T2;
        for (int i = 0; i < n; i++) tmp[i] = -((double)i / sr) / m->T2;
        arrExp(tmp, tmp, n);
        for (int i = 0; i < n; i++) phase[i] += k * (1 - tmp[i]);
    }
    for (int i = 0; i < n; i++) phase[i] *= 2 * M_PI;
}

void dgen_mode_columns(const DGenModeParams *m, int n, double sr, double *c, double *s) {
    if (n <= 0) return;
    double *env = dgen_xmalloc(sizeof(double) * (size_t)n * 3), *phase = env + n, *tmp = phase + n;
    envelopeAndPhase(m, n, sr, env, phase, tmp);
    arrSinCos(s, c, phase, n);
    for (int i = 0; i < n; i++) { c[i] *= env[i]; s[i] *= env[i]; }
    free(env);
}

void dgen_mode_add(const DGenModeParams *m, double A, double phi, int n, double sr, double *y) {
    if (n <= 0 || A == 0) return;
    double *env = dgen_xmalloc(sizeof(double) * (size_t)n * 3), *phase = env + n, *tmp = phase + n;
    envelopeAndPhase(m, n, sr, env, phase, tmp);
    for (int i = 0; i < n; i++) phase[i] += phi;
    arrSin(phase, phase, n);
    for (int i = 0; i < n; i++) y[i] += A * env[i] * phase[i];
    free(env);
}

void dgen_damped_columns(double lr, double w, int n, double *c, double *s) {
    if (n <= 0) return;
    double *a = dgen_xmalloc(sizeof(double) * (size_t)n * 2), *e = a + n;
    for (int i = 0; i < n; i++) a[i] = lr * (double)i;
    arrExp(e, a, n);
    for (int i = 0; i < n; i++) a[i] = w * (double)i;
    arrSinCos(s, c, a, n);
    for (int i = 0; i < n; i++) { c[i] *= e[i]; s[i] *= e[i]; }
    free(a);
}
