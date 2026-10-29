// C twin of VILLAIN's strand stage (villain_stems/dsp.lisp `strand` macro + strands block)
// and of the generalised E5 wire layer. float32 arithmetic like the dgen C backend.
#include <math.h>
#include <string.h>

typedef struct {
  int nstr;            // number of strands (<= 8)
  float det[8];        // per-strand tuning ratios
  float zs;            // head -> wire drive: z = 0.06*zs*head
  float gap;           // strainer gap, in units of 0.06 (VILLAIN gapv = 0.06*gap)
  float wire_hz;       // wire mode 1 (mode 2 = ratio2 * wire_hz)
  float ratio2;        // 2.41 in VILLAIN
  float t60_ms;        // mode-1 T60 in ms (VILLAIN: max(1, c_wire_ms)); mode 2 = 0.4x
  float ceil;          // overlap ceiling (0.02)
  float hp;            // leaky-integrator coefficient of the 2 DC blockers (0.1)
  float level;         // output gain (60 * c_rattle in VILLAIN)
  float push1, push2;  // contact injection into wire modes (0.769, 0.4615)
  float stiff;         // push = ov*(1 + stiff*min(ov*1000,1)), 0.8
} WireP;

void wires(const float *head, int n, const WireP *p, float *out, float *approach_out) {
  const float sr = 48000.f, twopi = 6.28318530717958647692f;
  float k1[8], k2[8];
  float wc1 = twopi * (fminf(fmaxf(p->wire_hz, 20.f), 0.2f * sr) / sr);
  float wc2 = twopi * (fminf(fmaxf(p->ratio2 * p->wire_hz, 20.f), 0.3f * sr) / sr);
  for (int s = 0; s < p->nstr; s++) { k1[s] = cosf(wc1 * p->det[s]); k2[s] = cosf(wc2 * p->det[s]); }
  float r1 = expf(-6.9077553f / (sr * 0.001f * p->t60_ms));
  float r2 = expf(-6.9077553f / (sr * 0.0004f * p->t60_ms));
  float y1[8] = {0}, y2[8] = {0}, u1[8] = {0}, u2[8] = {0};
  float cm1 = 0, cm2 = 0, gapv = 0.06f * p->gap;
  for (int i = 0; i < n; i++) {
    float z = 0.06f * p->zs * head[i];
    float approach = 0;
    for (int s = 0; s < p->nstr; s++) {
      float w1 = 2 * r1 * k1[s] * y1[s] - r1 * r1 * y2[s];
      float w2 = 2 * r2 * k2[s] * u1[s] - r2 * r2 * u2[s];
      float w = w1 + 0.5f * w2;
      float ov = fminf(fmaxf(z - w - gapv, 0.f), p->ceil);
      float push = ov * (1 + p->stiff * fminf(ov * 1000.f, 1.f));
      y2[s] = y1[s]; y1[s] = w1 + push * p->push1;
      u2[s] = u1[s]; u1[s] = w2 + push * p->push2;
      approach += ov;
    }
    float cml1 = cm1 + p->hp * (approach - cm1); cm1 = cml1;
    float cma = approach - cml1;
    float cml2 = cm2 + p->hp * (cma - cm2); cm2 = cml2;
    out[i] = p->level * (cma - cml2);
    if (approach_out) approach_out[i] = approach;
  }
}

// VILLAIN TPT SVF steps, (3,2) Pade prewarp
static float svf_g(float hz) { float x = 3.14159265358979f * fminf(fmaxf(hz, 10.f), 0.33f * 48000.f) / 48000.f;
  return x * (15 - x * x) / (15 - 6 * x * x); }
void svf(const float *x, int n, float hz, float kk, int hp, float *y) {
  float gk = svf_g(hz), s1 = 0, s2 = 0;
  float a1 = 1 / (1 + gk * (gk + kk)), a2 = gk * a1, a3 = gk * a2;
  for (int i = 0; i < n; i++) {
    float v3 = x[i] - s2, v1 = a1 * s1 + a2 * v3, v2 = s2 + a2 * s1 + a3 * v3;
    s1 = 2 * v1 - s1; s2 = 2 * v2 - s2;
    y[i] = hp ? (x[i] - kk * v1 - v2) : v2;
  }
}
