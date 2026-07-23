#include "phase1_compat.h"

// Enable profiling only when DGEN_PROFILE is defined by build flags

float32x4_t vfmodq_f32(float32x4_t a, float32x4_t b) {
  // a - floor(a / b) * b  (faster and correct for positive ranges)
  float32x4_t q = vdivq_f32(a, b);
  float32x4_t q_floor = vrndmq_f32(q);  // floor
  return vsubq_f32(a, vmulq_f32(b, q_floor));
}

static inline uint32x4_t mask_nz_f32(float32x4_t x) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    // eq0 = (x == 0.0f)
    uint32x4_t eq0  = vceqq_f32(x, zero);
    // non-zero mask = bitwise NOT of eq0
    return vmvnq_u32(eq0);
}

static inline float32x4_t boolmask_to_float(uint32x4_t m) {
    float32x4_t ones  = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    // Select 1.0f where mask bits are 1, else 0.0f
    return vbslq_f32(m, ones, zeros);
}

static inline float32x4_t simd_and_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = vandq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

static inline float32x4_t simd_or_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = vorrq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

static inline float32x4_t simd_xor_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = veorq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

// Replace NaN/Inf with 0 so a single bad node can't poison the whole graph.
static inline float sanitize_out_f32(float v) {
    return isfinite(v) ? v : 0.0f;
}
static inline float32x4_t sanitize_out_f32x4(float32x4_t v) {
    uint32x4_t finite = vcltq_f32(vabsq_f32(v), vdupq_n_f32(INFINITY));
    return vbslq_f32(finite, v, vdupq_n_f32(0.0f));
}

const int VOICE_COUNT = 1;
const int SCRATCH_STRIDE = 512;
float t2_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t3_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t4_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t7_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t27_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
// Memory size required: 1024 floats

void setParamValue(int cellId, float val) {
  //memory[cellId] = val;
}

void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state, void * restrict buffers, float hostSampleRate) {
  int frameCount = nframes;  // Use audiograph frame count parameter
  int i = 0;
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(1.0f);
  float32x4_t c3 = vdupq_n_f32(16.0f);
  float *memory = (float*)state;
  int voiceIndex = 0;
  if (voiceIndex < 0) voiceIndex = 0;
  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;
  int _scratchBase = voiceIndex * SCRATCH_STRIDE;
  float *t2 = t2_g + _scratchBase;
  float *t3 = t3_g + _scratchBase;
  float *t4 = t4_g + _scratchBase;
  float *t7 = t7_g + _scratchBase;
  float *t27 = t27_g + _scratchBase;
  /* frameCount available as function parameter */
  /* t4 declared globally */
  /* t3 declared globally */
  /* t2 declared globally */
  float32x4_t simd2 = vdupq_n_f32(memory[32 + (int)0.0]); vst1q_f32(t2 + i, simd2);
  float32x4_t simd3 = vdupq_n_f32(memory[33 + (int)0.0]); vst1q_f32(t3 + i, simd3);
  float32x4_t simd4 = vdupq_n_f32(memory[34 + (int)0.0]); vst1q_f32(t4 + i, simd4);
  for (int i = 0; i < frameCount; i += 1) {
    t2[i] = t2[0];
    /* t7 declared globally */
    float t5 = hostSampleRate;
    float t6 = t2[0] / t5;
    t7[i] = memory[35];
    float t8 = t7[i] + t6;
    float t9 = 0.0 > 0.0f ? 0.0 : t8;
    float t10 = t9;
    float t11 = t10;
    float t12 = floorf(t11);
    float t13 = t12;
    float t14 = t9 - t13;
    memory[35] = t14;
    float t16 = t14 >= 1.0;
    if (t16) {
      float t18 = t14 - 1.0;
      memory[35] = t18;
    }
    if (0.0) {
      memory[35] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd7 = vld1q_f32(t7 + i); /* extra */
    t7[i] = t7[i];
    /* t27 declared globally */
    float32x4_t simd24 = simd7;
    float32x4_t simd25 = vsubq_f32(simd24, vrndmq_f32(simd24));
    float32x4_t simd26 = simd25;
    float32x4_t simd27 = vmulq_f32(simd26, c3); vst1q_f32(t27 + i, simd27);
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd27 = vld1q_f32(t27 + i); /* extra */
    t27[i] = t27[i];
    t4[i] = t4[0];
    t3[i] = t3[0];
    float t28 = (t27[i] - floorf(t27[i] / 16.0f) * 16.0f);
    float t29 = t28 < 0.0;
    float t30 = t28 + 16.0;
    float t31 = t29 > 0.0f ? t30 : t28;
    float t32 = fminf(t3[0], 1.0);
    float t33 = fmaxf(0.0, t32);
    float t34 = floorf(t33);
    float t35 = t33 - t34;
    float t36 = t34 + 1.0;
    float t37 = fminf(t36, 1.0);
    float t38 = floorf(t31);
    float t39 = t31 - t38;
    float t40 = t38 + 1.0;
    float t41 = t40 >= 16.0;
    float t42 = t38 + 1.0;
    float t43 = t41 > 0.0f ? 0.0 : t42;
    float t44 = 16.0 * t34;
    float t45 = 16.0 * t37;
    float t46 = t44 + t38;
    float t47 = t44 + t43;
    float t48 = t45 + t38;
    float t49 = t45 + t43;
    float t50 = (int)t46;
    float t51 = memory[0 + (isfinite((int) t50) ? (int) t50 : 0)];
    float t52 = (int)t47;
    float t53 = memory[0 + (isfinite((int) t52) ? (int) t52 : 0)];
    float t54 = (int)t48;
    float t55 = memory[0 + (isfinite((int) t54) ? (int) t54 : 0)];
    float t56 = (int)t49;
    float t57 = memory[0 + (isfinite((int) t56) ? (int) t56 : 0)];
    float t58 = 1.0 - t39;
    float t59 = t51 * t58;
    float t60 = t53 * t39;
    float t61 = t59 + t60;
    float t62 = 1.0 - t39;
    float t63 = t55 * t62;
    float t64 = t57 * t39;
    float t65 = t63 + t64;
    float t66 = 1.0 - t35;
    float t67 = t61 * t66;
    float t68 = t65 * t35;
    float t69 = t67 + t68;
    float t70 = t4[0] * t69;
    out[0][i] = sanitize_out_f32(t70);
  }
}