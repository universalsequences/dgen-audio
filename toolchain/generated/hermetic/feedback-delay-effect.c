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
float t1_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t2_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t3_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t4_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t5_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t22_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t34_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t35_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t38_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t48_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
// Memory size required: 88005 floats

void setParamValue(int cellId, float val) {
  //memory[cellId] = val;
}

void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state, void * restrict buffers, float hostSampleRate) {
  int frameCount = nframes;  // Use audiograph frame count parameter
  int i = 0;
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(1.0f);
  float32x4_t c3 = vdupq_n_f32(88000.0f);
  float *memory = (float*)state;
  int voiceIndex = 0;
  if (voiceIndex < 0) voiceIndex = 0;
  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;
  int _scratchBase = voiceIndex * SCRATCH_STRIDE;
  float *t1 = t1_g + _scratchBase;
  float *t2 = t2_g + _scratchBase;
  float *t3 = t3_g + _scratchBase;
  float *t4 = t4_g + _scratchBase;
  float *t5 = t5_g + _scratchBase;
  float *t22 = t22_g + _scratchBase;
  float *t34 = t34_g + _scratchBase;
  float *t35 = t35_g + _scratchBase;
  float *t38 = t38_g + _scratchBase;
  float *t48 = t48_g + _scratchBase;
  /* frameCount available as function parameter */
  for (int i = 0; i < frameCount; i += 4) {
    /* t4 declared globally */
    /* t3 declared globally */
    /* t2 declared globally */
    /* t1 declared globally */
    float32x4_t simd1 = vdupq_n_f32(memory[0 + (int)0.0]); vst1q_f32(t1 + i, simd1);
    float32x4_t simd2 = vdupq_n_f32(memory[1 + (int)0.0]); vst1q_f32(t2 + i, simd2);
    float32x4_t simd3 = vdupq_n_f32(memory[2 + (int)0.0]); vst1q_f32(t3 + i, simd3);
    float32x4_t simd4 = vld1q_f32(in[0] + i); vst1q_f32(t4 + i, simd4);
  }
  for (int i = 0; i < frameCount; i += 1) {
    /* t5 declared globally */
    t5[i] = memory[88004];
    float t6 = t5[i] + 1.0;
    float t7 = 0.0 > 0.0f ? 0.0 : t6;
    float t8 = t7;
    float t9 = (t8 / 88000.0f);
    float t10 = floorf(t9);
    float t11 = t10 * 88000.0;
    float t12 = t7 - t11;
    memory[88004] = t12;
    float t14 = t12 >= 88000.0;
    if (t14) {
      float t16 = t12 - 88000.0;
      memory[88004] = t16;
    }
    if (0.0) {
      memory[88004] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd5 = vld1q_f32(t5 + i); /* extra */
    t5[i] = t5[i];
    float32x4_t simd1 = vld1q_f32(t1 + i); /* extra */
    t1[i] = t1[i];
    /* t38 declared globally */
    /* t35 declared globally */
    /* t34 declared globally */
    /* t22 declared globally */
    float32x4_t simd22 = vrndmq_f32(simd5); vst1q_f32(t22 + i, simd22);
    float32x4_t simd23 = vsubq_f32(simd22, simd1);
    float32x4_t simd24 = vdivq_f32(simd23, vdupq_n_f32(88000.0f));
    float32x4_t simd25 = vrndmq_f32(simd24);
    float32x4_t simd26 = vmulq_f32(simd25, c3);
    float32x4_t simd27 = vsubq_f32(simd23, simd26);
    float32x4_t simd28 = vbslq_f32(vcgeq_f32(simd27, c3), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f));
    float32x4_t simd29 = vsubq_f32(simd27, c3);
    float32x4_t simd30 = vbslq_f32(vcgtq_f32(simd28, vdupq_n_f32(0.0f)), simd29, simd27);
    float32x4_t simd31 = vbslq_f32(vcltq_f32(simd30, c1), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f));
    float32x4_t simd32 = vaddq_f32(simd30, c3);
    float32x4_t simd33 = vbslq_f32(vcgtq_f32(simd31, vdupq_n_f32(0.0f)), simd32, simd30);
    float32x4_t simd34 = vrndmq_f32(simd33); vst1q_f32(t34 + i, simd34);
    float32x4_t simd35 = vsubq_f32(simd33, simd34); vst1q_f32(t35 + i, simd35);
    float32x4_t simd36 = vaddq_f32(simd34, c2);
    float32x4_t simd37 = vbslq_f32(vcgeq_f32(simd36, c3), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f));
    float32x4_t simd38 = vbslq_f32(vcgtq_f32(simd37, vdupq_n_f32(0.0f)), c1, simd36); vst1q_f32(t38 + i, simd38);
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd38 = vld1q_f32(t38 + i); /* extra */
    t38[i] = t38[i];
    float32x4_t simd35 = vld1q_f32(t35 + i); /* extra */
    t35[i] = t35[i];
    float32x4_t simd34 = vld1q_f32(t34 + i); /* extra */
    t34[i] = t34[i];
    float32x4_t simd22 = vld1q_f32(t22 + i); /* extra */
    t22[i] = t22[i];
    float32x4_t simd4 = vld1q_f32(t4 + i); /* extra */
    t4[i] = t4[i];
    float32x4_t simd2 = vld1q_f32(t2 + i); /* extra */
    t2[i] = t2[i];
    /* t48 declared globally */
    float t39 = memory[3];
    float t40 = t2[i] * t39;
    float t41 = t4[i] + t40;
    memory[4 + (int)t22[i]] = t41;
    float t43 = memory[4 + (isfinite((int) t34[i]) ? (int) t34[i] : 0)];
    float t44 = memory[4 + (isfinite((int) t38[i]) ? (int) t38[i] : 0)];
    float t45 = 1.0 - t35[i];
    float t46 = t43 * t45;
    float t47 = t44 * t35[i];
    t48[i] = t46 + t47;
    memory[3] = t48[i];
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd48 = vld1q_f32(t48 + i); /* extra */
    t48[i] = t48[i];
    float32x4_t simd4 = vld1q_f32(t4 + i); /* extra */
    t4[i] = t4[i];
    float32x4_t simd3 = vld1q_f32(t3 + i); /* extra */
    t3[i] = t3[i];
    float32x4_t simd50 = vsubq_f32(c2, simd3);
    float32x4_t simd51 = vmulq_f32(simd4, simd50);
    float32x4_t simd52 = vmulq_f32(simd48, simd3);
    float32x4_t simd53 = vaddq_f32(simd51, simd52);
    vst1q_f32(out[0] + i, sanitize_out_f32x4(simd53));
  }
}