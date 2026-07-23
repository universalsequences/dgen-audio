#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

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
float t6_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
// Memory size required: 1024 floats

void setParamValue(int cellId, float val) {
  //memory[cellId] = val;
}

void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state, void * restrict buffers, float hostSampleRate) {
  int frameCount = nframes;  // Use audiograph frame count parameter
  int i = 0;
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(1.0f);
  float32x4_t c3 = vdupq_n_f32(6.283185f);
  float32x4_t c4 = vdupq_n_f32(12.56637f);
  float32x4_t c5 = vdupq_n_f32(0.125f);
  float *memory = (float*)state;
  int voiceIndex = 0;
  if (voiceIndex < 0) voiceIndex = 0;
  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;
  int _scratchBase = voiceIndex * SCRATCH_STRIDE;
  float *t1 = t1_g + _scratchBase;
  float *t2 = t2_g + _scratchBase;
  float *t3 = t3_g + _scratchBase;
  float *t6 = t6_g + _scratchBase;
  /* frameCount available as function parameter */
  /* t3 declared globally */
  /* t2 declared globally */
  /* t1 declared globally */
  float32x4_t simd1 = vdupq_n_f32(memory[0 + (int)0.0]); vst1q_f32(t1 + i, simd1);
  float32x4_t simd2 = vdupq_n_f32(memory[1 + (int)0.0]); vst1q_f32(t2 + i, simd2);
  float32x4_t simd3 = vdupq_n_f32(memory[2 + (int)0.0]); vst1q_f32(t3 + i, simd3);
  for (int i = 0; i < frameCount; i += 1) {
    t1[i] = t1[0];
    /* t6 declared globally */
    float t4 = hostSampleRate;
    float t5 = t1[0] / t4;
    t6[i] = memory[3];
    float t7 = t6[i] + t5;
    float t8 = 0.0 > 0.0f ? 0.0 : t7;
    float t9 = t8;
    float t10 = t9;
    float t11 = floorf(t10);
    float t12 = t11;
    float t13 = t8 - t12;
    memory[3] = t13;
    float t15 = t13 >= 1.0;
    if (t15) {
      float t17 = t13 - 1.0;
      memory[3] = t17;
    }
    if (0.0) {
      memory[3] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd6 = vld1q_f32(t6 + i); /* extra */
    t6[i] = t6[i];
    t3[i] = t3[0];
    t2[i] = t2[0];
    float32x4_t simd23 = vmulq_f32(simd6, c3);
    float32x4_t simd24 = vsinf(simd23);
    float32x4_t simd25 = vmulq_f32(simd6, c4);
    float32x4_t simd26 = vcosf(simd25);
    float32x4_t simd27 = vmulq_f32(simd26, c5);
    float32x4_t simd28 = vaddq_f32(simd24, simd27);
    float32x4_t simd29 = vmulq_f32(simd2, simd28);
    float32x4_t simd30 = vtanhf(simd29);
    float32x4_t simd31 = vmulq_f32(simd3, simd30);
    vst1q_f32(out[0] + i, sanitize_out_f32x4(simd31));
  }
}