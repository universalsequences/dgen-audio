#include "dgen_runtime.h"

static inline float32x4_t vfmodq_f32(float32x4_t a, float32x4_t b) {
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

enum { VOICE_COUNT = 1, SCRATCH_STRIDE = 512 };
static float t11_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
static float t12_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
static float t29_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
static float t30_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
static float t31_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
// Memory size required: 3700 floats

void dgen_set_param_value_v1(int32_t cell_id, float value) {
  (void)cell_id;
  (void)value;
  //memory[cellId] = val;
}

void dgen_process_v1(const float * const *in, float * const *out, uint32_t nframes, void *state, const DGenProcessContextV1 *context, const DGenHostServicesV1 *host) {
  int frameCount = (int)nframes;
  float hostSampleRate = (context != NULL && context->abi_version == DGEN_ABI_VERSION_V1 && context->struct_size >= sizeof(DGenProcessContextV1)) ? context->sample_rate : 0.0f;
  int i = 0;
  float32x4_t c1 = vdupq_n_f32(1.0f);
  float32x4_t c2 = vdupq_n_f32(0.0f);
  float32x4_t c3 = vdupq_n_f32(79.0f);
  float32x4_t c4 = vdupq_n_f32(8.0f);
  float32x4_t c5 = vdupq_n_f32(16.0f);
  float32x4_t c6 = vdupq_n_f32(4.0f);
  float32x4_t c7 = vdupq_n_f32(0.0625f);
  float32x4_t c8 = vdupq_n_f32(0.5f);
  float *memory = (float*)state;
  int voiceIndex = 0;
  if (voiceIndex < 0) voiceIndex = 0;
  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;
  int _scratchBase = voiceIndex * SCRATCH_STRIDE;
  float *t11 = t11_g + _scratchBase;
  float *t12 = t12_g + _scratchBase;
  float *t29 = t29_g + _scratchBase;
  float *t30 = t30_g + _scratchBase;
  float *t31 = t31_g + _scratchBase;
  /* frameCount available as function parameter */
  for (int i = 0; i < frameCount; i += 4) {
    /* t11 declared globally */
    float32x4_t simd11 = vld1q_f32(in[0] + i); vst1q_f32(t11 + i, simd11);
  }
  for (int i = 0; i < frameCount; i += 1) {
    /* t12 declared globally */
    t12[i] = memory[3423];
    float t13 = t12[i] + 1.0;
    float t14 = 0.0 > 0.0f ? 0.0 : t13;
    float t15 = t14;
    float t16 = (t15 / 79.0f);
    float t17 = floorf(t16);
    float t18 = t17 * 79.0;
    float t19 = t14 - t18;
    memory[3423] = t19;
    float t21 = t19 >= 79.0;
    if (t21) {
      float t23 = t19 - 79.0;
      memory[3423] = t23;
    }
    if (0.0) {
      memory[3423] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd12 = vld1q_f32(t12 + i); /* extra */
    t12[i] = t12[i];
    /* t29 declared globally */
    float32x4_t simd29 = vrndmq_f32(simd12); vst1q_f32(t29 + i, simd29);
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
    float32x4_t simd11 = vld1q_f32(t11 + i); /* extra */
    t11[i] = t11[i];
    /* t30 declared globally */
    memory[3344 + (int)t29[i]] = t11[i];
    /* t31 declared globally */
    t31[i] = memory[3424];
    float t32 = t31[i] + 1.0;
    float t33 = 0.0 > 0.0f ? 0.0 : t32;
    float t34 = t33;
    float t35 = (t34 / 8.0f);
    float t36 = floorf(t35);
    float t37 = t36 * 8.0;
    float t38 = t33 - t37;
    memory[3424] = t38;
    float t40 = t38 >= 8.0;
    if (t40) {
      float t42 = t38 - 8.0;
      memory[3424] = t42;
    }
    if (0.0) {
      memory[3424] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd31 = vld1q_f32(t31 + i); /* extra */
    t31[i] = t31[i];
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd30 = vld1q_f32(t30 + i); /* extra */
    t30[i] = t30[i];
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      for (int t48 = 0; t48 < 16; t48++) {
        /* [1mUOp[0m(op: [38;5;51mreshape[0m([16]), value: empty) */
      }
      for (int t49 = 0; t49 < 16; t49++) {
        int t50 = t49;
        int t51 = t50;
        int t52 = t51 / 16;
        int t53 = t52 * 16;
        int t54 = t51 - t53;
        float t55 = (int)t29[i];
        int t56 = t55 - 16;
        int t57 = t56 + 1;
        int t58 = t57 + t54;
        int t59 = t58 + 79;
        int t60 = t59 % 79;
        int t61 = t52 * 79;
        int t62 = t61 + t60;
        float t63 = memory[3344 + t62];
        float t64 = memory[32 + (isfinite(t49) ? (int) t49 : 0)];
        float t65 = t63 * t64;
        int t66 = i;
        int t67 = t66 * 16;
        int t68 = t67 + t49;
        memory[272 + t68] = t65;
      }
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      for (int simd70 = 0; simd70 < 16; simd70+=4) {
        int t71 = i;
        int t72 = t71 * 16;
        int t73 = t72 + simd70;
        float32x4_t simd74 = vld1q_f32(&memory[272 + (int)t73]);
        vst1q_f32(&memory[176 + (int)simd70], simd74);
        vst1q_f32(&memory[192 + (int)simd70], c2);
      }
      {
  static DGenFFTSetupV1 _dgen_fft_setup_4 = NULL;
  if (host != NULL &&
      host->abi_version == DGEN_ABI_VERSION_V1 &&
      host->struct_size >= sizeof(DGenHostServicesV1) &&
      host->fft_setup_create_fn != NULL &&
      host->fft_forward_fn != NULL) {
    if (_dgen_fft_setup_4 == NULL) {
      _dgen_fft_setup_4 = host->fft_setup_create_fn(4u);
    }
    if (_dgen_fft_setup_4 != NULL) {
      host->fft_forward_fn(
        _dgen_fft_setup_4,
        &memory[176],
        &memory[192],
        4u);
    }
  }
}
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      float t79 = memory[3681 + (isfinite(0.0) ? (int) 0.0 : 0)];
      float t80 = (int)t79;
      int t81 = t80 * 16;
      int t82 = t80 + 4;
      int t83 = t82 * 16;
      for (int simd84 = 0; simd84 < 16; simd84+=4) {
        float32x4_t simd85 = vld1q_f32(&memory[176 + (int)simd84]);
        float32x4_t simd86 = vld1q_f32(&memory[192 + (int)simd84]);
        int t87 = t81 + simd84;
        vst1q_f32(&memory[3425 + (int)t87], simd85);
        int t89 = t83 + simd84;
        vst1q_f32(&memory[3425 + (int)t89], simd85);
        int t91 = t81 + simd84;
        vst1q_f32(&memory[3553 + (int)t91], simd86);
        int t93 = t83 + simd84;
        vst1q_f32(&memory[3553 + (int)t93], simd86);
      }
      {
  int _dgen_p = (int)memory[3681];
  memset(&memory[208], 0, 16 * sizeof(float));
  memset(&memory[224], 0, 16 * sizeof(float));
  if (host != NULL &&
      host->abi_version == DGEN_ABI_VERSION_V1 &&
      host->struct_size >= sizeof(DGenHostServicesV1) &&
      host->complex_multiply_accumulate_fn != NULL) {
    for (int _dgen_k = 0; _dgen_k < 4; _dgen_k++) {
      int _dgen_ring_off = (_dgen_p + 4 - _dgen_k) * 16;
      int _dgen_ir_off = _dgen_k * 16;
      host->complex_multiply_accumulate_fn(
        &memory[3425 + _dgen_ring_off],
        &memory[3553 + _dgen_ring_off],
        &memory[48 + _dgen_ir_off],
        &memory[112 + _dgen_ir_off],
        &memory[208],
        &memory[224],
        16u);
    }
  }
}
      int t97 = t80 + 1;
      int t98 = t97 >= 4;
      int t99 = t98 > 0 ? 0 : t97;
      float t100 = (float)t99;
      memory[3681 + (int)0.0] = t100;
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      for (int simd102 = 0; simd102 < 16; simd102+=4) {
        float32x4_t simd103 = vld1q_f32(&memory[208 + (int)simd102]);
        float32x4_t simd104 = vld1q_f32(&memory[224 + (int)simd102]);
        vst1q_f32(&memory[240 + (int)simd102], simd103);
        vst1q_f32(&memory[256 + (int)simd102], simd104);
      }
      {
  static DGenFFTSetupV1 _dgen_fft_setup_4 = NULL;
  if (host != NULL &&
      host->abi_version == DGEN_ABI_VERSION_V1 &&
      host->struct_size >= sizeof(DGenHostServicesV1) &&
      host->fft_setup_create_fn != NULL &&
      host->fft_inverse_fn != NULL) {
    if (_dgen_fft_setup_4 == NULL) {
      _dgen_fft_setup_4 = host->fft_setup_create_fn(4u);
    }
    if (_dgen_fft_setup_4 != NULL) {
      host->fft_inverse_fn(
        _dgen_fft_setup_4,
        &memory[240],
        &memory[256],
        4u);
    }
  }
}
      for (int simd109 = 0; simd109 < 16; simd109+=4) {
        float32x4_t simd110 = vld1q_f32(&memory[240 + (int)simd109]);
        float32x4_t simd111 = vmulq_f32(simd110, c7);
        vst1q_f32(&memory[240 + (int)simd109], simd111);
      }
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      for (int t9 = 0; t9 < 16; t9+=1) {
        float t114 = memory[240 + t9];
        float t115 = memory[32 + t9];
        float t116 = t114 * t115;
        int t117 = i;
        int t118 = t117 * 16;
        int t119 = t118 + t9;
        memory[1296 + t119] = t116;
      }
    }
    /* skip scalar load */
    if (t31[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
      for (int t10 = 0; t10 < 16; t10+=1) {
        int t122 = i;
        int t123 = t122 * 16;
        int t124 = t123 + t10;
        float t125 = memory[1296 + t124];
        float t126 = t125 * 0.5;
        int t127 = i;
        int t128 = t127 * 16;
        int t129 = t128 + t10;
        memory[2320 + t129] = t126;
      }
    }
    float32x4_t simd29 = vld1q_f32(t29 + i); /* extra */
    t29[i] = t29[i];
    float t132 = memory[3698 + (isfinite(0.0) ? (int) 0.0 : 0)];
    float t133 = memory[3699 + (isfinite(0.0) ? (int) 0.0 : 0)];
    float t134 = t133 == 0.0;
    if (t134) {
      for (int t136 = 0; t136 < 16; t136++) {
        int t137 = i;
        int t138 = t137 * 16;
        int t139 = t138 + t136;
        float t140 = memory[2320 + t139];
        float t141 = t132 + t136;
        float t142 = t141 >= 16.0;
        float t143 = t141 - 16.0;
        float t144 = t142 > 0.0f ? t143 : t141;
        float t145 = (int)t144;
        float t146 = memory[3682 + (isfinite(t145) ? (int) t145 : 0)];
        float t147 = t146 + t140;
        memory[3682 + (int)t145] = t147;
      }
    }
    float t151 = (int)t132;
    float t152 = memory[3682 + (isfinite(t151) ? (int) t151 : 0)];
    float t153 = (int)t132;
    memory[3682 + (int)t153] = 0.0;
    float t155 = t132 + 1.0;
    float t156 = t155 >= 16.0;
    float t157 = t156 > 0.0f ? 0.0 : t155;
    memory[3698 + (int)0.0] = t157;
    float t159 = t133 + 1.0;
    float t160 = t159 >= 8.0;
    float t161 = t160 > 0.0f ? 0.0 : t159;
    memory[3699 + (int)0.0] = t161;
    out[0][i] = dgen_sanitize_f32(t152);
  }
}