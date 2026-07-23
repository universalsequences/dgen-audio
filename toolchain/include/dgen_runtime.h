#ifndef DGEN_RUNTIME_H
#define DGEN_RUNTIME_H

/*
 * DGen generated-code runtime ABI, version 1.
 *
 * This header is a public, versioned contract between generated DGen dylibs
 * and their host. It deliberately depends only on Clang resource headers and
 * the small libSystem symbol allowlist owned by DGen.
 */

#include <arm_neon.h>
#include <stdint.h>

#define DGEN_ABI_VERSION_V1 1u
#define DGEN_RUNTIME_HEADER_VERSION 1u
#define DGEN_EXPORT __attribute__((visibility("default")))

typedef __SIZE_TYPE__ size_t;

#ifndef NULL
#define NULL ((void *)0)
#endif

#define INFINITY (__builtin_inff())
#define M_LOG10E 0.43429448190325182765

/* Direct libSystem/libm surface emitted by CRenderer. */
extern float sinf(float);
extern float cosf(float);
extern float tanf(float);
extern float atanf(float);
extern float atan2f(float, float);
extern float tanhf(float);
extern float expf(float);
extern float logf(float);
extern float log10f(float);
extern float sqrtf(float);
extern float powf(float, float);
extern float fmodf(float, float);
extern float fminf(float, float);
extern float fmaxf(float, float);
extern float floorf(float);
extern float ceilf(float);
extern float roundf(float);
extern float copysignf(float, float);
extern float fabsf(float);

extern void *memcpy(void *restrict destination, const void *restrict source, size_t count);
extern void *memset(void *destination, int value, size_t count);

/*
 * Do not implement finite classification with __builtin_isfinite here.
 * Under DGen's explicitly selected finite-math optimization policy, Clang may
 * legally fold such a classification away. Inspecting the IEEE-754 exponent
 * bits preserves boundary containment under those flags.
 */
static inline int dgen_isfinite_f32(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

#define isfinite(value) dgen_isfinite_f32((float)(value))

static inline float dgen_sanitize_f32(float value) {
  return dgen_isfinite_f32(value) ? value : 0.0f;
}

static inline float32x4_t dgen_sanitize_f32x4(float32x4_t value) {
  uint32x4_t exponent = vandq_u32(
    vreinterpretq_u32_f32(value),
    vdupq_n_u32(UINT32_C(0x7f800000)));
  uint32x4_t finite = vmvnq_u32(
    vceqq_u32(exponent, vdupq_n_u32(UINT32_C(0x7f800000))));
  return vbslq_f32(finite, value, vdupq_n_f32(0.0f));
}

/*
 * Four-lane math names are part of the renderer/runtime-header boundary, not
 * the host-service ABI. Phase 2's measured lowering keeps these calls direct
 * and inline. The selected implementations are recorded in
 * docs/vector-math-lowering.md.
 */
#define DGEN_UNARY_VECTOR_WRAPPER(vector_name, scalar_name)                 \
  static inline float32x4_t vector_name(float32x4_t value) {               \
    return (float32x4_t){                                                    \
      scalar_name(vgetq_lane_f32(value, 0)),                                 \
      scalar_name(vgetq_lane_f32(value, 1)),                                 \
      scalar_name(vgetq_lane_f32(value, 2)),                                 \
      scalar_name(vgetq_lane_f32(value, 3))                                  \
    };                                                                       \
  }

#define DGEN_BINARY_VECTOR_WRAPPER(vector_name, scalar_name)                \
  static inline float32x4_t vector_name(float32x4_t lhs, float32x4_t rhs) { \
    return (float32x4_t){                                                    \
      scalar_name(vgetq_lane_f32(lhs, 0), vgetq_lane_f32(rhs, 0)),          \
      scalar_name(vgetq_lane_f32(lhs, 1), vgetq_lane_f32(rhs, 1)),          \
      scalar_name(vgetq_lane_f32(lhs, 2), vgetq_lane_f32(rhs, 2)),          \
      scalar_name(vgetq_lane_f32(lhs, 3), vgetq_lane_f32(rhs, 3))           \
    };                                                                       \
  }

DGEN_UNARY_VECTOR_WRAPPER(vsinf, sinf)
DGEN_UNARY_VECTOR_WRAPPER(vcosf, cosf)
DGEN_UNARY_VECTOR_WRAPPER(vtanf, tanf)
DGEN_UNARY_VECTOR_WRAPPER(vatanf, atanf)
DGEN_UNARY_VECTOR_WRAPPER(vtanhf, tanhf)
DGEN_UNARY_VECTOR_WRAPPER(vexpf, expf)
DGEN_UNARY_VECTOR_WRAPPER(vlogf, logf)
DGEN_UNARY_VECTOR_WRAPPER(vsqrtf, sqrtf)
DGEN_BINARY_VECTOR_WRAPPER(vatan2f, atan2f)
DGEN_BINARY_VECTOR_WRAPPER(vpowf, powf)

#undef DGEN_UNARY_VECTOR_WRAPPER
#undef DGEN_BINARY_VECTOR_WRAPPER

typedef void *DGenFFTSetupV1;

typedef struct DGenProcessContextV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  float sample_rate;
  uint32_t reserved;
} DGenProcessContextV1;

typedef struct DGenHostServicesV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  DGenFFTSetupV1 (*fft_setup_create_fn)(uint32_t log2_size);
  void (*fft_forward_fn)(
    DGenFFTSetupV1 setup,
    float *real,
    float *imaginary,
    uint32_t log2_size);
  void (*fft_inverse_fn)(
    DGenFFTSetupV1 setup,
    float *real,
    float *imaginary,
    uint32_t log2_size);
  void (*complex_multiply_accumulate_fn)(
    const float *lhs_real,
    const float *lhs_imaginary,
    const float *rhs_real,
    const float *rhs_imaginary,
    float *accumulator_real,
    float *accumulator_imaginary,
    uint32_t element_count);
} DGenHostServicesV1;

DGEN_EXPORT void dgen_process_v1(
  const float *const *inputs,
  float *const *outputs,
  uint32_t frame_count,
  void *state,
  const DGenProcessContextV1 *context,
  const DGenHostServicesV1 *host);

DGEN_EXPORT void dgen_set_param_value_v1(int32_t cell_id, float value);

#endif
