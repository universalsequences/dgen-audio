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
  /*
   * Break the optimizer's finite-math provenance between the float and its
   * integer representation. The empty constraint emits no instruction, but
   * prevents -ffinite-math-only from proving this test true.
   */
  __asm__ __volatile__("" : "+r"(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

#define isfinite(value) dgen_isfinite_f32((float)(value))

static inline float dgen_sanitize_f32(float value) {
  return dgen_isfinite_f32(value) ? value : 0.0f;
}

static inline float32x4_t dgen_sanitize_f32x4(float32x4_t value) {
  uint32x4_t bits = vreinterpretq_u32_f32(value);
  __asm__ __volatile__("" : "+w"(bits));
  uint32x4_t exponent = vandq_u32(
    bits,
    vdupq_n_u32(UINT32_C(0x7f800000)));
  uint32x4_t finite = vmvnq_u32(
    vceqq_u32(exponent, vdupq_n_u32(UINT32_C(0x7f800000))));
  return vbslq_f32(finite, value, vdupq_n_f32(0.0f));
}

/*
 * Four-lane math names are part of the renderer/runtime-header boundary, not
 * the host-service ABI. These clean-room NEON polynomials use standard range
 * reduction and polynomial evaluation identities. Sine and cosine use a
 * three-term Cody-Waite reduction guaranteed for float32 arguments with
 * |x| <= 1.0e6. Beyond that range, float32 argument quantization, rather than
 * the reduction constants, becomes the limiting source of phase accuracy.
 * Measured domains, accuracy, and speed are recorded in
 * docs/vector-math-lowering.md.
 */
static inline float32x4_t dgen_reduce_two_pi_f32x4(float32x4_t x) {
  const float32x4_t inv_two_pi = vdupq_n_f32(0x1.45f306p-3f);
  const float32x4_t two_pi_hi = vdupq_n_f32(0x1.921fb6p+2f);
  const float32x4_t two_pi_mid = vdupq_n_f32(-0x1.777a5cp-23f);
  const float32x4_t two_pi_lo = vdupq_n_f32(-0x1.0p-47f);
  float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_two_pi));
  x = vfmsq_f32(x, n, two_pi_hi);
  x = vfmsq_f32(x, n, two_pi_mid);
  return vfmsq_f32(x, n, two_pi_lo);
}

static inline float32x4_t dgen_poly_vsinf(float32x4_t x) {
  x = dgen_reduce_two_pi_f32x4(x);
  uint32x4_t over = vcgtq_f32(x, vdupq_n_f32(1.5707963267948966f));
  uint32x4_t under = vcltq_f32(x, vdupq_n_f32(-1.5707963267948966f));
  x = vbslq_f32(over, vsubq_f32(vdupq_n_f32(3.1415926535897932f), x), x);
  x = vbslq_f32(under, vsubq_f32(vdupq_n_f32(-3.1415926535897932f), x), x);
  float32x4_t x2 = vmulq_f32(x, x);
  float32x4_t p = vdupq_n_f32(-2.50521084e-8f);
  p = vfmaq_f32(vdupq_n_f32(2.75573192e-6f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-1.98412698e-4f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(8.33333333e-3f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-1.66666667e-1f), p, x2);
  return vfmaq_f32(x, vmulq_f32(x, x2), p);
}

static inline float32x4_t dgen_poly_vcosf(float32x4_t x) {
  x = dgen_reduce_two_pi_f32x4(x);
  float32x4_t ax = vabsq_f32(x);
  uint32x4_t reflected = vcgtq_f32(ax, vdupq_n_f32(1.5707963267948966f));
  x = vbslq_f32(
    reflected,
    vsubq_f32(vdupq_n_f32(3.1415926535897932f), ax),
    ax);
  float32x4_t x2 = vmulq_f32(x, x);
  float32x4_t p = vdupq_n_f32(-2.75573192e-7f);
  p = vfmaq_f32(vdupq_n_f32(2.48015873e-5f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-1.38888889e-3f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(4.16666667e-2f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-5.0e-1f), p, x2);
  float32x4_t result = vfmaq_f32(vdupq_n_f32(1.0f), x2, p);
  return vbslq_f32(reflected, vnegq_f32(result), result);
}

static inline float32x4_t dgen_poly_vexpf(float32x4_t x) {
  x = vmaxq_f32(vdupq_n_f32(-80.0f), vminq_f32(x, vdupq_n_f32(80.0f)));
  float32x4_t nf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.4426950408889634f)));
  float32x4_t r = vfmsq_f32(x, nf, vdupq_n_f32(0.6931471805599453f));
  float32x4_t p = vdupq_n_f32(1.38888889e-3f);
  p = vfmaq_f32(vdupq_n_f32(8.33333333e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(4.16666667e-2f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.66666667e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(5.0e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  int32x4_t n = vcvtq_s32_f32(nf);
  uint32x4_t exponent = vshlq_n_u32(
    vreinterpretq_u32_s32(vaddq_s32(n, vdupq_n_s32(127))),
    23);
  return vmulq_f32(p, vreinterpretq_f32_u32(exponent));
}

static inline float32x4_t dgen_poly_vlogf(float32x4_t x) {
  uint32x4_t bits = vreinterpretq_u32_f32(x);
  int32x4_t exponent = vsubq_s32(
    vreinterpretq_s32_u32(vshrq_n_u32(bits, 23)),
    vdupq_n_s32(127));
  uint32x4_t mantissa_bits = vorrq_u32(
    vandq_u32(bits, vdupq_n_u32(UINT32_C(0x007fffff))),
    vdupq_n_u32(UINT32_C(0x3f800000)));
  float32x4_t m = vreinterpretq_f32_u32(mantissa_bits);
  uint32x4_t upper = vcgtq_f32(m, vdupq_n_f32(1.4142135623730951f));
  m = vbslq_f32(upper, vmulq_n_f32(m, 0.5f), m);
  exponent = vaddq_s32(
    exponent,
    vreinterpretq_s32_u32(vandq_u32(upper, vdupq_n_u32(1u))));
  float32x4_t y = vdivq_f32(
    vsubq_f32(m, vdupq_n_f32(1.0f)),
    vaddq_f32(m, vdupq_n_f32(1.0f)));
  float32x4_t y2 = vmulq_f32(y, y);
  float32x4_t p = vdupq_n_f32(1.0f / 9.0f);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 7.0f), p, y2);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), p, y2);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), p, y2);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, y2);
  return vfmaq_f32(
    vmulq_n_f32(vmulq_f32(y, p), 2.0f),
    vcvtq_f32_s32(exponent),
    vdupq_n_f32(0.6931471805599453f));
}

static inline float32x4_t dgen_poly_vtanhf(float32x4_t x) {
  float32x4_t magnitude = vabsq_f32(x);
  float32x4_t exponential = dgen_poly_vexpf(vmulq_n_f32(magnitude, -2.0f));
  float32x4_t result = vdivq_f32(
    vsubq_f32(vdupq_n_f32(1.0f), exponential),
    vaddq_f32(vdupq_n_f32(1.0f), exponential));
  return vbslq_f32(
    vcltq_f32(x, vdupq_n_f32(0.0f)),
    vnegq_f32(result),
    result);
}

static inline float32x4_t vsinf(float32x4_t value) {
  return dgen_poly_vsinf(value);
}

static inline float32x4_t vcosf(float32x4_t value) {
  return dgen_poly_vcosf(value);
}

static inline float32x4_t vtanhf(float32x4_t value) {
  return dgen_poly_vtanhf(value);
}

static inline float32x4_t vexpf(float32x4_t value) {
  return dgen_poly_vexpf(value);
}

static inline float32x4_t vlogf(float32x4_t value) {
  return dgen_poly_vlogf(value);
}

/* Less frequent families keep the accurate, unsurprising lane-wise lowering. */
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

DGEN_UNARY_VECTOR_WRAPPER(vtanf, tanf)
DGEN_UNARY_VECTOR_WRAPPER(vatanf, atanf)
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
