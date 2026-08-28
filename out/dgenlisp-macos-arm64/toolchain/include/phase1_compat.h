#ifndef DGEN_PHASE1_COMPAT_H
#define DGEN_PHASE1_COMPAT_H

/*
 * Phase 1 fixture-only compatibility header.
 *
 * This makes today's generated C compile without an Apple SDK. It is not the
 * versioned Phase 2 dgen_runtime.h and must not be installed as a production
 * public contract.
 */

#include <arm_neon.h>
#include <stdint.h>

typedef __SIZE_TYPE__ size_t;

#ifndef NULL
#define NULL ((void *)0)
#endif

#define INFINITY (__builtin_inff())
#define M_LOG10E 0.43429448190325182765
#define isfinite(value) __builtin_isfinite((double)(value))

/* Current scalar math surface from CRenderer.swift. */
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
extern double fabs(double);

extern void *memcpy(void *restrict, const void *restrict, size_t);
extern void *memset(void *, int, size_t);

/*
 * Today's SIMD renderer spells vector math as Accelerate vForce calls.
 * Lane-wise wrappers preserve the generated algorithm for this prototype while
 * keeping the linked fixture dylib on the libSystem-only side of the contract.
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

/*
 * Declarations below let the current spectral fixture compile to an object
 * without Accelerate headers. Its vDSP calls intentionally remain unresolved;
 * the Phase 1 libSystem-only run proof excludes that fixture pending Phase 2's
 * host-service table.
 */
typedef unsigned long vDSP_Length;
typedef long vDSP_Stride;
typedef int FFTDirection;
typedef int FFTRadix;
typedef void *FFTSetup;

typedef struct {
  float *realp;
  float *imagp;
} DSPSplitComplex;

enum {
  kFFTRadix2 = 0,
  kFFTDirection_Forward = 1,
  kFFTDirection_Inverse = -1
};

extern FFTSetup vDSP_create_fftsetup(vDSP_Length, FFTRadix);
extern void vDSP_fft_zip(
  FFTSetup, DSPSplitComplex *, vDSP_Stride, vDSP_Length, FFTDirection);
extern void vDSP_zvma(
  const DSPSplitComplex *, vDSP_Stride,
  const DSPSplitComplex *, vDSP_Stride,
  const DSPSplitComplex *, vDSP_Stride,
  DSPSplitComplex *, vDSP_Stride,
  vDSP_Length);

#endif
