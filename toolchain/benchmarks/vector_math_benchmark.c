#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Clean-room polynomial candidates written for this benchmark from standard
 * range-reduction identities and Taylor/minimax-style polynomial evaluation.
 * No third-party implementation or source code is incorporated.
 */

typedef float32x4_t (*UnaryVectorFn)(float32x4_t);

static inline float32x4_t dgen_poly_sin(float32x4_t x) {
  const float32x4_t inv_two_pi = vdupq_n_f32(0.15915494309189535f);
  const float32x4_t two_pi = vdupq_n_f32(6.2831853071795865f);
  x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));

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

static inline float32x4_t dgen_poly_cos(float32x4_t x) {
  const float32x4_t inv_two_pi = vdupq_n_f32(0.15915494309189535f);
  const float32x4_t two_pi = vdupq_n_f32(6.2831853071795865f);
  x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));

  float32x4_t ax = vabsq_f32(x);
  uint32x4_t reflected = vcgtq_f32(ax, vdupq_n_f32(1.5707963267948966f));
  float32x4_t reduced = vsubq_f32(vdupq_n_f32(3.1415926535897932f), ax);
  x = vbslq_f32(reflected, reduced, ax);

  float32x4_t x2 = vmulq_f32(x, x);
  float32x4_t p = vdupq_n_f32(-2.75573192e-7f);
  p = vfmaq_f32(vdupq_n_f32(2.48015873e-5f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-1.38888889e-3f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(4.16666667e-2f), p, x2);
  p = vfmaq_f32(vdupq_n_f32(-5.0e-1f), p, x2);
  float32x4_t result = vfmaq_f32(vdupq_n_f32(1.0f), x2, p);
  return vbslq_f32(reflected, vnegq_f32(result), result);
}

static inline float32x4_t dgen_poly_exp(float32x4_t x) {
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
  uint32x4_t exponent = vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(n, vdupq_n_s32(127))), 23);
  return vmulq_f32(p, vreinterpretq_f32_u32(exponent));
}

static inline float32x4_t dgen_poly_log(float32x4_t x) {
  uint32x4_t bits = vreinterpretq_u32_f32(x);
  int32x4_t exponent = vsubq_s32(
    vreinterpretq_s32_u32(vshrq_n_u32(bits, 23)),
    vdupq_n_s32(127));
  uint32x4_t mantissa_bits = vorrq_u32(
    vandq_u32(bits, vdupq_n_u32(0x007fffffu)),
    vdupq_n_u32(0x3f800000u));
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
  float32x4_t log_m = vmulq_n_f32(vmulq_f32(y, p), 2.0f);
  return vfmaq_f32(log_m, vcvtq_f32_s32(exponent), vdupq_n_f32(0.6931471805599453f));
}

static inline float32x4_t dgen_poly_tanh(float32x4_t x) {
  float32x4_t ax = vabsq_f32(x);
  float32x4_t e = dgen_poly_exp(vmulq_n_f32(ax, -2.0f));
  float32x4_t y = vdivq_f32(
    vsubq_f32(vdupq_n_f32(1.0f), e),
    vaddq_f32(vdupq_n_f32(1.0f), e));
  return vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0f)), vnegq_f32(y), y);
}

#define LANEWISE_WRAPPER(name, scalar)                                      \
  static inline float32x4_t name(float32x4_t x) {                          \
    return (float32x4_t){                                                    \
      scalar(vgetq_lane_f32(x, 0)), scalar(vgetq_lane_f32(x, 1)),           \
      scalar(vgetq_lane_f32(x, 2)), scalar(vgetq_lane_f32(x, 3))};          \
  }

LANEWISE_WRAPPER(dgen_scalar_sin, sinf)
LANEWISE_WRAPPER(dgen_scalar_cos, cosf)
LANEWISE_WRAPPER(dgen_scalar_tanh, tanhf)
LANEWISE_WRAPPER(dgen_scalar_exp, expf)
LANEWISE_WRAPPER(dgen_scalar_log, logf)

static double seconds_now(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC_RAW, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static volatile float benchmark_sink;

static double benchmark(
  UnaryVectorFn function,
  const float *input,
  int frame_count,
  int iterations) {
  float32x4_t accumulator = vdupq_n_f32(0.0f);
  double start = seconds_now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (int frame = 0; frame < frame_count; frame += 4) {
      float32x4_t value = vld1q_f32(input + frame);
      accumulator = vaddq_f32(accumulator, function(value));
    }
  }
  double elapsed = seconds_now() - start;
  benchmark_sink = vaddvq_f32(accumulator);
  return elapsed * 1.0e9 / ((double)frame_count * (double)iterations);
}

static uint32_t ordered_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static void accuracy(
  const char *name,
  UnaryVectorFn candidate,
  float (*reference)(float),
  float minimum,
  float maximum) {
  const int count = 1 << 20;
  double maximum_absolute = 0.0;
  uint32_t maximum_ulp = 0;
  for (int base = 0; base < count; base += 4) {
    float lanes[4];
    for (int lane = 0; lane < 4; ++lane) {
      int index = base + lane;
      lanes[lane] = minimum + (maximum - minimum) * (float)index / (float)(count - 1);
    }
    float result[4];
    vst1q_f32(result, candidate(vld1q_f32(lanes)));
    for (int lane = 0; lane < 4; ++lane) {
      float expected = reference(lanes[lane]);
      double absolute = fabs((double)result[lane] - (double)expected);
      uint32_t lhs = ordered_bits(result[lane]);
      uint32_t rhs = ordered_bits(expected);
      uint32_t ulp = lhs > rhs ? lhs - rhs : rhs - lhs;
      if (absolute > maximum_absolute) maximum_absolute = absolute;
      if (ulp > maximum_ulp) maximum_ulp = ulp;
    }
  }
  printf("accuracy,%s,max_abs,%.9g,max_ulp,%u\n", name, maximum_absolute, maximum_ulp);
}

typedef struct {
  const char *name;
  UnaryVectorFn baseline;
  UnaryVectorFn scalar;
  UnaryVectorFn polynomial;
  float (*reference)(float);
  float minimum;
  float maximum;
} FunctionFamily;

int main(void) {
  FunctionFamily families[] = {
    {"sin", vsinf, dgen_scalar_sin, dgen_poly_sin, sinf, -6.28318531f, 6.28318531f},
    {"cos", vcosf, dgen_scalar_cos, dgen_poly_cos, cosf, -6.28318531f, 6.28318531f},
    {"tanh", vtanhf, dgen_scalar_tanh, dgen_poly_tanh, tanhf, -8.0f, 8.0f},
    {"exp", vexpf, dgen_scalar_exp, dgen_poly_exp, expf, -10.0f, 10.0f},
    {"log", vlogf, dgen_scalar_log, dgen_poly_log, logf, 0.0001f, 100.0f},
  };
  const int frame_counts[] = {64, 256, 1024};
  float input[1024];

  puts("kind,function,implementation,frames,ns_per_sample");
  for (size_t family_index = 0;
       family_index < sizeof(families) / sizeof(families[0]);
       ++family_index) {
    FunctionFamily family = families[family_index];
    for (int frame = 0; frame < 1024; ++frame) {
      input[frame] = family.minimum
        + (family.maximum - family.minimum) * (float)(frame + 1) / 1025.0f;
    }
    for (size_t size_index = 0;
         size_index < sizeof(frame_counts) / sizeof(frame_counts[0]);
         ++size_index) {
      int frames = frame_counts[size_index];
      int iterations = 64000000 / frames;
      double baseline = benchmark(family.baseline, input, frames, iterations);
      double scalar = benchmark(family.scalar, input, frames, iterations);
      double polynomial = benchmark(family.polynomial, input, frames, iterations);
      printf("speed,%s,vecLib,%d,%.6f\n", family.name, frames, baseline);
      printf("speed,%s,scalar-libm,%d,%.6f\n", family.name, frames, scalar);
      printf("speed,%s,polynomial,%d,%.6f\n", family.name, frames, polynomial);
    }
    accuracy(family.name, family.polynomial, family.reference, family.minimum, family.maximum);
  }
  return benchmark_sink == FLT_MAX;
}
