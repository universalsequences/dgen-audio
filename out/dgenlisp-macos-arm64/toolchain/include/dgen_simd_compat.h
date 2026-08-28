#ifndef DGEN_SIMD_COMPAT_H
#define DGEN_SIMD_COMPAT_H

/*
 * DGen SIMD compatibility shim.
 *
 * The DGen code generator (CRenderer) emits ARM NEON intrinsics unconditionally.
 * On AArch64 / ARM this header is a plain passthrough to <arm_neon.h>, so the
 * macOS/arm64 path is byte-for-byte identical to what it always was.
 *
 * On every other target this header supplies the exact subset of NEON that the
 * generated code and dgen_runtime.h actually use, implemented on top of the
 * Clang/GCC generic vector extension (__attribute__((vector_size(16)))).
 *
 * This substitution is faithful because the existing DGen sources already treat
 * float32x4_t as a generic vector type -- e.g. `(float32x4_t){a, b, c, d}`
 * initializer lists and lane subscripting -- rather than as an opaque register.
 *
 * Semantics that are deliberately reproduced bit-exactly:
 *   - comparisons yield all-ones / all-zeros lane masks (0xFFFFFFFF / 0), not 0/1
 *   - vbslq_f32 is a per-BIT select, not a per-lane ternary
 *   - vfmaq/vfmsq take the accumulator FIRST and are single-rounded (fused)
 *   - vrndnq_f32 is round-to-nearest-EVEN, vrndaq_f32 is round-half-away-from-zero
 *   - vreinterpretq_* are pure bit casts, never value conversions
 *   - vld1q/vst1q are UNALIGNED memory accesses
 *
 * This header intentionally includes no libc headers other than <stdint.h>:
 * dgen_runtime.h is compiled in a freestanding-ish mode where it provides its
 * own size_t / NULL / INFINITY and its own libm extern declarations. All scalar
 * helpers below therefore use __builtin_* forms that need no declaration.
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

/*
 * Optimizer barrier for a 128-bit integer vector. Expands to exactly the
 * inline asm dgen_runtime.h used before this shim existed: an empty asm with
 * a read-write ARM vector-register ("w") operand, which emits no instruction
 * but severs -ffinite-math-only's provenance between a float and its integer
 * representation.
 */
#define DGEN_SIMD_OPT_BARRIER_U32(v) __asm__ __volatile__("" : "+w"(v))

#else /* ---------------------------- portable ---------------------------- */

#include <stdint.h>

#if defined(__clang__)
#define DGEN_SIMD_INLINE static inline __attribute__((always_inline, unused))
#else
#define DGEN_SIMD_INLINE static inline __attribute__((always_inline))
#endif

/* ------------------------------------------------------------------ types */

typedef float    float32x4_t __attribute__((vector_size(16), aligned(16)));
typedef uint32_t uint32x4_t  __attribute__((vector_size(16), aligned(16)));
typedef int32_t  int32x4_t   __attribute__((vector_size(16), aligned(16)));

/* 64-bit halves, needed only so vget_low/high style code would still build. */
typedef float    float32x2_t __attribute__((vector_size(8), aligned(8)));

/*
 * Internal: comparison result type produced by Clang/GCC for a 4x float
 * comparison is a *signed* 32-bit vector of 0 / -1. A C-style cast between
 * two vector types of identical width is a bit cast under the GCC vector
 * extension, so this reinterprets rather than converts.
 */
#define DGEN_SIMD_MASK(expr) ((uint32x4_t)(expr))

/* ------------------------------------------------------- splat / creation */

DGEN_SIMD_INLINE float32x4_t vdupq_n_f32(float v) {
  return (float32x4_t){v, v, v, v};
}

DGEN_SIMD_INLINE uint32x4_t vdupq_n_u32(uint32_t v) {
  return (uint32x4_t){v, v, v, v};
}

DGEN_SIMD_INLINE int32x4_t vdupq_n_s32(int32_t v) {
  return (int32x4_t){v, v, v, v};
}

/* ------------------------------------------------- unaligned load / store */

/*
 * NEON's LD1/ST1 have no alignment requirement. The vector typedefs above are
 * 16-byte aligned, so a bare pointer cast would emit an aligned x86 move and
 * fault on the 4-byte-aligned float pointers DGen passes around. Go through
 * __builtin_memcpy, which the optimizer folds into a single MOVUPS.
 */
DGEN_SIMD_INLINE float32x4_t vld1q_f32(const float *p) {
  float32x4_t r;
  __builtin_memcpy(&r, p, 16);
  return r;
}

DGEN_SIMD_INLINE void vst1q_f32(float *p, float32x4_t v) {
  __builtin_memcpy(p, &v, 16);
}

/* --------------------------------------------------------- lane accessors */

/*
 * NEON requires an immediate lane index; the generic vector extension does not,
 * so a real function works here and keeps type checking.
 */
DGEN_SIMD_INLINE float vgetq_lane_f32(float32x4_t v, int lane) {
  return v[lane];
}

DGEN_SIMD_INLINE uint32_t vgetq_lane_u32(uint32x4_t v, int lane) {
  return v[lane];
}

DGEN_SIMD_INLINE int32_t vgetq_lane_s32(int32x4_t v, int lane) {
  return v[lane];
}

DGEN_SIMD_INLINE float32x4_t vsetq_lane_f32(float v, float32x4_t vec, int lane) {
  vec[lane] = v;
  return vec;
}

/* -------------------------------------------------------------- bit casts */

/* Same-width vector casts are reinterpretations, not conversions. */
DGEN_SIMD_INLINE uint32x4_t vreinterpretq_u32_f32(float32x4_t v) { return (uint32x4_t)v; }
DGEN_SIMD_INLINE float32x4_t vreinterpretq_f32_u32(uint32x4_t v) { return (float32x4_t)v; }
DGEN_SIMD_INLINE int32x4_t  vreinterpretq_s32_u32(uint32x4_t v) { return (int32x4_t)v; }
DGEN_SIMD_INLINE uint32x4_t vreinterpretq_u32_s32(int32x4_t v)  { return (uint32x4_t)v; }
DGEN_SIMD_INLINE int32x4_t  vreinterpretq_s32_f32(float32x4_t v) { return (int32x4_t)v; }
DGEN_SIMD_INLINE float32x4_t vreinterpretq_f32_s32(int32x4_t v)  { return (float32x4_t)v; }

/* ------------------------------------------------------------- arithmetic */

DGEN_SIMD_INLINE float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return a + b; }
DGEN_SIMD_INLINE float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return a - b; }
DGEN_SIMD_INLINE float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return a * b; }
DGEN_SIMD_INLINE float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) { return a / b; }

DGEN_SIMD_INLINE float32x4_t vmulq_n_f32(float32x4_t a, float b) {
  return a * vdupq_n_f32(b);
}

DGEN_SIMD_INLINE int32x4_t vaddq_s32(int32x4_t a, int32x4_t b) { return a + b; }
DGEN_SIMD_INLINE int32x4_t vsubq_s32(int32x4_t a, int32x4_t b) { return a - b; }

/*
 * FNEG / FABS on NEON are pure sign-bit operations: they are exact for NaN,
 * infinities and signed zero, and cannot be reassociated away by -ffast-math.
 * Implementing them bitwise (rather than as `-a` / a conditional) preserves
 * that under -ffinite-math-only.
 */
DGEN_SIMD_INLINE float32x4_t vnegq_f32(float32x4_t a) {
  return (float32x4_t)((uint32x4_t)a ^ vdupq_n_u32(UINT32_C(0x80000000)));
}

DGEN_SIMD_INLINE float32x4_t vabsq_f32(float32x4_t a) {
  return (float32x4_t)((uint32x4_t)a & vdupq_n_u32(UINT32_C(0x7fffffff)));
}

/*
 * Fused multiply-add. NOTE THE OPERAND ORDER: the accumulator is FIRST.
 *   vfmaq_f32(a, b, c) == a + b * c
 *   vfmsq_f32(a, b, c) == a - b * c
 * __builtin_elementwise_fma is a single-rounded FMA, matching NEON's FMLA/FMLS
 * bit-for-bit on any target with FMA hardware. -march=x86-64-v3 implies FMA3 and
 * lowers this to VFMADD231PS, so it is exact there -- verified in codegen.
 *
 * CAVEAT: on a pre-FMA x86 target (-march=x86-64 / x86-64-v2) combined with
 * -ffast-math, LLVM is permitted to relax the intrinsic into a separate multiply
 * and add rather than calling libm fmaf. That reintroduces a second rounding
 * step and makes results differ from arm64 in the last ulp. Build the generated
 * code for x86-64-v3 or newer (or drop -ffast-math) to keep FMA exact.
 */
DGEN_SIMD_INLINE float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return __builtin_elementwise_fma(b, c, a);
}

DGEN_SIMD_INLINE float32x4_t vfmsq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return __builtin_elementwise_fma(-b, c, a);
}

DGEN_SIMD_INLINE float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return a + b * c;
}

/*
 * FMIN / FMAX. __builtin_elementwise_min/max map to LLVM minnum/maxnum, which
 * agree with NEON FMIN/FMAX on all numeric inputs. They differ only on NaN
 * (minnum returns the non-NaN operand; NEON returns the NaN) and on the
 * +0 / -0 tie, which LLVM leaves unspecified. DGen compiles with
 * -ffast-math (-ffinite-math-only), under which NaN inputs are already
 * undefined, so this deviation is unreachable in practice.
 */
DGEN_SIMD_INLINE float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
  return __builtin_elementwise_min(a, b);
}

DGEN_SIMD_INLINE float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
  return __builtin_elementwise_max(a, b);
}

/* -------------------------------------------------------------- rounding */

/* FRINTM: round toward -inf. */
DGEN_SIMD_INLINE float32x4_t vrndmq_f32(float32x4_t a) {
  return __builtin_elementwise_floor(a);
}

/* FRINTP: round toward +inf. */
DGEN_SIMD_INLINE float32x4_t vrndpq_f32(float32x4_t a) {
  return __builtin_elementwise_ceil(a);
}

/* FRINTZ: round toward zero. */
DGEN_SIMD_INLINE float32x4_t vrndq_f32(float32x4_t a) {
  return __builtin_elementwise_trunc(a);
}

/*
 * FRINTN: round to nearest, ties to EVEN. This is NOT roundf(). Using
 * __builtin_elementwise_roundeven keeps the tie rule exact and, unlike rintf,
 * is independent of the dynamic rounding mode.
 */
DGEN_SIMD_INLINE float32x4_t vrndnq_f32(float32x4_t a) {
  return __builtin_elementwise_roundeven(a);
}

/* FRINTA: round to nearest, ties AWAY FROM ZERO. This one really is roundf(). */
DGEN_SIMD_INLINE float32x4_t vrndaq_f32(float32x4_t a) {
  return __builtin_elementwise_round(a);
}

/* ---------------------------------------------------------- conversions */

/* FCVTZS: float -> int32, truncating toward zero (C conversion semantics). */
DGEN_SIMD_INLINE int32x4_t vcvtq_s32_f32(float32x4_t a) {
  return __builtin_convertvector(a, int32x4_t);
}

DGEN_SIMD_INLINE uint32x4_t vcvtq_u32_f32(float32x4_t a) {
  return __builtin_convertvector(a, uint32x4_t);
}

DGEN_SIMD_INLINE float32x4_t vcvtq_f32_s32(int32x4_t a) {
  return __builtin_convertvector(a, float32x4_t);
}

DGEN_SIMD_INLINE float32x4_t vcvtq_f32_u32(uint32x4_t a) {
  return __builtin_convertvector(a, float32x4_t);
}

/* --------------------------------------------------------- comparisons */

/*
 * Every comparison yields a per-lane mask of 0xFFFFFFFF (true) or 0x00000000
 * (false), matching NEON. Clang produces 0 / -1 in a signed 32-bit vector; the
 * DGEN_SIMD_MASK bit cast relabels it as uint32x4_t without changing bits.
 */
DGEN_SIMD_INLINE uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b) { return DGEN_SIMD_MASK(a == b); }
DGEN_SIMD_INLINE uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b) { return DGEN_SIMD_MASK(a >  b); }
DGEN_SIMD_INLINE uint32x4_t vcgeq_f32(float32x4_t a, float32x4_t b) { return DGEN_SIMD_MASK(a >= b); }
DGEN_SIMD_INLINE uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b) { return DGEN_SIMD_MASK(a <  b); }
DGEN_SIMD_INLINE uint32x4_t vcleq_f32(float32x4_t a, float32x4_t b) { return DGEN_SIMD_MASK(a <= b); }

DGEN_SIMD_INLINE uint32x4_t vceqq_u32(uint32x4_t a, uint32x4_t b) { return DGEN_SIMD_MASK(a == b); }
DGEN_SIMD_INLINE uint32x4_t vcgtq_u32(uint32x4_t a, uint32x4_t b) { return DGEN_SIMD_MASK(a >  b); }
DGEN_SIMD_INLINE uint32x4_t vcltq_u32(uint32x4_t a, uint32x4_t b) { return DGEN_SIMD_MASK(a <  b); }

DGEN_SIMD_INLINE uint32x4_t vceqq_s32(int32x4_t a, int32x4_t b) { return DGEN_SIMD_MASK(a == b); }
DGEN_SIMD_INLINE uint32x4_t vcgtq_s32(int32x4_t a, int32x4_t b) { return DGEN_SIMD_MASK(a >  b); }
DGEN_SIMD_INLINE uint32x4_t vcltq_s32(int32x4_t a, int32x4_t b) { return DGEN_SIMD_MASK(a <  b); }

/* ------------------------------------------------------------- bitwise */

DGEN_SIMD_INLINE uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b) { return a & b; }
DGEN_SIMD_INLINE uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b) { return a | b; }
DGEN_SIMD_INLINE uint32x4_t veorq_u32(uint32x4_t a, uint32x4_t b) { return a ^ b; }
DGEN_SIMD_INLINE uint32x4_t vbicq_u32(uint32x4_t a, uint32x4_t b) { return a & ~b; }

/* VMVN is a bitwise NOT, not a logical negation. */
DGEN_SIMD_INLINE uint32x4_t vmvnq_u32(uint32x4_t a) { return ~a; }

DGEN_SIMD_INLINE int32x4_t vandq_s32(int32x4_t a, int32x4_t b) { return a & b; }
DGEN_SIMD_INLINE int32x4_t vorrq_s32(int32x4_t a, int32x4_t b) { return a | b; }
DGEN_SIMD_INLINE int32x4_t veorq_s32(int32x4_t a, int32x4_t b) { return a ^ b; }

/*
 * Shift counts are immediates on NEON but need not be here. Splatting the
 * count keeps the operation well defined for the generic vector extension.
 */
DGEN_SIMD_INLINE uint32x4_t vshlq_n_u32(uint32x4_t a, int n) {
  return a << vdupq_n_u32((uint32_t)n);
}

DGEN_SIMD_INLINE uint32x4_t vshrq_n_u32(uint32x4_t a, int n) {
  return a >> vdupq_n_u32((uint32_t)n);
}

/* Arithmetic (sign-propagating) shift right, matching SSHR. */
DGEN_SIMD_INLINE int32x4_t vshrq_n_s32(int32x4_t a, int n) {
  return a >> vdupq_n_s32(n);
}

DGEN_SIMD_INLINE int32x4_t vshlq_n_s32(int32x4_t a, int n) {
  return a << vdupq_n_s32(n);
}

/* ----------------------------------------------------------- selection */

/*
 * Bitwise select. This is per-BIT, not per-lane:
 *   result = (a & mask) | (b & ~mask)
 * A lane-wise ternary would be wrong for any mask that is not uniformly all
 * ones or all zeros, and DGen does build such masks (see dgen_sanitize_f32x4).
 */
DGEN_SIMD_INLINE float32x4_t vbslq_f32(uint32x4_t mask, float32x4_t a, float32x4_t b) {
  uint32x4_t ab = (uint32x4_t)a;
  uint32x4_t bb = (uint32x4_t)b;
  return (float32x4_t)((ab & mask) | (bb & ~mask));
}

DGEN_SIMD_INLINE uint32x4_t vbslq_u32(uint32x4_t mask, uint32x4_t a, uint32x4_t b) {
  return (a & mask) | (b & ~mask);
}

DGEN_SIMD_INLINE int32x4_t vbslq_s32(uint32x4_t mask, int32x4_t a, int32x4_t b) {
  return (int32x4_t)(((uint32x4_t)a & mask) | ((uint32x4_t)b & ~mask));
}

/* -------------------------------------------------------- horizontal ops */

/*
 * FADDP-based horizontal add. The pairwise association ((a0+a1)+(a2+a3)) is
 * written out explicitly so the summation order matches AArch64's lowering of
 * vaddvq_f32 rather than a left-to-right chain.
 */
DGEN_SIMD_INLINE float vaddvq_f32(float32x4_t a) {
  return (a[0] + a[1]) + (a[2] + a[3]);
}

DGEN_SIMD_INLINE float vmaxvq_f32(float32x4_t a) {
  float r0 = a[0] > a[1] ? a[0] : a[1];
  float r1 = a[2] > a[3] ? a[2] : a[3];
  return r0 > r1 ? r0 : r1;
}

DGEN_SIMD_INLINE float vminvq_f32(float32x4_t a) {
  float r0 = a[0] < a[1] ? a[0] : a[1];
  float r1 = a[2] < a[3] ? a[2] : a[3];
  return r0 < r1 ? r0 : r1;
}

/* ------------------------------------------------------------ shuffles */

/*
 * vextq_f32(a, b, n): the 4-lane window starting at lane n of the
 * concatenation [a0 a1 a2 a3 b0 b1 b2 b3]. n must be an integer constant
 * expression, so this has to stay a macro (__builtin_shufflevector requires
 * literal indices).
 */
#define vextq_f32(a, b, n) \
  (__builtin_shufflevector((a), (b), (n) + 0, (n) + 1, (n) + 2, (n) + 3))

#define vextq_u32(a, b, n) \
  (__builtin_shufflevector((a), (b), (n) + 0, (n) + 1, (n) + 2, (n) + 3))

#define vextq_s32(a, b, n) \
  (__builtin_shufflevector((a), (b), (n) + 0, (n) + 1, (n) + 2, (n) + 3))

DGEN_SIMD_INLINE float32x4_t vcombine_f32(float32x2_t lo, float32x2_t hi) {
  return (float32x4_t){lo[0], lo[1], hi[0], hi[1]};
}

DGEN_SIMD_INLINE float32x2_t vget_low_f32(float32x4_t a)  { return (float32x2_t){a[0], a[1]}; }
DGEN_SIMD_INLINE float32x2_t vget_high_f32(float32x4_t a) { return (float32x2_t){a[2], a[3]}; }


/* --------------------------------------------------- optimizer barrier */

/*
 * Portable counterpart of the ARM "+w" barrier above. "+x" is the x86 SSE
 * register constraint; anything else falls back to a full memory clobber,
 * which is heavier but equally opaque to the optimizer.
 */
#if defined(__x86_64__) || defined(__i386__)
#define DGEN_SIMD_OPT_BARRIER_U32(v) __asm__ __volatile__("" : "+x"(v))
#else
#define DGEN_SIMD_OPT_BARRIER_U32(v) __asm__ __volatile__("" : "+g"(v) :: "memory")
#endif

#endif /* ARM vs portable */

#endif /* DGEN_SIMD_COMPAT_H */
