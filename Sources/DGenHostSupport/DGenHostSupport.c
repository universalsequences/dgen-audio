#include "DGenHostSupport.h"

#if defined(__APPLE__)

#include <Accelerate/Accelerate.h>

static DGenHostFFTSetupV1 fft_setup_create(uint32_t log2_size) {
  return (DGenHostFFTSetupV1)vDSP_create_fftsetup(
    (vDSP_Length)log2_size, kFFTRadix2);
}

static void fft_forward(
  DGenHostFFTSetupV1 setup,
  float *real,
  float *imaginary,
  uint32_t log2_size) {
  DSPSplitComplex split = {.realp = real, .imagp = imaginary};
  vDSP_fft_zip(
    (FFTSetup)setup, &split, 1, (vDSP_Length)log2_size,
    kFFTDirection_Forward);
}

static void fft_inverse(
  DGenHostFFTSetupV1 setup,
  float *real,
  float *imaginary,
  uint32_t log2_size) {
  DSPSplitComplex split = {.realp = real, .imagp = imaginary};
  vDSP_fft_zip(
    (FFTSetup)setup, &split, 1, (vDSP_Length)log2_size,
    kFFTDirection_Inverse);
}

static void complex_multiply_accumulate(
  const float *lhs_real,
  const float *lhs_imaginary,
  const float *rhs_real,
  const float *rhs_imaginary,
  float *accumulator_real,
  float *accumulator_imaginary,
  uint32_t element_count) {
  DSPSplitComplex lhs = {
    .realp = (float *)lhs_real, .imagp = (float *)lhs_imaginary};
  DSPSplitComplex rhs = {
    .realp = (float *)rhs_real, .imagp = (float *)rhs_imaginary};
  DSPSplitComplex accumulator = {
    .realp = accumulator_real, .imagp = accumulator_imaginary};
  vDSP_zvma(
    &lhs, 1, &rhs, 1, &accumulator, 1, &accumulator, 1,
    (vDSP_Length)element_count);
}

#else  /* !__APPLE__ : portable reference implementation */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Portable stand-in for Accelerate's split-complex FFT.
 *
 * Semantics are matched to vDSP_fft_zip with kFFTRadix2:
 *   - in-place, split (separate real/imaginary) arrays, unit stride
 *   - size is 2^log2_size
 *   - forward uses the exp(-2*pi*i*n*k/N) kernel
 *   - inverse uses exp(+2*pi*i*n*k/N) and is NOT normalized by 1/N
 *     (vDSP leaves that scaling to the caller; we do the same so both
 *      platforms produce bit-comparable magnitudes).
 */

typedef struct DGenPortableFFTSetup {
  uint32_t max_log2_size;
  uint32_t table_count; /* == (1u << max_log2_size) / 2, or 0 */
  double *twiddle_real; /* cos(-2*pi*j/N) */
  double *twiddle_imaginary; /* sin(-2*pi*j/N) */
} DGenPortableFFTSetup;

static DGenHostFFTSetupV1 fft_setup_create(uint32_t log2_size) {
  DGenPortableFFTSetup *setup =
    (DGenPortableFFTSetup *)calloc(1, sizeof(DGenPortableFFTSetup));
  if (setup == NULL) {
    return NULL;
  }
  setup->max_log2_size = log2_size;
  if (log2_size >= 1 && log2_size < 31) {
    size_t n = (size_t)1 << log2_size;
    size_t half = n / 2;
    setup->twiddle_real = (double *)malloc(half * sizeof(double));
    setup->twiddle_imaginary = (double *)malloc(half * sizeof(double));
    if (setup->twiddle_real == NULL || setup->twiddle_imaginary == NULL) {
      free(setup->twiddle_real);
      free(setup->twiddle_imaginary);
      setup->twiddle_real = NULL;
      setup->twiddle_imaginary = NULL;
      setup->table_count = 0;
    } else {
      for (size_t j = 0; j < half; ++j) {
        double angle = -2.0 * M_PI * (double)j / (double)n;
        setup->twiddle_real[j] = cos(angle);
        setup->twiddle_imaginary[j] = sin(angle);
      }
      setup->table_count = (uint32_t)half;
    }
  }
  return (DGenHostFFTSetupV1)setup;
}

/* In-place decimation-in-time radix-2 FFT over split real/imaginary arrays. */
static void fft_execute(
  DGenHostFFTSetupV1 opaque_setup,
  float *real,
  float *imaginary,
  uint32_t log2_size,
  int inverse) {
  const DGenPortableFFTSetup *setup =
    (const DGenPortableFFTSetup *)opaque_setup;
  if (real == NULL || imaginary == NULL || log2_size >= 31) {
    return;
  }

  const size_t n = (size_t)1 << log2_size;
  if (n < 2) {
    return; /* N == 1 is the identity transform */
  }

  /* Bit-reversal permutation. */
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      float tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      float ti = imaginary[i];
      imaginary[i] = imaginary[j];
      imaginary[j] = ti;
    }
  }

  /* Reuse the setup's twiddle table when it covers this transform size. */
  const int use_table = (setup != NULL && setup->table_count != 0 &&
                         log2_size <= setup->max_log2_size);
  const size_t table_n =
    use_table ? ((size_t)1 << setup->max_log2_size) : 0;

  for (size_t len = 2; len <= n; len <<= 1) {
    const size_t half = len >> 1;
    const size_t table_step = use_table ? (table_n / len) : 0;
    for (size_t base = 0; base < n; base += len) {
      for (size_t k = 0; k < half; ++k) {
        double wr;
        double wi;
        if (use_table) {
          const size_t index = k * table_step;
          wr = setup->twiddle_real[index];
          wi = setup->twiddle_imaginary[index];
        } else {
          const double angle = -2.0 * M_PI * (double)k / (double)len;
          wr = cos(angle);
          wi = sin(angle);
        }
        if (inverse) {
          wi = -wi; /* conjugate kernel, no 1/N normalization */
        }

        const size_t a = base + k;
        const size_t b = a + half;
        const double br = (double)real[b];
        const double bi = (double)imaginary[b];
        const double tr = br * wr - bi * wi;
        const double ti = br * wi + bi * wr;
        const double ar = (double)real[a];
        const double ai = (double)imaginary[a];
        real[a] = (float)(ar + tr);
        imaginary[a] = (float)(ai + ti);
        real[b] = (float)(ar - tr);
        imaginary[b] = (float)(ai - ti);
      }
    }
  }
}

static void fft_forward(
  DGenHostFFTSetupV1 setup,
  float *real,
  float *imaginary,
  uint32_t log2_size) {
  fft_execute(setup, real, imaginary, log2_size, 0);
}

static void fft_inverse(
  DGenHostFFTSetupV1 setup,
  float *real,
  float *imaginary,
  uint32_t log2_size) {
  fft_execute(setup, real, imaginary, log2_size, 1);
}

/* Matches vDSP_zvma: accumulator = lhs * rhs + accumulator (complex). */
static void complex_multiply_accumulate(
  const float *lhs_real,
  const float *lhs_imaginary,
  const float *rhs_real,
  const float *rhs_imaginary,
  float *accumulator_real,
  float *accumulator_imaginary,
  uint32_t element_count) {
  for (uint32_t i = 0; i < element_count; ++i) {
    const float lr = lhs_real[i];
    const float li = lhs_imaginary[i];
    const float rr = rhs_real[i];
    const float ri = rhs_imaginary[i];
    accumulator_real[i] += lr * rr - li * ri;
    accumulator_imaginary[i] += lr * ri + li * rr;
  }
}

#endif /* __APPLE__ */

static const DGenHostServicesV1 kHostServicesV1 = {
  .abi_version = 1,
  .struct_size = sizeof(DGenHostServicesV1),
  .fft_setup_create_fn = fft_setup_create,
  .fft_forward_fn = fft_forward,
  .fft_inverse_fn = fft_inverse,
  .complex_multiply_accumulate_fn = complex_multiply_accumulate};

const DGenHostServicesV1 *dgen_reference_host_services_v1(void) {
  return &kHostServicesV1;
}
