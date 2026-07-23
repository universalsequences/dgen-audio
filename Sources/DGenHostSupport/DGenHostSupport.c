#include "DGenHostSupport.h"

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
