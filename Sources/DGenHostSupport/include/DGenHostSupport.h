#ifndef DGEN_HOST_SUPPORT_H
#define DGEN_HOST_SUPPORT_H

#include <stdint.h>

typedef void *DGenHostFFTSetupV1;

typedef struct DGenHostServicesV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  DGenHostFFTSetupV1 (*fft_setup_create_fn)(uint32_t log2_size);
  void (*fft_forward_fn)(
    DGenHostFFTSetupV1 setup,
    float *real,
    float *imaginary,
    uint32_t log2_size);
  void (*fft_inverse_fn)(
    DGenHostFFTSetupV1 setup,
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

const DGenHostServicesV1 *dgen_reference_host_services_v1(void);

#endif
