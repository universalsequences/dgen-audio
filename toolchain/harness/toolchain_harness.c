#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DGEN_ABI_VERSION_V1 1u

typedef void *DGenFFTSetupV1;

typedef struct {
  uint32_t abi_version;
  uint32_t struct_size;
  float sample_rate;
  uint32_t reserved;
} DGenProcessContextV1;

typedef struct {
  uint32_t abi_version;
  uint32_t struct_size;
  DGenFFTSetupV1 (*fft_setup_create_fn)(uint32_t);
  void (*fft_forward_fn)(DGenFFTSetupV1, float *, float *, uint32_t);
  void (*fft_inverse_fn)(DGenFFTSetupV1, float *, float *, uint32_t);
  void (*complex_multiply_accumulate_fn)(
    const float *, const float *, const float *, const float *,
    float *, float *, uint32_t);
} DGenHostServicesV1;

typedef void (*DGenProcessFn)(
  const float *const *inputs,
  float *const *outputs,
  uint32_t frame_count,
  void *state,
  const DGenProcessContextV1 *context,
  const DGenHostServicesV1 *host);

typedef void (*DGenSetParamValueFn)(int32_t cell_id, float value);

typedef enum {
  DGEN_SCALAR_SYNTH,
  DGEN_FEEDBACK_DELAY,
  DGEN_WAVETABLE,
  DGEN_SPECTRAL
} FixtureKind;

typedef struct {
  void *handle;
  DGenProcessFn process;
  DGenSetParamValueFn set_param_value;
  float *memory;
  size_t memory_count;
} LoadedFixture;

static DGenFFTSetupV1 host_fft_setup_create(uint32_t log2_size) {
  return (DGenFFTSetupV1)vDSP_create_fftsetup((vDSP_Length)log2_size, kFFTRadix2);
}

static void host_fft_forward(
  DGenFFTSetupV1 setup, float *real, float *imaginary, uint32_t log2_size) {
  DSPSplitComplex split = {.realp = real, .imagp = imaginary};
  vDSP_fft_zip(
    (FFTSetup)setup, &split, 1, (vDSP_Length)log2_size,
    kFFTDirection_Forward);
}

static void host_fft_inverse(
  DGenFFTSetupV1 setup, float *real, float *imaginary, uint32_t log2_size) {
  DSPSplitComplex split = {.realp = real, .imagp = imaginary};
  vDSP_fft_zip(
    (FFTSetup)setup, &split, 1, (vDSP_Length)log2_size,
    kFFTDirection_Inverse);
}

static void host_complex_multiply_accumulate(
  const float *lhs_real, const float *lhs_imaginary,
  const float *rhs_real, const float *rhs_imaginary,
  float *accumulator_real, float *accumulator_imaginary,
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

static const DGenHostServicesV1 kHostServices = {
  .abi_version = DGEN_ABI_VERSION_V1,
  .struct_size = sizeof(DGenHostServicesV1),
  .fft_setup_create_fn = host_fft_setup_create,
  .fft_forward_fn = host_fft_forward,
  .fft_inverse_fn = host_fft_inverse,
  .complex_multiply_accumulate_fn = host_complex_multiply_accumulate};

static const DGenProcessContextV1 kProcessContext = {
  .abi_version = DGEN_ABI_VERSION_V1,
  .struct_size = sizeof(DGenProcessContextV1),
  .sample_rate = 48000.0f,
  .reserved = 0};

static const float kWavetable[32] = {
  0.0f, 0.0f, 0.38268343f, 0.125f, 0.70710677f, 0.25f,
  0.9238795f, 0.375f, 1.0f, 0.5f, 0.9238795f, 0.625f,
  0.70710677f, 0.75f, 0.38268343f, 0.875f, 0.0f, -1.0f,
  -0.38268343f, -0.875f, -0.70710677f, -0.75f,
  -0.9238795f, -0.625f, -1.0f, -0.5f, -0.9238795f, -0.375f,
  -0.70710677f, -0.25f, -0.38268343f, -0.125f};

static const float kSpectralHann[16] = {
  0.0f, 0.038060218f, 0.14644659f, 0.3086583f,
  0.49999997f, 0.69134164f, 0.8535534f, 0.96193975f,
  1.0f, 0.9619398f, 0.85355353f, 0.69134176f,
  0.5f, 0.30865842f, 0.14644668f, 0.038060308f};

static void usage(const char *program) {
  fprintf(
    stderr,
    "usage: %s scalar-synth|feedback-delay-effect|wavetable-instrument|"
    "spectral-effect REFERENCE.dylib CANDIDATE.dylib "
    "[blocks] [frames] [tolerance]\n",
    program);
}

static int parse_kind(const char *name, FixtureKind *kind) {
  if (strcmp(name, "scalar-synth") == 0) {
    *kind = DGEN_SCALAR_SYNTH;
  } else if (strcmp(name, "feedback-delay-effect") == 0) {
    *kind = DGEN_FEEDBACK_DELAY;
  } else if (strcmp(name, "wavetable-instrument") == 0) {
    *kind = DGEN_WAVETABLE;
  } else if (strcmp(name, "spectral-effect") == 0) {
    *kind = DGEN_SPECTRAL;
  } else {
    return 0;
  }
  return 1;
}

static size_t memory_count_for_kind(FixtureKind kind) {
  switch (kind) {
    case DGEN_SCALAR_SYNTH: return 4;
    case DGEN_FEEDBACK_DELAY: return 88005;
    case DGEN_WAVETABLE: return 36;
    case DGEN_SPECTRAL: return 3700;
  }
}

static void initialize_memory(FixtureKind kind, float *memory) {
  switch (kind) {
    case DGEN_SCALAR_SYNTH:
      memory[0] = 220.0f;
      memory[1] = 1.75f;
      memory[2] = 0.2f;
      break;
    case DGEN_FEEDBACK_DELAY:
      memory[0] = 37.0f;
      memory[1] = 0.35f;
      memory[2] = 0.4f;
      break;
    case DGEN_WAVETABLE:
      memcpy(memory, kWavetable, sizeof(kWavetable));
      memory[32] = 110.0f;
      memory[33] = 0.35f;
      memory[34] = 0.2f;
      break;
    case DGEN_SPECTRAL:
      memory[0] = 1.0f;
      memory[8] = 0.35f;
      memory[16] = 0.15f;
      memory[24] = 0.05f;
      memcpy(memory + 32, kSpectralHann, sizeof(kSpectralHann));
      for (int lane = 0; lane < 16; ++lane) {
        memory[48 + lane] = 1.0f;
        memory[64 + lane] = 0.35f;
        memory[80 + lane] = 0.15f;
        memory[96 + lane] = 0.05f;
      }
      break;
  }
}

static int load_fixture(
  const char *path, FixtureKind kind, LoadedFixture *fixture) {
  memset(fixture, 0, sizeof(*fixture));
  fixture->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (fixture->handle == NULL) {
    fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
    return 0;
  }

  void *symbol = dlsym(fixture->handle, "dgen_process_v1");
  if (symbol == NULL) {
    fprintf(stderr, "dlsym(dgen_process_v1, %s): %s\n", path, dlerror());
    dlclose(fixture->handle);
    return 0;
  }
  memcpy(&fixture->process, &symbol, sizeof(fixture->process));

  symbol = dlsym(fixture->handle, "dgen_set_param_value_v1");
  if (symbol == NULL) {
    fprintf(stderr, "dlsym(dgen_set_param_value_v1, %s): %s\n", path, dlerror());
    dlclose(fixture->handle);
    return 0;
  }
  memcpy(
    &fixture->set_param_value, &symbol, sizeof(fixture->set_param_value));

  fixture->memory_count = memory_count_for_kind(kind);
  fixture->memory = calloc(fixture->memory_count, sizeof(float));
  if (fixture->memory == NULL) {
    fprintf(stderr, "calloc failed for %s\n", path);
    dlclose(fixture->handle);
    return 0;
  }
  initialize_memory(kind, fixture->memory);
  return 1;
}

static void unload_fixture(LoadedFixture *fixture) {
  free(fixture->memory);
  if (fixture->handle != NULL) dlclose(fixture->handle);
}

int main(int argc, char **argv) {
  if (argc < 4 || argc > 7) {
    usage(argv[0]);
    return 2;
  }

  FixtureKind kind;
  if (!parse_kind(argv[1], &kind)) {
    usage(argv[0]);
    return 2;
  }

  const int block_count = argc >= 5 ? atoi(argv[4]) : 128;
  const int frame_count = argc >= 6 ? atoi(argv[5]) : 64;
  const float tolerance = argc >= 7 ? strtof(argv[6], NULL) : 2.0e-5f;
  if (block_count <= 0 || frame_count <= 0 || frame_count % 4 != 0) {
    fprintf(stderr, "blocks and frames must be positive; frames must be divisible by 4\n");
    return 2;
  }

  LoadedFixture reference;
  LoadedFixture candidate;
  if (!load_fixture(argv[2], kind, &reference)) return 1;
  if (!load_fixture(argv[3], kind, &candidate)) {
    unload_fixture(&reference);
    return 1;
  }

  float *input = calloc((size_t)frame_count, sizeof(float));
  float *reference_output = calloc((size_t)frame_count, sizeof(float));
  float *candidate_output = calloc((size_t)frame_count, sizeof(float));
  if (input == NULL || reference_output == NULL || candidate_output == NULL) {
    fprintf(stderr, "audio-buffer allocation failed\n");
    unload_fixture(&candidate);
    unload_fixture(&reference);
    free(input);
    free(reference_output);
    free(candidate_output);
    return 1;
  }

  const float *inputs[1] = {input};
  float *reference_outputs[1] = {reference_output};
  float *candidate_outputs[1] = {candidate_output};
  double max_error = 0.0;
  double squared_error = 0.0;
  double reference_energy = 0.0;
  double candidate_checksum = 0.0;
  size_t compared_count = 0;

  for (int block = 0; block < block_count; ++block) {
    for (int frame = 0; frame < frame_count; ++frame) {
      const int absolute_frame = block * frame_count + frame;
      const float tone = 0.25f * sinf(
        2.0f * 3.14159265358979323846f * 330.0f *
        (float)absolute_frame / 48000.0f);
      input[frame] = tone + (absolute_frame == 0 ? 0.5f : 0.0f);
    }
    memset(reference_output, 0, (size_t)frame_count * sizeof(float));
    memset(candidate_output, 0, (size_t)frame_count * sizeof(float));

    reference.process(
      inputs, reference_outputs, (uint32_t)frame_count,
      reference.memory, &kProcessContext, &kHostServices);
    candidate.process(
      inputs, candidate_outputs, (uint32_t)frame_count,
      candidate.memory, &kProcessContext, &kHostServices);

    for (int frame = 0; frame < frame_count; ++frame) {
      if (!isfinite(reference_output[frame]) || !isfinite(candidate_output[frame])) {
        fprintf(stderr, "non-finite output at block %d frame %d\n", block, frame);
        return 1;
      }
      const double error = fabs(
        (double)reference_output[frame] - (double)candidate_output[frame]);
      if (error > max_error) max_error = error;
      squared_error += error * error;
      reference_energy +=
        (double)reference_output[frame] * (double)reference_output[frame];
      candidate_checksum +=
        (double)candidate_output[frame] * (double)(compared_count + 1);
      ++compared_count;
    }
  }

  const double rms_error = sqrt(squared_error / (double)compared_count);
  const double reference_rms = sqrt(reference_energy / (double)compared_count);
  printf(
    "%s: samples=%zu max_error=%.9g rms_error=%.9g "
    "reference_rms=%.9g candidate_checksum=%.12g tolerance=%.9g\n",
    argv[1], compared_count, max_error, rms_error,
    reference_rms, candidate_checksum, (double)tolerance);

  free(candidate_output);
  free(reference_output);
  free(input);
  unload_fixture(&candidate);
  unload_fixture(&reference);
  return max_error <= (double)tolerance ? 0 : 1;
}
