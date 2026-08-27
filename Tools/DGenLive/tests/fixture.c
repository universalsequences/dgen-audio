#include "dgen_runtime.h"

void dgen_set_param_value_v1(int32_t cell_id, float value) {
  (void)cell_id;
  (void)value;
}

void dgen_process_v1(const float *const *inputs, float *const *outputs,
                     uint32_t frame_count, void *state,
                     const DGenProcessContextV1 *context,
                     const DGenHostServicesV1 *host) {
  (void)inputs;
  (void)host;
  float *memory = state;
  float sample_rate = context ? context->sample_rate : 0.0f;
  for (uint32_t i = 0; i < frame_count; ++i)
    outputs[0][i] = memory[0] + memory[1] + memory[10] + sample_rate / 44100.0f;
}
