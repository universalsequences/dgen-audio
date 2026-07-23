#include "dgen_runtime.h"

void dgen_set_param_value_v1(int32_t cell_id, float value) {
  (void)cell_id;
  (void)value;
}

void dgen_process_v1(
  const float *const *inputs,
  float *const *outputs,
  uint32_t frame_count,
  void *state,
  const DGenProcessContextV1 *context,
  const DGenHostServicesV1 *host) {
  (void)inputs;
  (void)state;
  (void)context;
  (void)host;
  uint32_t nan_bits = UINT32_C(0x7fc00000);
  uint32_t infinity_bits = UINT32_C(0x7f800000);
  float nan_value;
  float infinity_value;
  __builtin_memcpy(&nan_value, &nan_bits, sizeof(nan_value));
  __builtin_memcpy(&infinity_value, &infinity_bits, sizeof(infinity_value));
  for (uint32_t frame = 0; frame < frame_count; ++frame) {
    outputs[0][frame] = dgen_sanitize_f32(
      (frame & 1u) == 0u ? nan_value : infinity_value);
  }
}
