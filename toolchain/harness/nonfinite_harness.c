#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void (*DGenProcessFn)(
  const float *const *,
  float *const *,
  uint32_t,
  void *,
  const void *,
  const void *);

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s NONFINITE_FIXTURE.dylib\n", argv[0]);
    return 2;
  }
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  void *symbol = dlsym(handle, "dgen_process_v1");
  DGenProcessFn process = NULL;
  memcpy(&process, &symbol, sizeof(process));
  if (process == NULL) {
    fprintf(stderr, "dlsym: %s\n", dlerror());
    return 1;
  }
  float input[4] = {0};
  float output[4] = {1, 1, 1, 1};
  const float *inputs[1] = {input};
  float *outputs[1] = {output};
  process(inputs, outputs, 4, NULL, NULL, NULL);
  for (int index = 0; index < 4; ++index) {
    if (!isfinite(output[index]) || output[index] != 0.0f) {
      fprintf(stderr, "containment failed at %d: %g\n", index, output[index]);
      return 1;
    }
  }
  puts("NaN/Inf containment passed under production flags.");
  dlclose(handle);
  return 0;
}
