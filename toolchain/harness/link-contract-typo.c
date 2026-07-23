extern float dgen_deliberate_link_typo(float);

__attribute__((visibility("default")))
float dgen_typo_probe(float value) {
  return dgen_deliberate_link_typo(value);
}
