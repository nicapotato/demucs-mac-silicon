#include "weights.h"

DcWeights dc_weights_load(const char *safetensors_path, const char *json_path,
                          mlx_stream stream) {
  DcWeights w;
  memset(&w, 0, sizeof(w));
  w.stream = stream;
  w.tensors = mlx_map_string_to_array_new();
  w.metadata = mlx_map_string_to_string_new();
  mlx_stream cpu = mlx_default_cpu_stream_new();
  DC_CHECK(mlx_load_safetensors(&w.tensors, &w.metadata, safetensors_path, cpu));
  mlx_stream_free(cpu);
  dc_config_load_json(&w.cfg, json_path);
  return w;
}

void dc_weights_free(DcWeights *w) {
  if (!w) {
    return;
  }
  mlx_map_string_to_array_free(w->tensors);
  mlx_map_string_to_string_free(w->metadata);
  memset(w, 0, sizeof(*w));
}

static void format_key(char *buf, size_t cap, const char *fmt, va_list ap) {
  if (vsnprintf(buf, cap, fmt, ap) >= (int)cap) {
    dc_die("weight key overflow");
  }
}

mlx_array dc_w(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  return dc_require_weight(w->tensors, key);
}

bool dc_w_has(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  return dc_has_weight(w->tensors, key);
}
