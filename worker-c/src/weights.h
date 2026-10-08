#ifndef DEMUCS_C_WEIGHTS_H
#define DEMUCS_C_WEIGHTS_H

#include "config.h"

typedef struct {
  mlx_map_string_to_array tensors;
  mlx_map_string_to_string metadata;
  DcConfig cfg;
  mlx_stream stream;
} DcWeights;

DcWeights dc_weights_load(const char *safetensors_path, const char *json_path,
                          mlx_stream stream);
void dc_weights_free(DcWeights *w);
mlx_array dc_w(const DcWeights *w, const char *fmt, ...);
bool dc_w_has(const DcWeights *w, const char *fmt, ...);

#endif
