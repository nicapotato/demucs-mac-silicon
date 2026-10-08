#ifndef DEMUCS_C_LAYERS_H
#define DEMUCS_C_LAYERS_H

#include "ops.h"
#include "metal_kernels.h"

typedef struct {
  char prefix[128];
  int chin;
  int chout;
  int kernel;
  int stride;
  int pad; /* kernel//4 or 0 */
  int freq;
  int empty;
  int last;
  int rewrite;
  int dconv;
  int norm; /* outer groupnorm enabled */
  int fused_norm1;
  int fused_norm2;
  int context;
  int context_freq;
  int norm_groups;
  int dconv_depth;
  int dconv_comp;
  float dconv_init;
} DcLayerSpec;

mlx_array dc_dconv(mlx_array x, const DcWeights *w, const char *prefix,
                   int channels, int depth, int compress, mlx_stream s);
mlx_array dc_henc(mlx_array x, mlx_array inject, const DcWeights *w,
                  const DcLayerSpec *spec, mlx_stream s);
/* returns z; if pre_out non-NULL stores pre-conv_tr activation (y) */
mlx_array dc_hdec(mlx_array x, mlx_array skip, int length, const DcWeights *w,
                  const DcLayerSpec *spec, mlx_array *pre_out, mlx_stream s);

#endif
