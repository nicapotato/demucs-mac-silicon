#ifndef DEMUCS_C_HTDEMUCS_H
#define DEMUCS_C_HTDEMUCS_H

#include "stft.h"
#include "transformer.h"

#define DC_MAX_DEPTH 8

typedef struct {
  DcWeights *w;
  DcStft stft;
  DcLayerSpec encoder[DC_MAX_DEPTH];
  DcLayerSpec tencoder[DC_MAX_DEPTH];
  DcLayerSpec decoder[DC_MAX_DEPTH];
  DcLayerSpec tdecoder[DC_MAX_DEPTH];
  int n_enc;
  int n_tenc;
  int n_dec;
  int n_tdec;
  mlx_stream stream;
  int use_compile;
  mlx_closure compiled;
  mlx_closure compile_raw;
} DcModel;

void dc_model_init(DcModel *m, DcWeights *w);
void dc_model_free(DcModel *m);
void dc_model_enable_compile(DcModel *m);
mlx_array dc_htdemucs_forward(DcModel *m, mlx_array mix);
mlx_array dc_model_forward(DcModel *m, mlx_array mix);

#endif
