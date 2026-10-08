#ifndef DEMUCS_C_OPS_H
#define DEMUCS_C_OPS_H

#include "weights.h"

mlx_array dc_conv1d_ncl(mlx_array x, mlx_array weight, mlx_array bias, int stride,
                        int padding, int dilation, mlx_stream s);
mlx_array dc_conv2d_nchw(mlx_array x, mlx_array weight, mlx_array bias,
                         int stride0, int stride1, int pad0, int pad1,
                         mlx_stream s);
mlx_array dc_conv_transpose1d_ncl(mlx_array x, mlx_array weight, mlx_array bias,
                                  int stride, int padding, mlx_stream s);
mlx_array dc_conv_transpose2d_nchw(mlx_array x, mlx_array weight, mlx_array bias,
                                   int stride0, int stride1, mlx_stream s);
mlx_array dc_linear(mlx_array x, mlx_array weight, mlx_array bias, mlx_stream s);
mlx_array dc_layer_norm(mlx_array x, mlx_array weight, mlx_array bias, float eps,
                        mlx_stream s);
mlx_array dc_mha(mlx_array q, mlx_array k, mlx_array v, mlx_array wq,
                 mlx_array bq, mlx_array wk, mlx_array bk, mlx_array wv,
                 mlx_array bv, mlx_array wo, mlx_array bo, int nheads,
                 mlx_stream s);
mlx_array dc_embedding(mlx_array weight, mlx_array indices, mlx_stream s);

mlx_array dc_conv1d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         int stride, int padding, int dilation, mlx_stream s);
mlx_array dc_conv2d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         int stride0, int stride1, int pad0, int pad1,
                         mlx_stream s);
mlx_array dc_convtr1d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                           int stride, int padding, mlx_stream s);
mlx_array dc_convtr2d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                           int stride0, int stride1, mlx_stream s);
mlx_array dc_linear_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         mlx_stream s);
mlx_array dc_ln_pref(mlx_array x, const DcWeights *w, const char *prefix,
                     mlx_stream s);

#endif
