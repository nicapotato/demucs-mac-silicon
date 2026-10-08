#include "ops.h"

#include "metal_kernels.h"

#include <math.h>

mlx_array dc_conv1d_ncl(mlx_array x, mlx_array weight, mlx_array bias, int stride,
                        int padding, int dilation, mlx_stream s) {
  int axes[3] = {0, 2, 1};
  mlx_array xt = dc_transpose(x, axes, 3, s);
  mlx_array xc = dc_contiguous(xt, s);
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_conv1d(&y, xc, weight, stride, padding, dilation, 1, s));
  mlx_array yb = dc_add(y, bias, s);
  mlx_array ot = dc_transpose(yb, axes, 3, s);
  mlx_array out = dc_contiguous(ot, s);
  mlx_array_free(xt);
  mlx_array_free(xc);
  mlx_array_free(y);
  mlx_array_free(yb);
  mlx_array_free(ot);
  return out;
}

mlx_array dc_conv2d_nchw(mlx_array x, mlx_array weight, mlx_array bias,
                         int stride0, int stride1, int pad0, int pad1,
                         mlx_stream s) {
  int axes_in[4] = {0, 2, 3, 1};
  mlx_array xt = dc_transpose(x, axes_in, 4, s);
  mlx_array xc = dc_contiguous(xt, s);
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_conv2d(&y, xc, weight, stride0, stride1, pad0, pad1, 1, 1, 1,
                      s));
  mlx_array yb = dc_add(y, bias, s);
  int axes_out[4] = {0, 3, 1, 2};
  mlx_array ot = dc_transpose(yb, axes_out, 4, s);
  mlx_array out = dc_contiguous(ot, s);
  mlx_array_free(xt);
  mlx_array_free(xc);
  mlx_array_free(y);
  mlx_array_free(yb);
  mlx_array_free(ot);
  return out;
}

mlx_array dc_conv_transpose1d_ncl(mlx_array x, mlx_array weight, mlx_array bias,
                                  int stride, int padding, mlx_stream s) {
  int axes[3] = {0, 2, 1};
  mlx_array xt = dc_transpose(x, axes, 3, s);
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_conv_transpose1d(&y, xt, weight, stride, padding, 1, 0, 1, s));
  mlx_array yb = dc_add(y, bias, s);
  mlx_array out = dc_transpose(yb, axes, 3, s);
  mlx_array_free(xt);
  mlx_array_free(y);
  mlx_array_free(yb);
  return out;
}

mlx_array dc_conv_transpose2d_nchw(mlx_array x, mlx_array weight, mlx_array bias,
                                   int stride0, int stride1, mlx_stream s) {
  int axes_in[4] = {0, 2, 3, 1};
  mlx_array xt = dc_transpose(x, axes_in, 4, s);
  mlx_array xc = dc_contiguous(xt, s);
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_conv_transpose2d(&y, xc, weight, stride0, stride1, 0, 0, 1, 1, 0,
                                0, 1, s));
  mlx_array yb = dc_add(y, bias, s);
  int axes_out[4] = {0, 3, 1, 2};
  mlx_array ot = dc_transpose(yb, axes_out, 4, s);
  mlx_array out = dc_contiguous(ot, s);
  mlx_array_free(xt);
  mlx_array_free(xc);
  mlx_array_free(y);
  mlx_array_free(yb);
  mlx_array_free(ot);
  return out;
}

mlx_array dc_linear(mlx_array x, mlx_array weight, mlx_array bias, mlx_stream s) {
  int axes[2] = {1, 0};
  mlx_array wt = dc_transpose(weight, axes, 2, s);
  mlx_array y = dc_matmul(x, wt, s);
  mlx_array out = dc_add(y, bias, s);
  mlx_array_free(wt);
  mlx_array_free(y);
  return out;
}

mlx_array dc_layer_norm(mlx_array x, mlx_array weight, mlx_array bias, float eps,
                        mlx_stream s) {
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_fast_layer_norm(&y, x, weight, bias, eps, s));
  return y;
}

mlx_array dc_embedding(mlx_array weight, mlx_array indices, mlx_stream s) {
  return dc_take_axis(weight, indices, 0, s);
}

mlx_array dc_mha(mlx_array q, mlx_array k, mlx_array v, mlx_array wq,
                 mlx_array bq, mlx_array wk, mlx_array bk, mlx_array wv,
                 mlx_array bv, mlx_array wo, mlx_array bo, int nheads,
                 mlx_stream s) {
  mlx_array queries = dc_linear(q, wq, bq, s);
  mlx_array keys = dc_linear(k, wk, bk, s);
  mlx_array values = dc_linear(v, wv, bv, s);
  int B = dc_dim(queries, 0);
  int Tq = dc_dim(queries, 1);
  int dim = dc_dim(queries, 2);
  if (dim % nheads != 0) {
    dc_die("mha dim %d not divisible by heads %d", dim, nheads);
  }
  int head_dim = dim / nheads;
  int ushape[2] = {nheads, head_dim};
  mlx_array qh = dc_unflatten(queries, -1, ushape, 2, s);
  mlx_array kh = dc_unflatten(keys, -1, ushape, 2, s);
  mlx_array vh = dc_unflatten(values, -1, ushape, 2, s);
  int tax[4] = {0, 2, 1, 3};
  mlx_array qt = dc_transpose(qh, tax, 4, s);
  mlx_array kt = dc_transpose(kh, tax, 4, s);
  mlx_array vt = dc_transpose(vh, tax, 4, s);
  float scale = 1.0f / sqrtf((float)head_dim);
  mlx_array attn = mlx_array_new();
  DC_CHECK(mlx_fast_scaled_dot_product_attention(&attn, qt, kt, vt, scale, "",
                                                 mlx_array_empty, mlx_array_empty,
                                                 false, s));
  int tax2[4] = {0, 2, 1, 3};
  mlx_array at = dc_transpose(attn, tax2, 4, s);
  mlx_array flat = dc_flatten(at, -2, -1, s);
  mlx_array out = dc_linear(flat, wo, bo, s);
  (void)B;
  (void)Tq;
  mlx_array_free(queries);
  mlx_array_free(keys);
  mlx_array_free(values);
  mlx_array_free(qh);
  mlx_array_free(kh);
  mlx_array_free(vh);
  mlx_array_free(qt);
  mlx_array_free(kt);
  mlx_array_free(vt);
  mlx_array_free(attn);
  mlx_array_free(at);
  mlx_array_free(flat);
  return out;
}

static void pref_key(char *buf, size_t cap, const char *prefix, const char *suffix) {
  snprintf(buf, cap, "%s.%s", prefix, suffix);
}

mlx_array dc_conv1d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         int stride, int padding, int dilation, mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "conv.weight");
  pref_key(bk, sizeof(bk), prefix, "conv.bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_conv1d_ncl(x, weight, bias, stride, padding, dilation, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

mlx_array dc_conv2d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         int stride0, int stride1, int pad0, int pad1,
                         mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "conv.weight");
  pref_key(bk, sizeof(bk), prefix, "conv.bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_conv2d_nchw(x, weight, bias, stride0, stride1, pad0, pad1, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

mlx_array dc_convtr1d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                           int stride, int padding, mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "conv.weight");
  pref_key(bk, sizeof(bk), prefix, "conv.bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_conv_transpose1d_ncl(x, weight, bias, stride, padding, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

mlx_array dc_convtr2d_pref(mlx_array x, const DcWeights *w, const char *prefix,
                           int stride0, int stride1, mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "conv.weight");
  pref_key(bk, sizeof(bk), prefix, "conv.bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_conv_transpose2d_nchw(x, weight, bias, stride0, stride1, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

mlx_array dc_linear_pref(mlx_array x, const DcWeights *w, const char *prefix,
                         mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "weight");
  pref_key(bk, sizeof(bk), prefix, "bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_linear(x, weight, bias, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

mlx_array dc_ln_pref(mlx_array x, const DcWeights *w, const char *prefix,
                     mlx_stream s) {
  char wk[256], bk[256];
  pref_key(wk, sizeof(wk), prefix, "weight");
  pref_key(bk, sizeof(bk), prefix, "bias");
  mlx_array weight = dc_require_weight(w->tensors, wk);
  mlx_array bias = dc_require_weight(w->tensors, bk);
  mlx_array y = dc_layer_norm(x, weight, bias, 1e-5f, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}
