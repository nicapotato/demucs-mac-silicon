#include "transformer.h"

#include "metal_kernels.h"

#include <math.h>

mlx_array dc_groupnorm_last(mlx_array x, mlx_array weight, mlx_array bias,
                            int num_groups, float eps, mlx_stream s);

static mlx_array layer_scale_last(mlx_array x, const DcWeights *w,
                                  const char *prefix, mlx_stream s) {
  char key[256];
  snprintf(key, sizeof(key), "%s.scale", prefix);
  if (!dc_has_weight(w->tensors, key)) {
    return dc_copy(x);
  }
  mlx_array scale = dc_require_weight(w->tensors, key);
  mlx_array y = dc_mul(x, scale, s);
  mlx_array_free(scale);
  return y;
}

static mlx_array gelu_or_relu(mlx_array x, int use_gelu, mlx_stream s) {
  if (use_gelu) {
    return dc_gelu(x, s);
  }
  mlx_array zero = mlx_array_new_float(0.0f);
  mlx_array y = dc_maximum(x, zero, s);
  mlx_array_free(zero);
  return y;
}

static mlx_array mha_pref(mlx_array q, mlx_array k, mlx_array v,
                          const DcWeights *w, const char *prefix, int nheads,
                          mlx_stream s) {
  char qwp[256], kwp[256], vwp[256], owp[256];
  snprintf(qwp, sizeof(qwp), "%s.query_proj", prefix);
  snprintf(kwp, sizeof(kwp), "%s.key_proj", prefix);
  snprintf(vwp, sizeof(vwp), "%s.value_proj", prefix);
  snprintf(owp, sizeof(owp), "%s.out_proj", prefix);
  mlx_array qq = dc_linear_pref(q, w, qwp, s);
  mlx_array kk = dc_linear_pref(k, w, kwp, s);
  mlx_array vv = dc_linear_pref(v, w, vwp, s);
  /* reuse dc_mha internals via projections already applied: fake by calling
     scaled dot product directly */
  int dim = dc_dim(qq, 2);
  if (dim % nheads != 0) {
    dc_die("attn dim");
  }
  int head_dim = dim / nheads;
  int ushape[2] = {nheads, head_dim};
  mlx_array qh = dc_unflatten(qq, -1, ushape, 2, s);
  mlx_array kh = dc_unflatten(kk, -1, ushape, 2, s);
  mlx_array vh = dc_unflatten(vv, -1, ushape, 2, s);
  int tax[4] = {0, 2, 1, 3};
  mlx_array qt = dc_transpose(qh, tax, 4, s);
  mlx_array kt = dc_transpose(kh, tax, 4, s);
  mlx_array vt = dc_transpose(vh, tax, 4, s);
  float scale = 1.0f / sqrtf((float)head_dim);
  mlx_array attn = mlx_array_new();
  DC_CHECK(mlx_fast_scaled_dot_product_attention(&attn, qt, kt, vt, scale, "",
                                                 mlx_array_empty, mlx_array_empty,
                                                 false, s));
  mlx_array at = dc_transpose(attn, tax, 4, s);
  mlx_array flat = dc_flatten(at, -2, -1, s);
  mlx_array out = dc_linear_pref(flat, w, owp, s);
  mlx_array_free(qq);
  mlx_array_free(kk);
  mlx_array_free(vv);
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

static mlx_array maybe_norm_out(mlx_array x, const DcWeights *w,
                                const char *layer_prefix, mlx_stream s) {
  char key[256];
  snprintf(key, sizeof(key), "%s.norm_out.gn.weight", layer_prefix);
  if (!dc_has_weight(w->tensors, key)) {
    return dc_copy(x);
  }
  mlx_array weight = dc_require_weight(w->tensors, key);
  snprintf(key, sizeof(key), "%s.norm_out.gn.bias", layer_prefix);
  mlx_array bias = dc_require_weight(w->tensors, key);
  /* MyGroupNorm wraps nn.GroupNorm which expects channel-last in MLX default?
     pytorch_compatible GroupNorm uses channel-first for NCHW-style but
     transformer tensors here are (B, T, C) channel-last.
     nn.GroupNorm in MLX normalizes over groups on the last dim when
     pytorch_compatible? Check: MLX GroupNorm uses the feature axis. */
  mlx_array y = dc_groupnorm_last(x, weight, bias, 1, 1e-5f, s);
  mlx_array_free(weight);
  mlx_array_free(bias);
  return y;
}

static mlx_array self_attn_layer(mlx_array x, const DcWeights *w,
                                 const char *prefix, const DcConfig *cfg,
                                 mlx_stream s) {
  char n1[192], n2[192], attn[192], l1[192], l2[192], g1[192], g2[192];
  snprintf(n1, sizeof(n1), "%s.norm1", prefix);
  snprintf(n2, sizeof(n2), "%s.norm2", prefix);
  snprintf(attn, sizeof(attn), "%s.attn", prefix);
  snprintf(l1, sizeof(l1), "%s.linear1", prefix);
  snprintf(l2, sizeof(l2), "%s.linear2", prefix);
  snprintf(g1, sizeof(g1), "%s.gamma_1", prefix);
  snprintf(g2, sizeof(g2), "%s.gamma_2", prefix);
  mlx_array xn = dc_ln_pref(x, w, n1, s);
  mlx_array a = mha_pref(xn, xn, xn, w, attn, cfg->t_heads, s);
  mlx_array ag = layer_scale_last(a, w, g1, s);
  mlx_array x1 = dc_add(x, ag, s);
  mlx_array x2n = dc_ln_pref(x1, w, n2, s);
  mlx_array h = dc_linear_pref(x2n, w, l1, s);
  mlx_array ha = gelu_or_relu(h, cfg->t_gelu, s);
  mlx_array h2 = dc_linear_pref(ha, w, l2, s);
  mlx_array hg = layer_scale_last(h2, w, g2, s);
  mlx_array y = dc_add(x1, hg, s);
  mlx_array yo = maybe_norm_out(y, w, prefix, s);
  mlx_array_free(xn);
  mlx_array_free(a);
  mlx_array_free(ag);
  mlx_array_free(x1);
  mlx_array_free(x2n);
  mlx_array_free(h);
  mlx_array_free(ha);
  mlx_array_free(h2);
  mlx_array_free(hg);
  mlx_array_free(y);
  return yo;
}

static mlx_array cross_attn_layer(mlx_array q, mlx_array k, const DcWeights *w,
                                  const char *prefix, const DcConfig *cfg,
                                  mlx_stream s) {
  char n1[192], n2[192], n3[192], attn[192], l1[192], l2[192], g1[192], g2[192];
  snprintf(n1, sizeof(n1), "%s.norm1", prefix);
  snprintf(n2, sizeof(n2), "%s.norm2", prefix);
  snprintf(n3, sizeof(n3), "%s.norm3", prefix);
  snprintf(attn, sizeof(attn), "%s.cross_attn", prefix);
  snprintf(l1, sizeof(l1), "%s.linear1", prefix);
  snprintf(l2, sizeof(l2), "%s.linear2", prefix);
  snprintf(g1, sizeof(g1), "%s.gamma_1", prefix);
  snprintf(g2, sizeof(g2), "%s.gamma_2", prefix);
  mlx_array qn = dc_ln_pref(q, w, n1, s);
  mlx_array kn = dc_ln_pref(k, w, n2, s);
  mlx_array a = mha_pref(qn, kn, kn, w, attn, cfg->t_heads, s);
  mlx_array ag = layer_scale_last(a, w, g1, s);
  mlx_array x = dc_add(q, ag, s);
  mlx_array x3 = dc_ln_pref(x, w, n3, s);
  mlx_array h = dc_linear_pref(x3, w, l1, s);
  mlx_array ha = gelu_or_relu(h, cfg->t_gelu, s);
  mlx_array h2 = dc_linear_pref(ha, w, l2, s);
  mlx_array hg = layer_scale_last(h2, w, g2, s);
  mlx_array y = dc_add(x, hg, s);
  mlx_array yo = maybe_norm_out(y, w, prefix, s);
  mlx_array_free(qn);
  mlx_array_free(kn);
  mlx_array_free(a);
  mlx_array_free(ag);
  mlx_array_free(x);
  mlx_array_free(x3);
  mlx_array_free(h);
  mlx_array_free(ha);
  mlx_array_free(h2);
  mlx_array_free(hg);
  mlx_array_free(y);
  return yo;
}

mlx_array dc_groupnorm_last(mlx_array x, mlx_array weight, mlx_array bias,
                            int num_groups, float eps, mlx_stream s) {
  /* Match MLX nn.GroupNorm(pytorch_compatible=True): channels are last,
     stats are over all non-batch dims inside each group. */
  int n = dc_ndim(x);
  int B = dc_dim(x, 0);
  int C = dc_dim(x, n - 1);
  if (C % num_groups != 0) {
    dc_die("groupnorm_last C %d groups %d", C, num_groups);
  }
  int cpg = C / num_groups;
  int spatial = 1;
  for (int i = 1; i < n - 1; i++) {
    spatial *= dc_dim(x, i);
  }
  int r4[4] = {B, spatial, num_groups, cpg};
  mlx_array xr = dc_reshape(x, r4, 4, s);
  int tax[4] = {0, 2, 1, 3};
  mlx_array xt = dc_transpose(xr, tax, 4, s);
  int r3[3] = {B, num_groups, spatial * cpg};
  mlx_array xf = dc_reshape(xt, r3, 3, s);
  mlx_array ln = dc_layer_norm(xf, mlx_array_empty, mlx_array_empty, eps, s);
  int r4b[4] = {B, num_groups, spatial, cpg};
  mlx_array xb = dc_reshape(ln, r4b, 4, s);
  mlx_array xbt = dc_transpose(xb, tax, 4, s);
  int orig[DC_MAX_DIM];
  for (int i = 0; i < n; i++) {
    orig[i] = dc_dim(x, i);
  }
  mlx_array xout = dc_reshape(xbt, orig, n, s);
  mlx_array y = dc_mul(xout, weight, s);
  mlx_array yb = dc_add(y, bias, s);
  mlx_array_free(xr);
  mlx_array_free(xt);
  mlx_array_free(xf);
  mlx_array_free(ln);
  mlx_array_free(xb);
  mlx_array_free(xbt);
  mlx_array_free(xout);
  mlx_array_free(y);
  return yb;
}

static mlx_array sin_embedding(int length, int dim, float max_period,
                               mlx_stream s) {
  if (dim % 2 != 0) {
    dc_die("sin embed dim must be even");
  }
  int half = dim / 2;
  mlx_array pos = dc_arange(length, MLX_FLOAT32, s); /* [T] */
  mlx_array pos3 = dc_reshape(pos, (int[]){length, 1, 1}, 3, s);
  mlx_array adim = dc_arange(half, MLX_FLOAT32, s);
  mlx_array adim3 = dc_reshape(adim, (int[]){1, 1, half}, 3, s);
  /* phase = pos / (max_period ** (adim / (half_dim - 1))) */
  mlx_array hp = mlx_array_new_float((float)(half - 1));
  mlx_array ratio = dc_div(adim3, hp, s);
  mlx_array base = mlx_array_new_float(max_period);
  mlx_array den = dc_power(base, ratio, s);
  mlx_array phase = dc_div(pos3, den, s);
  mlx_array cs = dc_cos(phase, s);
  mlx_array sn = dc_sin(phase, s);
  mlx_array parts[2] = {cs, sn};
  mlx_array pe = dc_concat(parts, 2, -1, s);
  mlx_array_free(pos);
  mlx_array_free(pos3);
  mlx_array_free(adim);
  mlx_array_free(adim3);
  mlx_array_free(hp);
  mlx_array_free(ratio);
  mlx_array_free(base);
  mlx_array_free(den);
  mlx_array_free(phase);
  mlx_array_free(cs);
  mlx_array_free(sn);
  return pe; /* [T, 1, C] */
}

static mlx_array sin_embedding_2d(int d_model, int height, int width,
                                  float max_period, mlx_stream s) {
  if (d_model % 4 != 0) {
    dc_die("2d sin embed dim must be divisible by 4");
  }
  int half = d_model / 2;
  /* div_term = exp(arange(0, half, 2) * -(log(max_period)/half)) */
  int nterm = half / 2;
  mlx_array idx = mlx_array_new();
  DC_CHECK(mlx_arange(&idx, 0.0, (double)half, 2.0, MLX_FLOAT32, s));
  float scale = -logf(max_period) / (float)half;
  mlx_array sc = mlx_array_new_float(scale);
  mlx_array dt = dc_mul(idx, sc, s);
  mlx_array e = mlx_array_new();
  DC_CHECK(mlx_exp(&e, dt, s));
  mlx_array pos_w = dc_arange(width, MLX_FLOAT32, s);
  mlx_array pos_h = dc_arange(height, MLX_FLOAT32, s);
  mlx_array pw = dc_reshape(pos_w, (int[]){width, 1}, 2, s);
  mlx_array ph = dc_reshape(pos_h, (int[]){height, 1}, 2, s);
  mlx_array dt1 = dc_reshape(e, (int[]){1, nterm}, 2, s);
  mlx_array ang_w = dc_mul(pw, dt1, s); /* [W, nterm] */
  mlx_array ang_h = dc_mul(ph, dt1, s); /* [H, nterm] */
  mlx_array sin_w = dc_sin(ang_w, s);
  mlx_array cos_w = dc_cos(ang_w, s);
  mlx_array sin_h = dc_sin(ang_h, s);
  mlx_array cos_h = dc_cos(ang_h, s);
  /* Python: sin_w.transpose(1,0).reshape(-1,1,W) then broadcast H */
  int tax[2] = {1, 0};
  mlx_array swt = dc_transpose(sin_w, tax, 2, s); /* [nterm, W] */
  mlx_array cwt = dc_transpose(cos_w, tax, 2, s);
  mlx_array sht = dc_transpose(sin_h, tax, 2, s); /* [nterm, H] */
  mlx_array cht = dc_transpose(cos_h, tax, 2, s);
  mlx_array sw3 = dc_reshape(swt, (int[]){nterm, 1, width}, 3, s);
  mlx_array cw3 = dc_reshape(cwt, (int[]){nterm, 1, width}, 3, s);
  mlx_array sh3 = dc_reshape(sht, (int[]){nterm, height, 1}, 3, s);
  mlx_array ch3 = dc_reshape(cht, (int[]){nterm, height, 1}, 3, s);
  int bshape[3] = {nterm, height, width};
  mlx_array swb = dc_broadcast_to(sw3, bshape, 3, s);
  mlx_array cwb = dc_broadcast_to(cw3, bshape, 3, s);
  mlx_array shb = dc_broadcast_to(sh3, bshape, 3, s);
  mlx_array chb = dc_broadcast_to(ch3, bshape, 3, s);
  mlx_array stack_w[2] = {swb, cwb};
  mlx_array stw = dc_stack(stack_w, 2, 1, s); /* [nterm, 2, H, W] */
  mlx_array pew = dc_reshape(stw, (int[]){half, height, width}, 3, s);
  mlx_array stack_h[2] = {shb, chb};
  mlx_array sth = dc_stack(stack_h, 2, 1, s);
  mlx_array peh = dc_reshape(sth, (int[]){half, height, width}, 3, s);
  mlx_array parts[2] = {pew, peh};
  mlx_array pe = dc_concat(parts, 2, 0, s); /* [d_model, H, W] */
  mlx_array pe4 = dc_expand_dims(pe, 0, s); /* [1, C, H, W] */
  mlx_array_free(idx);
  mlx_array_free(sc);
  mlx_array_free(dt);
  mlx_array_free(e);
  mlx_array_free(pos_w);
  mlx_array_free(pos_h);
  mlx_array_free(pw);
  mlx_array_free(ph);
  mlx_array_free(dt1);
  mlx_array_free(ang_w);
  mlx_array_free(ang_h);
  mlx_array_free(sin_w);
  mlx_array_free(cos_w);
  mlx_array_free(sin_h);
  mlx_array_free(cos_h);
  mlx_array_free(swt);
  mlx_array_free(cwt);
  mlx_array_free(sht);
  mlx_array_free(cht);
  mlx_array_free(sw3);
  mlx_array_free(cw3);
  mlx_array_free(sh3);
  mlx_array_free(ch3);
  mlx_array_free(swb);
  mlx_array_free(cwb);
  mlx_array_free(shb);
  mlx_array_free(chb);
  mlx_array_free(stw);
  mlx_array_free(pew);
  mlx_array_free(sth);
  mlx_array_free(peh);
  mlx_array_free(pe);
  return pe4;
}

void dc_crosstransformer(mlx_array *x_io, mlx_array *xt_io, const DcWeights *w,
                         const DcConfig *cfg, mlx_stream s) {
  mlx_array x = *x_io;
  mlx_array xt = *xt_io;
  int B = dc_dim(x, 0);
  int C = dc_dim(x, 1);
  int Fr = dc_dim(x, 2);
  int T1 = dc_dim(x, 3);
  mlx_array pe2d = sin_embedding_2d(C, Fr, T1, cfg->t_max_period, s);
  int bshape[4] = {B, C, Fr, T1};
  mlx_array pe_b = dc_broadcast_to(pe2d, bshape, 4, s);
  int tax[4] = {0, 3, 2, 1};
  mlx_array pe_t = dc_transpose(pe_b, tax, 4, s);
  mlx_array pe_flat = dc_reshape(pe_t, (int[]){B, T1 * Fr, C}, 3, s);
  mlx_array xtf = dc_transpose(x, tax, 4, s);
  mlx_array xf = dc_reshape(xtf, (int[]){B, T1 * Fr, C}, 3, s);
  mlx_array xn = dc_ln_pref(xf, w, "crosstransformer.norm_in", s);
  mlx_array wpe = dc_mul_scalar(pe_flat, cfg->t_weight_pos_embed, s);
  mlx_array xpe = dc_add(xn, wpe, s);

  int Ct = dc_dim(xt, 1);
  int T2 = dc_dim(xt, 2);
  int tax2[3] = {0, 2, 1};
  mlx_array xtt = dc_transpose(xt, tax2, 3, s);
  mlx_array pos = sin_embedding(T2, Ct, cfg->t_max_period, s); /* [T,1,C] */
  int pax[3] = {1, 0, 2};
  mlx_array posb = dc_transpose(pos, pax, 3, s); /* [1,T,C] */
  mlx_array xtn = dc_ln_pref(xtt, w, "crosstransformer.norm_in_t", s);
  mlx_array posw = dc_mul_scalar(posb, cfg->t_weight_pos_embed, s);
  mlx_array xtpe = dc_add(xtn, posw, s);

  int classic_parity = cfg->t_cross_first ? 1 : 0;
  mlx_array cur_x = xpe;
  mlx_array cur_xt = xtpe;
  for (int idx = 0; idx < cfg->t_layers; idx++) {
    char lp[128], lpt[128];
    snprintf(lp, sizeof(lp), "crosstransformer.layers.%d", idx);
    snprintf(lpt, sizeof(lpt), "crosstransformer.layers_t.%d", idx);
    if (idx % 2 == classic_parity) {
      mlx_array nx = self_attn_layer(cur_x, w, lp, cfg, s);
      mlx_array nxt = self_attn_layer(cur_xt, w, lpt, cfg, s);
      mlx_array_free(cur_x);
      mlx_array_free(cur_xt);
      cur_x = nx;
      cur_xt = nxt;
    } else {
      mlx_array old_x = dc_copy(cur_x);
      mlx_array nx = cross_attn_layer(cur_x, cur_xt, w, lp, cfg, s);
      mlx_array nxt = cross_attn_layer(cur_xt, old_x, w, lpt, cfg, s);
      mlx_array_free(cur_x);
      mlx_array_free(cur_xt);
      mlx_array_free(old_x);
      cur_x = nx;
      cur_xt = nxt;
    }
  }
  mlx_array xr = dc_reshape(cur_x, (int[]){B, T1, Fr, C}, 4, s);
  int tax3[4] = {0, 3, 2, 1};
  mlx_array xout = dc_transpose(xr, tax3, 4, s);
  mlx_array xtout = dc_transpose(cur_xt, tax2, 3, s);

  mlx_array_free(pe2d);
  mlx_array_free(pe_b);
  mlx_array_free(pe_t);
  mlx_array_free(pe_flat);
  mlx_array_free(xtf);
  mlx_array_free(xf);
  mlx_array_free(xn);
  mlx_array_free(wpe);
  mlx_array_free(xtt);
  mlx_array_free(pos);
  mlx_array_free(posb);
  mlx_array_free(xtn);
  mlx_array_free(posw);
  mlx_array_free(cur_x);
  mlx_array_free(cur_xt);
  mlx_array_free(xr);

  mlx_array_free(*x_io);
  mlx_array_free(*xt_io);
  *x_io = xout;
  *xt_io = xtout;
}
