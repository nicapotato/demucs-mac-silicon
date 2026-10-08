#include "layers.h"

static mlx_array gn_weight(const DcWeights *w, const char *prefix,
                           const char *name) {
  char key[256];
  snprintf(key, sizeof(key), "%s.%s", prefix, name);
  return dc_require_weight(w->tensors, key);
}

static mlx_array freq_to_ncl(mlx_array y, mlx_stream s) {
  int B = dc_dim(y, 0);
  int C = dc_dim(y, 1);
  int Fr = dc_dim(y, 2);
  int T = dc_dim(y, 3);
  int tax[4] = {0, 2, 1, 3};
  mlx_array yt = dc_transpose(y, tax, 4, s);
  mlx_array yc = dc_contiguous(yt, s);
  int r[3] = {B * Fr, C, T};
  mlx_array yr = dc_reshape(yc, r, 3, s);
  mlx_array_free(yt);
  mlx_array_free(yc);
  return yr;
}

static mlx_array ncl_to_freq(mlx_array yd, int B, int Fr, int C, int T,
                             mlx_stream s) {
  int r4[4] = {B, Fr, C, T};
  mlx_array yb = dc_reshape(yd, r4, 4, s);
  mlx_array yc = dc_contiguous(yb, s);
  int tax2[4] = {0, 2, 1, 3};
  mlx_array y2 = dc_transpose(yc, tax2, 4, s);
  mlx_array_free(yb);
  mlx_array_free(yc);
  return y2;
}

mlx_array dc_dconv(mlx_array x, const DcWeights *w, const char *prefix,
                   int channels, int depth, int compress, mlx_stream s) {
  int hidden = channels / compress;
  mlx_array y = dc_copy(x);
  for (int d = 0; d < depth; d++) {
    int dilation = 1 << d;
    int padding = dilation * (3 / 2);
    char p0[256], p1[256], p3[256], p4[256], p6[256];
    snprintf(p0, sizeof(p0), "%s.layers.%d.layers.0", prefix, d);
    snprintf(p1, sizeof(p1), "%s.layers.%d.layers.1", prefix, d);
    snprintf(p3, sizeof(p3), "%s.layers.%d.layers.3", prefix, d);
    snprintf(p4, sizeof(p4), "%s.layers.%d.layers.4", prefix, d);
    snprintf(p6, sizeof(p6), "%s.layers.%d.layers.6.scale", prefix, d);
    mlx_array h = dc_conv1d_pref(y, w, p0, 1, padding, dilation, s);
    mlx_array w1 = gn_weight(w, p1, "weight");
    mlx_array b1 = gn_weight(w, p1, "bias");
    mlx_array h1 = dc_fused_groupnorm_gelu(h, w1, b1, 1, 1e-5f, s);
    mlx_array h2 = dc_conv1d_pref(h1, w, p3, 1, 0, 1, s);
    mlx_array w2 = gn_weight(w, p4, "weight");
    mlx_array b2 = gn_weight(w, p4, "bias");
    mlx_array h3 = dc_fused_groupnorm_glu(h2, w2, b2, 1, 1e-5f, s);
    mlx_array scale = dc_require_weight(w->tensors, p6);
    int sshape[2] = {dc_dim(scale, 0), 1};
    mlx_array sc = dc_reshape(scale, sshape, 2, s);
    mlx_array scaled = dc_mul(h3, sc, s);
    mlx_array y2 = dc_add(y, scaled, s);
    mlx_array_free(h);
    mlx_array_free(w1);
    mlx_array_free(b1);
    mlx_array_free(h1);
    mlx_array_free(h2);
    mlx_array_free(w2);
    mlx_array_free(b2);
    mlx_array_free(h3);
    mlx_array_free(scale);
    mlx_array_free(sc);
    mlx_array_free(scaled);
    mlx_array_free(y);
    y = y2;
    (void)hidden;
  }
  return y;
}

mlx_array dc_henc(mlx_array x_in, mlx_array inject, const DcWeights *w,
                  const DcLayerSpec *spec, mlx_stream s) {
  mlx_array x = dc_copy(x_in);
  if (!spec->freq && dc_ndim(x) == 4) {
    int B = dc_dim(x, 0);
    int C = dc_dim(x, 1);
    int Fr = dc_dim(x, 2);
    int T = dc_dim(x, 3);
    int r[3] = {B, C * Fr, T};
    mlx_array xr = dc_reshape(x, r, 3, s);
    mlx_array_free(x);
    x = xr;
  }
  if (!spec->freq) {
    int le = dc_dim(x, -1);
    if (le % spec->stride != 0) {
      mlx_array xp = dc_pad1d(x, 0, spec->stride - (le % spec->stride),
                              "constant", s);
      mlx_array_free(x);
      x = xp;
    }
  }
  char convp[160];
  snprintf(convp, sizeof(convp), "%s.conv", spec->prefix);
  mlx_array y;
  if (spec->freq) {
    int pad = spec->pad;
    y = dc_conv2d_pref(x, w, convp, spec->stride, 1, pad, 0, s);
  } else {
    y = dc_conv1d_pref(x, w, convp, spec->stride, spec->pad, 1, s);
  }
  mlx_array_free(x);
  if (spec->empty) {
    return y;
  }
  if (inject.ctx) {
    mlx_array inj = dc_copy(inject);
    if (dc_ndim(inj) == 3 && dc_ndim(y) == 4) {
      mlx_array e = dc_expand_dims(inj, 2, s);
      mlx_array_free(inj);
      inj = e;
    }
    mlx_array y2 = dc_add(y, inj, s);
    mlx_array_free(y);
    mlx_array_free(inj);
    y = y2;
  }
  if (spec->fused_norm1) {
    char np[160];
    snprintf(np, sizeof(np), "%s.norm1", spec->prefix);
    mlx_array nw = gn_weight(w, np, "weight");
    mlx_array nb = gn_weight(w, np, "bias");
    mlx_array y2 = dc_fused_groupnorm_gelu(y, nw, nb, spec->norm_groups, 1e-5f, s);
    mlx_array_free(nw);
    mlx_array_free(nb);
    mlx_array_free(y);
    y = y2;
  } else {
    mlx_array y2 = dc_gelu(y, s);
    mlx_array_free(y);
    y = y2;
  }
  if (spec->dconv) {
    char dp[160];
    snprintf(dp, sizeof(dp), "%s.dconv", spec->prefix);
    if (spec->freq) {
      int B = dc_dim(y, 0);
      int C = dc_dim(y, 1);
      int Fr = dc_dim(y, 2);
      int T = dc_dim(y, 3);
      mlx_array yr = freq_to_ncl(y, s);
      mlx_array yd = dc_dconv(yr, w, dp, C, spec->dconv_depth, spec->dconv_comp, s);
      mlx_array y2 = ncl_to_freq(yd, B, Fr, C, T, s);
      mlx_array_free(y);
      mlx_array_free(yr);
      mlx_array_free(yd);
      y = y2;
    } else {
      int C = dc_dim(y, 1);
      mlx_array y2 = dc_dconv(y, w, dp, C, spec->dconv_depth, spec->dconv_comp, s);
      mlx_array_free(y);
      y = y2;
    }
  }
  if (spec->rewrite) {
    char rp[160];
    snprintf(rp, sizeof(rp), "%s.rewrite", spec->prefix);
    mlx_array z;
    if (spec->freq) {
      z = dc_conv2d_pref(y, w, rp, 1, 1, spec->context, spec->context, s);
    } else {
      z = dc_conv1d_pref(y, w, rp, 1, spec->context, 1, s);
    }
    mlx_array z2;
    if (spec->fused_norm2) {
      char np[160];
      snprintf(np, sizeof(np), "%s.norm2", spec->prefix);
      mlx_array nw = gn_weight(w, np, "weight");
      mlx_array nb = gn_weight(w, np, "bias");
      z2 = dc_fused_groupnorm_glu(z, nw, nb, spec->norm_groups, 1e-5f, s);
      mlx_array_free(nw);
      mlx_array_free(nb);
    } else {
      z2 = dc_glu_axis1(z, s);
    }
    mlx_array_free(z);
    mlx_array_free(y);
    y = z2;
  }
  return y;
}

mlx_array dc_hdec(mlx_array x_in, mlx_array skip, int length, const DcWeights *w,
                  const DcLayerSpec *spec, mlx_array *pre_out, mlx_stream s) {
  mlx_array x = dc_copy(x_in);
  if (spec->freq && dc_ndim(x) == 3) {
    int B = dc_dim(x, 0);
    int T = dc_dim(x, 2);
    int r[4] = {B, spec->chin, -1, T};
    /* reshape B, chin, Fr, T where Fr = C/chin */
    int C = dc_dim(x, 1);
    if (C % spec->chin != 0) {
      dc_die("hdec freq reshape %d not divisible by chin %d", C, spec->chin);
    }
    r[2] = C / spec->chin;
    mlx_array xr = dc_reshape(x, r, 4, s);
    mlx_array_free(x);
    x = xr;
  }
  mlx_array y;
  if (!spec->empty) {
    if (!skip.ctx) {
      dc_die("hdec skip is required when empty=0");
    }
    mlx_array xs = dc_add(x, skip, s);
    mlx_array_free(x);
    x = xs;
    if (spec->rewrite) {
      char rp[160];
      snprintf(rp, sizeof(rp), "%s.rewrite", spec->prefix);
      mlx_array z;
      if (spec->freq) {
        if (spec->context_freq) {
          z = dc_conv2d_pref(x, w, rp, 1, 1, spec->context, spec->context, s);
        } else {
          z = dc_conv2d_pref(x, w, rp, 1, 1, 0, spec->context, s);
        }
      } else {
        z = dc_conv1d_pref(x, w, rp, 1, spec->context, 1, s);
      }
      if (spec->fused_norm1) {
        char np[160];
        snprintf(np, sizeof(np), "%s.norm1", spec->prefix);
        mlx_array nw = gn_weight(w, np, "weight");
        mlx_array nb = gn_weight(w, np, "bias");
        y = dc_fused_groupnorm_glu(z, nw, nb, spec->norm_groups, 1e-5f, s);
        mlx_array_free(nw);
        mlx_array_free(nb);
      } else {
        y = dc_glu_axis1(z, s);
      }
      mlx_array_free(z);
    } else {
      y = dc_copy(x);
    }
    if (spec->dconv) {
      char dp[160];
      snprintf(dp, sizeof(dp), "%s.dconv", spec->prefix);
      if (spec->freq) {
        int B = dc_dim(y, 0);
        int C = dc_dim(y, 1);
        int Fr = dc_dim(y, 2);
        int T = dc_dim(y, 3);
        mlx_array yr = freq_to_ncl(y, s);
        mlx_array yd = dc_dconv(yr, w, dp, C, spec->dconv_depth, spec->dconv_comp, s);
        mlx_array y2 = ncl_to_freq(yd, B, Fr, C, T, s);
        mlx_array_free(y);
        mlx_array_free(yr);
        mlx_array_free(yd);
        y = y2;
      } else {
        int C = dc_dim(y, 1);
        mlx_array y2 = dc_dconv(y, w, dp, C, spec->dconv_depth, spec->dconv_comp, s);
        mlx_array_free(y);
        y = y2;
      }
    }
  } else {
    y = dc_copy(x);
  }
  if (pre_out) {
    *pre_out = dc_copy(y);
  }
  char cp[160];
  snprintf(cp, sizeof(cp), "%s.conv_tr", spec->prefix);
  mlx_array z;
  if (spec->freq) {
    z = dc_convtr2d_pref(y, w, cp, spec->stride, 1, s);
  } else {
    z = dc_convtr1d_pref(y, w, cp, spec->stride, 0, s);
  }
  /* decoder norm2 is Identity when norm=False (typical HTDemucs) */
  if (spec->norm && dc_w_has(w, "%s.norm2.weight", spec->prefix)) {
    char np[160];
    snprintf(np, sizeof(np), "%s.norm2", spec->prefix);
    mlx_array nw = gn_weight(w, np, "weight");
    mlx_array nb = gn_weight(w, np, "bias");
    mlx_array zn = dc_groupnorm(z, nw, nb, spec->norm_groups, 1e-5f, s);
    mlx_array_free(z);
    mlx_array_free(nw);
    mlx_array_free(nb);
    z = zn;
  }
  if (spec->freq) {
    if (spec->pad) {
      int Fr = dc_dim(z, 2);
      mlx_array z2 = dc_slice_axis(z, 2, spec->pad, Fr - spec->pad, s);
      mlx_array_free(z);
      z = z2;
    }
  } else {
    mlx_array z2 = dc_slice_last(z, spec->pad, spec->pad + length, s);
    mlx_array_free(z);
    z = z2;
  }
  if (!spec->last) {
    mlx_array zg = dc_gelu(z, s);
    mlx_array_free(z);
    z = zg;
  }
  mlx_array_free(x);
  mlx_array_free(y);
  return z;
}
