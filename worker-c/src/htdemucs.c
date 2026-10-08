#include "htdemucs.h"

static mlx_array dc_squeeze_last(mlx_array a, mlx_stream s) {
  return dc_squeeze_axis(a, -1, s);
}

static void fill_spec(DcLayerSpec *sp, const char *prefix, int chin, int chout,
                      int kernel, int stride, int freq, int empty, int last,
                      int pad, int rewrite, int dconv, int norm, int context,
                      int context_freq, const DcConfig *cfg) {
  memset(sp, 0, sizeof(*sp));
  snprintf(sp->prefix, sizeof(sp->prefix), "%s", prefix);
  sp->chin = chin;
  sp->chout = chout;
  sp->kernel = kernel;
  sp->stride = stride;
  sp->pad = pad ? kernel / 4 : 0;
  sp->freq = freq;
  sp->empty = empty;
  sp->last = last;
  sp->rewrite = rewrite && !empty;
  sp->dconv = dconv && !empty;
  sp->norm = norm;
  sp->fused_norm1 = norm && !empty;
  sp->fused_norm2 = norm && rewrite && !empty;
  sp->context = context;
  sp->context_freq = context_freq;
  sp->norm_groups = cfg->norm_groups;
  sp->dconv_depth = cfg->dconv_depth;
  sp->dconv_comp = cfg->dconv_comp;
  sp->dconv_init = cfg->dconv_init;
}

void dc_model_init(DcModel *m, DcWeights *w) {
  memset(m, 0, sizeof(*m));
  m->w = w;
  m->stream = w->stream;
  m->use_compile = 0;
  m->compiled = mlx_closure_new();
  m->compile_raw = mlx_closure_new();
  const DcConfig *cfg = &w->cfg;
  m->stft = dc_stft_new(cfg->nfft, cfg->hop_length, m->stream);

  int chin = cfg->audio_channels;
  int chin_z = chin * (cfg->cac ? 2 : 1);
  int chout = cfg->channels_time > 0 ? cfg->channels_time : cfg->channels;
  int chout_z = cfg->channels;
  int freqs = cfg->nfft / 2;
  int n_tenc = 0;
  int n_tdec = 0;
  DcLayerSpec enc_tmp[DC_MAX_DEPTH];
  DcLayerSpec dec_tmp[DC_MAX_DEPTH];
  DcLayerSpec tenc_tmp[DC_MAX_DEPTH];
  DcLayerSpec tdec_tmp[DC_MAX_DEPTH];
  int n_enc = 0;
  int n_dec = 0;

  for (int index = 0; index < cfg->depth; index++) {
    int norm = index >= cfg->norm_starts;
    int freq = freqs > 1;
    int stri = cfg->stride;
    int ker = cfg->kernel_size;
    int pad = 1;
    int last_freq = 0;
    if (!freq) {
      ker = cfg->time_stride * 2;
      stri = cfg->time_stride;
    }
    if (freq && freqs <= cfg->kernel_size) {
      ker = freqs;
      pad = 0;
      last_freq = 1;
    }
    char ep[64], tp[64], dp[64], tdp[64];
    snprintf(ep, sizeof(ep), "encoder.%d", n_enc);
    fill_spec(&enc_tmp[n_enc], ep, chin_z, chout_z, ker, stri, freq, 0, 0, pad,
              cfg->rewrite, cfg->dconv_mode & 1, norm, cfg->context_enc, 1, cfg);
    n_enc++;
    if (freq) {
      snprintf(tp, sizeof(tp), "tencoder.%d", n_tenc);
      fill_spec(&tenc_tmp[n_tenc], tp, chin, chout, cfg->kernel_size, cfg->stride,
                0, last_freq, 0, 1, cfg->rewrite, cfg->dconv_mode & 1, norm,
                cfg->context_enc, 1, cfg);
      n_tenc++;
    }
    snprintf(dp, sizeof(dp), "decoder.%d", index); /* temporary; reindexed */
    fill_spec(&dec_tmp[n_dec], dp, chout_z, chin_z, ker, stri, freq, 0,
              index == 0, pad, cfg->rewrite, cfg->dconv_mode & 2, norm,
              cfg->context, 1, cfg);
    n_dec++;
    if (freq) {
      snprintf(tdp, sizeof(tdp), "tdecoder.%d", n_tdec);
      fill_spec(&tdec_tmp[n_tdec], tdp, chout, chin, cfg->kernel_size, cfg->stride,
                0, last_freq, index == 0, 1, cfg->rewrite, cfg->dconv_mode & 2,
                norm, cfg->context, 1, cfg);
      n_tdec++;
    }
    if (index == 0) {
      chin = cfg->audio_channels * cfg->n_sources;
      chin_z = chin * (cfg->cac ? 2 : 1);
    }
    chin = chout;
    chin_z = chout_z;
    chout = cfg->growth * chout;
    chout_z = cfg->growth * chout_z;
    if (freq) {
      if (freqs <= cfg->kernel_size) {
        freqs = 1;
      } else {
        freqs /= cfg->stride;
      }
    }
  }

  /* decoder.insert(0) => reverse construction order for names */
  m->n_enc = n_enc;
  m->n_tenc = n_tenc;
  m->n_dec = n_dec;
  m->n_tdec = n_tdec;
  for (int i = 0; i < n_enc; i++) {
    m->encoder[i] = enc_tmp[i];
  }
  for (int i = 0; i < n_tenc; i++) {
    m->tencoder[i] = tenc_tmp[i];
  }
  for (int i = 0; i < n_dec; i++) {
    m->decoder[i] = dec_tmp[n_dec - 1 - i];
    snprintf(m->decoder[i].prefix, sizeof(m->decoder[i].prefix), "decoder.%d", i);
  }
  for (int i = 0; i < n_tdec; i++) {
    m->tdecoder[i] = tdec_tmp[n_tdec - 1 - i];
    snprintf(m->tdecoder[i].prefix, sizeof(m->tdecoder[i].prefix), "tdecoder.%d",
             i);
  }
}

void dc_model_free(DcModel *m) {
  if (!m) {
    return;
  }
  dc_stft_free(&m->stft);
  if (m->compiled.ctx) {
    mlx_closure_free(m->compiled);
  }
  if (m->compile_raw.ctx) {
    mlx_closure_free(m->compile_raw);
  }
  m->use_compile = 0;
}

void dc_model_enable_compile(DcModel *m) {
  (void)m;
  dc_die("mlx_compile cannot trace the HTDemucs graph (fused Metal kernels + "
         "FFT). Omit --compile; the uncompiled C worker is the supported path");
}

mlx_array dc_model_forward(DcModel *m, mlx_array mix) {
  if (!m->use_compile) {
    return dc_htdemucs_forward(m, mix);
  }
  mlx_vector_array in = mlx_vector_array_new_value(mix);
  mlx_vector_array outv = mlx_vector_array_new();
  DC_CHECK(mlx_closure_apply(&outv, m->compiled, in));
  mlx_array y = mlx_array_new();
  DC_CHECK(mlx_vector_array_get(&y, outv, 0));
  mlx_vector_array_free(in);
  mlx_vector_array_free(outv);
  return y;
}

static mlx_array magnitude_cac(mlx_array z, mlx_stream s) {
  mlx_array re = dc_real(z, s);
  mlx_array im = dc_imag(z, s);
  mlx_array parts[2] = {re, im};
  mlx_array st = dc_stack(parts, 2, 2, s); /* B, C, 2, Fr, T */
  int B = dc_dim(z, 0);
  int C = dc_dim(z, 1);
  int Fr = dc_dim(z, 2);
  int T = dc_dim(z, 3);
  int r[4] = {B, C * 2, Fr, T};
  mlx_array m = dc_reshape(st, r, 4, s);
  mlx_array_free(re);
  mlx_array_free(im);
  mlx_array_free(st);
  return m;
}

static mlx_array mask_cac(mlx_array m, mlx_stream s) {
  /* m: B, S, C, Fr, T where C includes interleaved real/imag (2 * audio_ch) */
  int B = dc_dim(m, 0);
  int S = dc_dim(m, 1);
  int C2 = dc_dim(m, 2);
  int Fr = dc_dim(m, 3);
  int T = dc_dim(m, 4);
  if (C2 % 2 != 0) {
    dc_die("cac mask channels %d not even", C2);
  }
  int r[6] = {B, S, C2 / 2, 2, Fr, T};
  mlx_array u = dc_reshape(m, r, 6, s);
  int tax[6] = {0, 1, 2, 4, 5, 3};
  mlx_array t = dc_transpose(u, tax, 6, s);
  mlx_array real = dc_slice_axis(t, 5, 0, 1, s);
  mlx_array imag = dc_slice_axis(t, 5, 1, 2, s);
  mlx_array rs = dc_squeeze_last(real, s);
  mlx_array is_ = dc_squeeze_last(imag, s);
  mlx_array z = dc_complex(rs, is_, s);
  mlx_array_free(u);
  mlx_array_free(t);
  mlx_array_free(real);
  mlx_array_free(imag);
  mlx_array_free(rs);
  mlx_array_free(is_);
  (void)Fr;
  return z;
}

mlx_array dc_htdemucs_forward(DcModel *m, mlx_array mix_in) {
  const DcConfig *cfg = &m->w->cfg;
  mlx_stream s = m->stream;
  mlx_array mix = dc_copy(mix_in);
  int length = dc_dim(mix, -1);
  int length_pre_pad = 0;
  int training_length = cfg->valid_length;
  if (cfg->use_train_segment && length < training_length) {
    length_pre_pad = length;
    mlx_array p = dc_pad_last(mix, 0, training_length - length, "constant", s);
    mlx_array_free(mix);
    mix = p;
  }
  mlx_array z = dc_htdemucs_spec(&m->stft, mix, s);
  mlx_array mag = magnitude_cac(z, s);
  int axes3[3] = {1, 2, 3};
  mlx_array mean = dc_mean_axes(mag, axes3, 3, true, s);
  mlx_array stdv = dc_std_axes(mag, axes3, 3, true, s);
  mlx_array stde = dc_add_scalar(stdv, 1e-5f, s);
  mlx_array mag_c = dc_sub(mag, mean, s);
  mlx_array x = dc_div(mag_c, stde, s);
  mlx_array_free(mag_c);
  int axes2[2] = {1, 2};
  mlx_array meant = dc_mean_axes(mix, axes2, 2, true, s);
  mlx_array stdt = dc_std_axes(mix, axes2, 2, true, s);
  mlx_array stdte = dc_add_scalar(stdt, 1e-5f, s);
  mlx_array mix_c = dc_sub(mix, meant, s);
  mlx_array xt = dc_div(mix_c, stdte, s);
  mlx_array_free(mix_c);

  mlx_array saved[DC_MAX_DEPTH];
  mlx_array saved_t[DC_MAX_DEPTH];
  int lengths[DC_MAX_DEPTH];
  int lengths_t[DC_MAX_DEPTH];
  int n_saved = 0;
  int n_saved_t = 0;
  int n_len = 0;
  int n_lent = 0;

  for (int idx = 0; idx < m->n_enc; idx++) {
    lengths[n_len++] = dc_dim(x, -1);
    mlx_array inject = mlx_array_empty;
    if (idx < m->n_tenc) {
      lengths_t[n_lent++] = dc_dim(xt, -1);
      mlx_array xt2 = dc_henc(xt, mlx_array_empty, m->w, &m->tencoder[idx], s);
      mlx_array_free(xt);
      xt = xt2;
      if (!m->tencoder[idx].empty) {
        saved_t[n_saved_t++] = dc_copy(xt);
      } else {
        inject = xt;
      }
    }
    mlx_array x2 = dc_henc(x, inject, m->w, &m->encoder[idx], s);
    mlx_array_free(x);
    x = x2;
    if (idx == 0 && cfg->freq_emb != 0.0f &&
        dc_w_has(m->w, "freq_emb.embedding.weight")) {
      int Fr = dc_dim(x, 2);
      mlx_array frs = dc_arange(Fr, MLX_INT32, s);
      mlx_array wemb = dc_w(m->w, "freq_emb.embedding.weight");
      mlx_array emb = dc_embedding(wemb, frs, s); /* [Fr, C] */
      int tax[2] = {1, 0};
      mlx_array embt = dc_transpose(emb, tax, 2, s); /* [C, Fr] */
      mlx_array e3 = dc_expand_dims(embt, 0, s);
      mlx_array e4 = dc_expand_dims(e3, 3, s); /* [1,C,Fr,1] */
      mlx_array scaled = dc_mul_scalar(e4, cfg->freq_emb * cfg->emb_scale, s);
      mlx_array x3 = dc_add(x, scaled, s);
      mlx_array_free(x);
      x = x3;
      mlx_array_free(frs);
      mlx_array_free(wemb);
      mlx_array_free(emb);
      mlx_array_free(embt);
      mlx_array_free(e3);
      mlx_array_free(e4);
      mlx_array_free(scaled);
    }
    saved[n_saved++] = dc_copy(x);
  }

  if (cfg->t_layers > 0) {
    if (cfg->bottom_channels) {
      dc_die("bottom_channels != 0 is not supported in the C worker");
    }
    dc_crosstransformer(&x, &xt, m->w, cfg, s);
  }

  int offset = m->n_enc - m->n_tdec;
  for (int idx = 0; idx < m->n_dec; idx++) {
    if (n_saved <= 0 || n_len <= 0) {
      dc_die("decoder skip underflow");
    }
    mlx_array skip = saved[--n_saved];
    int len = lengths[--n_len];
    mlx_array pre = mlx_array_empty;
    mlx_array x2 = dc_hdec(x, skip, len, m->w, &m->decoder[idx], &pre, s);
    mlx_array_free(x);
    mlx_array_free(skip);
    x = x2;
    if (idx >= offset) {
      DcLayerSpec *tdec = &m->tdecoder[idx - offset];
      if (n_lent <= 0) {
        dc_die("tdecoder length underflow");
      }
      int length_t = lengths_t[--n_lent];
      if (tdec->empty) {
        mlx_array pre0 = dc_slice_axis(pre, 2, 0, 1, s);
        mlx_array pre_s = dc_squeeze_axis(pre0, 2, s);
        mlx_array xt2 = dc_hdec(pre_s, mlx_array_empty, length_t, m->w, tdec,
                                NULL, s);
        mlx_array_free(xt);
        mlx_array_free(pre0);
        mlx_array_free(pre_s);
        xt = xt2;
      } else {
        if (n_saved_t <= 0) {
          dc_die("tdecoder skip underflow");
        }
        mlx_array skip_t = saved_t[--n_saved_t];
        mlx_array xt2 = dc_hdec(xt, skip_t, length_t, m->w, tdec, NULL, s);
        mlx_array_free(xt);
        mlx_array_free(skip_t);
        xt = xt2;
      }
    }
    if (pre.ctx) {
      mlx_array_free(pre);
    }
  }
  if (n_saved || n_saved_t || n_lent) {
    dc_die("skip connections not fully consumed (%d %d %d)", n_saved, n_saved_t,
           n_lent);
  }

  int B = dc_dim(x, 0);
  int Fq = dc_dim(mag, 2);
  int T = dc_dim(mag, 3);
  int S = cfg->n_sources;
  int Ctot = dc_dim(x, 1);
  int r[5] = {B, S, Ctot / S, Fq, T};
  if (Ctot % S != 0) {
    dc_die("decoder channels %d not divisible by sources %d", Ctot, S);
  }
  mlx_array xr = dc_reshape(x, r, 5, s);
  mlx_array mean_e = dc_expand_dims(mean, 1, s);
  mlx_array std_e = dc_expand_dims(stde, 1, s);
  mlx_array scaled = dc_mul(xr, std_e, s);
  mlx_array xden = dc_add(scaled, mean_e, s);
  mlx_array_free(scaled);
  mlx_array zout = mask_cac(xden, s);
  mlx_array xi = dc_htdemucs_ispec(&m->stft, zout, training_length, s);

  int actual = dc_dim(xt, -1);
  int rt[4] = {B, S, dc_dim(xt, 1) / S, actual};
  mlx_array xtr = dc_reshape(xt, rt, 4, s);
  mlx_array meant_e = dc_expand_dims(meant, 1, s);
  mlx_array stdt_e = dc_expand_dims(stdte, 1, s);
  mlx_array xts = dc_mul(xtr, stdt_e, s);
  mlx_array xtd = dc_add(xts, meant_e, s);
  mlx_array_free(xts);
  mlx_array xtrim = dc_center_trim(xi, dc_dim(xtd, -1), s);
  mlx_array y = dc_add(xtd, xtrim, s);
  mlx_array y2 = dc_slice_last(y, 0, training_length, s);
  if (length_pre_pad) {
    mlx_array y3 = dc_slice_last(y2, 0, length_pre_pad, s);
    mlx_array_free(y2);
    y2 = y3;
  }

  mlx_array_free(mix);
  mlx_array_free(z);
  mlx_array_free(mag);
  mlx_array_free(mean);
  mlx_array_free(stdv);
  mlx_array_free(stde);
  mlx_array_free(x);
  mlx_array_free(meant);
  mlx_array_free(stdt);
  mlx_array_free(stdte);
  mlx_array_free(xt);
  mlx_array_free(xr);
  mlx_array_free(mean_e);
  mlx_array_free(std_e);
  mlx_array_free(xden);
  mlx_array_free(zout);
  mlx_array_free(xi);
  mlx_array_free(xtr);
  mlx_array_free(meant_e);
  mlx_array_free(stdt_e);
  mlx_array_free(xtd);
  mlx_array_free(xtrim);
  mlx_array_free(y);
  return y2;
}
