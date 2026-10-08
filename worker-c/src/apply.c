#include "apply.h"

#include <stdint.h>

/* CPython 3 MT19937 — matches random.Random(seed).randint. */
#define DC_MT_N 624
#define DC_MT_M 397
#define DC_MT_MATRIX_A 0x9908b0dfUL
#define DC_MT_UPPER 0x80000000UL
#define DC_MT_LOWER 0x7fffffffUL

typedef struct {
  uint32_t mt[DC_MT_N];
  int mti;
} DcPyRand;

static void mt_init_genrand(DcPyRand *r, uint32_t s) {
  r->mt[0] = s;
  for (int i = 1; i < DC_MT_N; i++) {
    r->mt[i] = (1812433253UL * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i);
  }
  r->mti = DC_MT_N;
}

static void mt_init_by_array(DcPyRand *r, const uint32_t *init_key, int key_length) {
  mt_init_genrand(r, 19650218UL);
  int i = 1;
  int j = 0;
  int k = DC_MT_N > key_length ? DC_MT_N : key_length;
  for (; k; k--) {
    r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525UL)) +
               init_key[j] + (uint32_t)j;
    i++;
    j++;
    if (i >= DC_MT_N) {
      r->mt[0] = r->mt[DC_MT_N - 1];
      i = 1;
    }
    if (j >= key_length) {
      j = 0;
    }
  }
  for (k = DC_MT_N - 1; k; k--) {
    r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941UL)) -
               (uint32_t)i;
    i++;
    if (i >= DC_MT_N) {
      r->mt[0] = r->mt[DC_MT_N - 1];
      i = 1;
    }
  }
  r->mt[0] = 0x80000000UL;
}

static void py_seed(DcPyRand *r, unsigned long n) {
  uint32_t key[8];
  int nkey = 0;
  if (n == 0) {
    key[0] = 0;
    nkey = 1;
  } else {
    unsigned long x = n;
    while (x && nkey < 8) {
      key[nkey++] = (uint32_t)(x & 0xffffffffUL);
      if (sizeof(unsigned long) <= 4) {
        break;
      }
      x >>= 32;
    }
  }
  mt_init_by_array(r, key, nkey);
}

static uint32_t mt_genrand_int32(DcPyRand *r) {
  uint32_t y;
  static const uint32_t mag01[2] = {0x0UL, DC_MT_MATRIX_A};
  if (r->mti >= DC_MT_N) {
    int kk;
    for (kk = 0; kk < DC_MT_N - DC_MT_M; kk++) {
      y = (r->mt[kk] & DC_MT_UPPER) | (r->mt[kk + 1] & DC_MT_LOWER);
      r->mt[kk] = r->mt[kk + DC_MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < DC_MT_N - 1; kk++) {
      y = (r->mt[kk] & DC_MT_UPPER) | (r->mt[kk + 1] & DC_MT_LOWER);
      r->mt[kk] = r->mt[kk + (DC_MT_M - DC_MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (r->mt[DC_MT_N - 1] & DC_MT_UPPER) | (r->mt[0] & DC_MT_LOWER);
    r->mt[DC_MT_N - 1] = r->mt[DC_MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    r->mti = 0;
  }
  y = r->mt[r->mti++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  return y;
}

static int bit_length_u32(uint32_t n) {
  int k = 0;
  while (n) {
    n >>= 1;
    k++;
  }
  return k;
}

static int py_randint(DcPyRand *r, int lo, int hi) {
  /* random.Random.randint(lo, hi) inclusive via _randbelow(width). */
  if (hi < lo) {
    dc_die("randint range");
  }
  uint32_t n = (uint32_t)(hi - lo + 1);
  int k = bit_length_u32(n);
  uint32_t v;
  do {
    v = k == 0 ? 0 : (mt_genrand_int32(r) >> (32 - k));
  } while (v >= n);
  return lo + (int)v;
}

static void note_progress(const DcApplyOpts *opts, int *done, int total) {
  (*done)++;
  if (opts->on_progress) {
    opts->on_progress(*done, total, opts->progress_user);
  }
}

static mlx_array ola_weight(int segment_length, float transition_power,
                            mlx_stream s) {
  int half = segment_length / 2;
  mlx_array up = dc_linspace_int(1, half + 1, s);
  mlx_array down = mlx_array_new();
  DC_CHECK(mlx_arange(&down, (double)(segment_length - half), 0.0, -1.0,
                      MLX_INT32, s));
  mlx_array uf = dc_astype(up, MLX_FLOAT32, s);
  mlx_array df = dc_astype(down, MLX_FLOAT32, s);
  mlx_array parts[2] = {uf, df};
  mlx_array w = dc_concat(parts, 2, 0, s);
  mlx_array wmax = mlx_array_new();
  DC_CHECK(mlx_max(&wmax, w, false, s));
  mlx_array wn = dc_div(w, wmax, s);
  mlx_array p = mlx_array_new_float(transition_power);
  mlx_array wp = dc_power(wn, p, s);
  mlx_array_free(up);
  mlx_array_free(down);
  mlx_array_free(uf);
  mlx_array_free(df);
  mlx_array_free(w);
  mlx_array_free(wmax);
  mlx_array_free(wn);
  mlx_array_free(p);
  return wp;
}

static mlx_array padded_chunk(mlx_array mix, int offset, int chunk_len,
                              int target_len, mlx_stream s) {
  int total = dc_dim(mix, -1);
  int delta = target_len - chunk_len;
  if (delta < 0) {
    dc_die("padded_chunk target < chunk");
  }
  int start = offset - delta / 2;
  int end = start + target_len;
  int correct_start = start < 0 ? 0 : start;
  int correct_end = end > total ? total : end;
  int pad_left = correct_start - start;
  int pad_right = end - correct_end;
  mlx_array sl = dc_slice_last(mix, correct_start, correct_end, s);
  if (pad_left || pad_right) {
    mlx_array p = dc_pad_last(sl, pad_left, pad_right, "constant", s);
    mlx_array_free(sl);
    sl = p;
  }
  return sl;
}

mlx_array dc_apply_model(DcModel *m, mlx_array mix, const DcApplyOpts *opts) {
  mlx_stream s = m->stream;
  int batch = dc_dim(mix, 0);
  int channels = dc_dim(mix, 1);
  int length = dc_dim(mix, 2);
  int sources = m->w->cfg.n_sources;

  if (opts->shifts > 0) {
    int max_shift = (int)(0.5f * (float)m->w->cfg.samplerate);
    mlx_array padded = padded_chunk(mix, 0, length, length + 2 * max_shift, s);
    mlx_array acc = mlx_array_new();
    int have = 0;
    DcPyRand pr;
    py_seed(&pr, opts->use_seed ? (unsigned long)opts->seed : 1UL);
    for (int i = 0; i < opts->shifts; i++) {
      int offset = py_randint(&pr, 0, max_shift);
      DcApplyOpts inner = *opts;
      inner.shifts = 0;
      inner.on_progress = (i == 0) ? opts->on_progress : NULL;
      mlx_array shifted =
          padded_chunk(padded, offset, length + max_shift - offset,
                       length + max_shift - offset, s);
      mlx_array out = dc_apply_model(m, shifted, &inner);
      mlx_array trimmed = dc_slice_last(out, max_shift - offset,
                                        dc_dim(out, -1), s);
      if (!have) {
        mlx_array_free(acc);
        acc = trimmed;
        have = 1;
      } else {
        mlx_array a2 = dc_add(acc, trimmed, s);
        mlx_array_free(acc);
        mlx_array_free(trimmed);
        acc = a2;
      }
      mlx_array_free(shifted);
      mlx_array_free(out);
    }
    mlx_array n = mlx_array_new_float((float)opts->shifts);
    mlx_array avg = dc_div(acc, n, s);
    mlx_array_free(padded);
    mlx_array_free(acc);
    mlx_array_free(n);
    dc_eval(avg);
    return avg;
  }

  if (!opts->split) {
    int valid = m->w->cfg.valid_length;
    mlx_array padded = padded_chunk(mix, 0, length, valid, s);
    mlx_array out = dc_model_forward(m, padded);
    mlx_array trim = dc_center_trim(out, length, s);
    mlx_array_free(padded);
    mlx_array_free(out);
    if (opts->on_progress) {
      opts->on_progress(1, 1, opts->progress_user);
    }
    return trim;
  }

  float segment = opts->segment > 0 ? opts->segment : m->w->cfg.segment;
  int segment_length = (int)((float)m->w->cfg.samplerate * segment);
  int stride = (int)((1.0f - opts->overlap) * (float)segment_length);
  if (stride <= 0) {
    dc_die("invalid overlap %f", opts->overlap);
  }
  int n_off = 0;
  for (int off = 0; off < length; off += stride) {
    n_off++;
  }
  mlx_array weight = ola_weight(segment_length, opts->transition_power, s);
  int oshape[4] = {batch, sources, channels, length};
  mlx_array out = dc_zeros(oshape, 4, MLX_FLOAT32, s);
  mlx_array sum_weight = dc_zeros((int[]){length}, 1, MLX_FLOAT32, s);
  int std_valid = m->w->cfg.valid_length;
  int segments_done = 0;

  mlx_array batch_in[32];
  int batch_idx[32];
  int nb = 0;
  int max_b = opts->batch_size;
  if (max_b > 32) {
    max_b = 32;
  }
  if (max_b < 1) {
    max_b = 1;
  }

  int offsets[4096];
  if (n_off > 4096) {
    dc_die("too many segments");
  }
  n_off = 0;
  for (int off = 0; off < length; off += stride) {
    offsets[n_off++] = off;
  }

  for (int i = 0; i <= n_off; i++) {
    int flush = (i == n_off) || (i < n_off && (length - offsets[i]) < segment_length);
    if (i < n_off && (length - offsets[i]) >= segment_length) {
      mlx_array ch =
          padded_chunk(mix, offsets[i], segment_length, std_valid, s);
      batch_in[nb] = ch;
      batch_idx[nb] = i;
      nb++;
      if (nb < max_b) {
        continue;
      }
      flush = 1;
    }
    if (flush && nb > 0) {
      mlx_array stacked = dc_stack(batch_in, nb, 0, s);
      int bseg = dc_dim(stacked, 0);
      int baudio = dc_dim(stacked, 1);
      int ch_st = dc_dim(stacked, 2);
      int T = dc_dim(stacked, 3);
      int r[3] = {bseg * baudio, ch_st, T};
      mlx_array flat = dc_reshape(stacked, r, 3, s);
      mlx_array yflat = dc_model_forward(m, flat);
      int So = dc_dim(yflat, 1);
      int Oc = dc_dim(yflat, 2);
      int Ot = dc_dim(yflat, 3);
      int r5[5] = {bseg, baudio, So, Oc, Ot};
      mlx_array yb = dc_reshape(yflat, r5, 5, s);
      for (int k = 0; k < nb; k++) {
        mlx_array chunk = dc_slice_axis(yb, 0, k, k + 1, s);
        int r4[4] = {baudio, So, Oc, Ot};
        mlx_array c4 = dc_reshape(chunk, r4, 4, s);
        mlx_array ct = dc_center_trim(c4, segment_length, s);
        int off = offsets[batch_idx[k]];
        int end = off + segment_length;
        int wshape[4] = {1, 1, 1, segment_length};
        mlx_array wr = dc_reshape(weight, wshape, 4, s);
        mlx_array upd = dc_mul(ct, wr, s);
        int starts[4] = {0, 0, 0, off};
        int stops[4] = {batch, sources, channels, end};
        mlx_array out2 = dc_slice_add(out, upd, starts, stops, s);
        int wstarts[1] = {off};
        int wstops[1] = {end};
        mlx_array sw2 = dc_slice_add(sum_weight, weight, wstarts, wstops, s);
        mlx_array_free(out);
        mlx_array_free(sum_weight);
        out = out2;
        sum_weight = sw2;
        mlx_array_free(chunk);
        mlx_array_free(c4);
        mlx_array_free(ct);
        mlx_array_free(wr);
        mlx_array_free(upd);
        note_progress(opts, &segments_done, n_off);
      }
      dc_eval2(out, sum_weight);
      mlx_array_free(stacked);
      mlx_array_free(flat);
      mlx_array_free(yflat);
      mlx_array_free(yb);
      for (int k = 0; k < nb; k++) {
        mlx_array_free(batch_in[k]);
      }
      nb = 0;
    }
    if (i < n_off && (length - offsets[i]) < segment_length) {
      int this_len = length - offsets[i];
      int valid = this_len;
      if (this_len < m->w->cfg.valid_length) {
        valid = m->w->cfg.valid_length;
      }
      mlx_array ch = padded_chunk(mix, offsets[i], this_len, valid, s);
      mlx_array y = dc_model_forward(m, ch);
      mlx_array ct = dc_center_trim(y, this_len, s);
      mlx_array wsl = dc_slice_last(weight, 0, this_len, s);
      int wshape[4] = {1, 1, 1, this_len};
      mlx_array wr = dc_reshape(wsl, wshape, 4, s);
      mlx_array upd = dc_mul(ct, wr, s);
      int off = offsets[i];
      int end = off + this_len;
      int starts[4] = {0, 0, 0, off};
      int stops[4] = {batch, sources, channels, end};
      mlx_array out2 = dc_slice_add(out, upd, starts, stops, s);
      int wstarts[1] = {off};
      int wstops[1] = {end};
      mlx_array sw2 = dc_slice_add(sum_weight, wsl, wstarts, wstops, s);
      mlx_array_free(out);
      mlx_array_free(sum_weight);
      out = out2;
      sum_weight = sw2;
      mlx_array_free(ch);
      mlx_array_free(y);
      mlx_array_free(ct);
      mlx_array_free(wsl);
      mlx_array_free(wr);
      mlx_array_free(upd);
      dc_eval2(out, sum_weight);
      note_progress(opts, &segments_done, n_off);
    }
  }

  mlx_array w4 = dc_reshape(sum_weight, (int[]){1, 1, 1, length}, 4, s);
  mlx_array y = dc_div(out, w4, s);
  dc_eval(y);
  mlx_array_free(weight);
  mlx_array_free(out);
  mlx_array_free(sum_weight);
  mlx_array_free(w4);
  (void)sources;
  return y;
}
