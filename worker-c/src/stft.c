#include "stft.h"

#include <math.h>
#include <stdint.h>

DcStft dc_stft_new(int n_fft, int hop, mlx_stream s) {
  DcStft st;
  st.n_fft = n_fft;
  st.hop = hop;
  float *w = (float *)malloc((size_t)n_fft * sizeof(float));
  if (!w) {
    dc_die("oom hann window");
  }
  for (int i = 0; i < n_fft; i++) {
    w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n_fft);
  }
  int wshape[1] = {n_fft};
  st.window = mlx_array_new_data(w, wshape, 1, MLX_FLOAT32);
  mlx_array wsq = dc_mul(st.window, st.window, s);
  st.window_sq = wsq;
  free(w);
  return st;
}

void dc_stft_free(DcStft *st) {
  if (!st) {
    return;
  }
  mlx_array_free(st->window);
  mlx_array_free(st->window_sq);
}

static mlx_array frame_signal(mlx_array x, int n_fft, int hop, mlx_stream s) {
  /* x: [B, T] contiguous */
  mlx_array xc = dc_contiguous(x, s);
  int B = dc_dim(xc, 0);
  int T = dc_dim(xc, 1);
  if (T < n_fft) {
    mlx_array padded = dc_pad_last(xc, 0, n_fft - T, "constant", s);
    mlx_array_free(xc);
    xc = padded;
    T = dc_dim(xc, 1);
  }
  int n_frames = (T - n_fft) / hop + 1;
  int shape[3] = {B, n_frames, n_fft};
  int64_t strides[3] = {T, hop, 1};
  mlx_array frames = mlx_array_new();
  DC_CHECK(mlx_as_strided(&frames, xc, shape, 3, strides, 3, 0, s));
  mlx_array_free(xc);
  return frames;
}

mlx_array dc_stft_bfn(DcStft *st, mlx_array x, mlx_stream s) {
  /* x: [B, T] -> [B, F, N] complex */
  int pad = st->n_fft / 2;
  mlx_array xp = dc_pad1d(x, pad, pad, "reflect", s);
  mlx_array frames = frame_signal(xp, st->n_fft, st->hop, s);
  mlx_array win = dc_mul(frames, st->window, s);
  mlx_array spec = mlx_array_new();
  DC_CHECK(mlx_fft_rfft(&spec, win, st->n_fft, -1, MLX_FFT_NORM_BACKWARD, s));
  int axes[3] = {0, 2, 1};
  mlx_array bfn = dc_transpose(spec, axes, 3, s);
  mlx_array_free(xp);
  mlx_array_free(frames);
  mlx_array_free(win);
  mlx_array_free(spec);
  return bfn;
}

static mlx_array ola_frames(mlx_array frames, int hop, mlx_stream s) {
  /* frames: [B, n_frames, n_fft] -> [B, hop*(n_frames-1)+n_fft] */
  int B = dc_dim(frames, 0);
  int n_frames = dc_dim(frames, 1);
  int n_fft = dc_dim(frames, 2);
  if (n_fft % hop != 0) {
    int out_len = hop * (n_frames - 1) + n_fft;
    int oshape[2] = {B, out_len};
    mlx_array acc = dc_zeros(oshape, 2, MLX_FLOAT32, s);
    for (int i = 0; i < n_frames; i++) {
      int start_t = i * hop;
      mlx_array frame = dc_slice_axis(frames, 1, i, i + 1, s);
      int fshape[2] = {B, n_fft};
      mlx_array flat = dc_reshape(frame, fshape, 2, s);
      int starts[2] = {0, start_t};
      int stops[2] = {B, start_t + n_fft};
      mlx_array acc2 = dc_slice_add(acc, flat, starts, stops, s);
      mlx_array_free(acc);
      mlx_array_free(frame);
      mlx_array_free(flat);
      acc = acc2;
    }
    return acc;
  }
  int ratio = n_fft / hop;
  int r[4] = {B, n_frames, ratio, hop};
  mlx_array u = dc_reshape(frames, r, 4, s);
  int out_hops = n_frames + ratio - 1;
  int ash[3] = {B, out_hops, hop};
  mlx_array acc = dc_zeros(ash, 3, MLX_FLOAT32, s);
  for (int k = 0; k < ratio; k++) {
    mlx_array sl = dc_slice_axis(u, 2, k, k + 1, s);
    int sh3[3] = {B, n_frames, hop};
    mlx_array flat = dc_reshape(sl, sh3, 3, s);
    int starts[3] = {0, k, 0};
    int stops[3] = {B, k + n_frames, hop};
    mlx_array acc2 = dc_slice_add(acc, flat, starts, stops, s);
    mlx_array_free(acc);
    mlx_array_free(sl);
    mlx_array_free(flat);
    acc = acc2;
  }
  int osh[2] = {B, out_hops * hop};
  mlx_array y = dc_reshape(acc, osh, 2, s);
  mlx_array_free(u);
  mlx_array_free(acc);
  return y;
}

mlx_array dc_istft_bfn(DcStft *st, mlx_array z, int length, mlx_stream s) {
  /* z: [B, F, N] complex -> [B, T] */
  int axes[3] = {0, 2, 1};
  mlx_array bnf = dc_transpose(z, axes, 3, s);
  mlx_array time_frames = mlx_array_new();
  DC_CHECK(mlx_fft_irfft(&time_frames, bnf, st->n_fft, -1, MLX_FFT_NORM_BACKWARD,
                         s));
  mlx_array win_frames = dc_mul(time_frames, st->window, s);
  int B = dc_dim(win_frames, 0);
  int n_frames = dc_dim(win_frames, 1);
  mlx_array acc = ola_frames(win_frames, st->hop, s);
  int wshape[3] = {1, 1, st->n_fft};
  mlx_array wr = dc_reshape(st->window_sq, wshape, 3, s);
  int bshape[3] = {B, n_frames, st->n_fft};
  mlx_array wexp = dc_broadcast_to(wr, bshape, 3, s);
  mlx_array wacc = ola_frames(wexp, st->hop, s);
  mlx_array y = dc_div(acc, wacc, s);
  int pad = st->n_fft / 2;
  int out_len = dc_dim(y, 1);
  int stop = pad + length;
  if (stop > out_len) {
    stop = out_len;
  }
  mlx_array trimmed = dc_slice_last(y, pad, stop, s);
  int T = dc_dim(trimmed, 1);
  if (T < length) {
    mlx_array padded = dc_pad_last(trimmed, 0, length - T, "constant", s);
    mlx_array_free(trimmed);
    trimmed = padded;
  } else if (T > length) {
    mlx_array cut = dc_slice_last(trimmed, 0, length, s);
    mlx_array_free(trimmed);
    trimmed = cut;
  }
  mlx_array_free(bnf);
  mlx_array_free(time_frames);
  mlx_array_free(win_frames);
  mlx_array_free(acc);
  mlx_array_free(wr);
  mlx_array_free(wexp);
  mlx_array_free(wacc);
  mlx_array_free(y);
  return trimmed;
}

mlx_array dc_htdemucs_spec(DcStft *st, mlx_array mix, mlx_stream s) {
  /* mix [B, C, T] -> z [B, C, F, N] */
  int B = dc_dim(mix, 0);
  int C = dc_dim(mix, 1);
  int T = dc_dim(mix, 2);
  int hl = st->hop;
  int le = (T + hl - 1) / hl;
  int pad = (hl / 2) * 3;
  int extra = pad + le * hl - T;
  mlx_array xp = dc_pad1d(mix, pad, extra, "reflect", s);
  int Tp = dc_dim(xp, 2);
  int rshape[2] = {B * C, Tp};
  mlx_array x2 = dc_reshape(xp, rshape, 2, s);
  mlx_array spec2 = dc_stft_bfn(st, x2, s);
  int F = dc_dim(spec2, 1);
  int N = dc_dim(spec2, 2);
  int s4[4] = {B, C, F, N};
  mlx_array z = dc_reshape(spec2, s4, 4, s);
  mlx_array z1 = dc_slice_axis(z, 2, 0, F - 1, s);
  int F2 = dc_dim(z1, 2);
  (void)F2;
  mlx_array z2 = dc_slice_axis(z1, 3, 2, 2 + le, s);
  mlx_array_free(xp);
  mlx_array_free(x2);
  mlx_array_free(spec2);
  mlx_array_free(z);
  mlx_array_free(z1);
  return z2;
}

mlx_array dc_htdemucs_ispec(DcStft *st, mlx_array z, int length, mlx_stream s) {
  int hl = st->hop;
  int n = dc_ndim(z);
  mlx_array zp = mlx_array_new();
  if (n == 5) {
    mlx_array_free(zp);
    zp = dc_pad_axis(z, 3, 0, 1, "constant", s);
    mlx_array zp2 = dc_pad_axis(zp, 4, 2, 2, "constant", s);
    mlx_array_free(zp);
    zp = zp2;
  } else if (n == 4) {
    mlx_array_free(zp);
    zp = dc_pad_axis(z, 2, 0, 1, "constant", s);
    mlx_array zp2 = dc_pad_axis(zp, 3, 2, 2, "constant", s);
    mlx_array_free(zp);
    zp = zp2;
  } else {
    dc_die("ispec expected 4D or 5D, got %d", n);
  }
  int pad = (hl / 2) * 3;
  int le_len = hl * ((length + hl - 1) / hl) + 2 * pad;
  mlx_array wav;
  if (dc_ndim(zp) == 5) {
    int B = dc_dim(zp, 0);
    int S = dc_dim(zp, 1);
    int C = dc_dim(zp, 2);
    int F = dc_dim(zp, 3);
    int N = dc_dim(zp, 4);
    int r2[3] = {B * S * C, F, N};
    mlx_array z2 = dc_reshape(zp, r2, 3, s);
    mlx_array w2 = dc_istft_bfn(st, z2, le_len, s);
    int T = dc_dim(w2, 1);
    int r4[4] = {B, S, C, T};
    wav = dc_reshape(w2, r4, 4, s);
    mlx_array_free(z2);
    mlx_array_free(w2);
  } else {
    int B = dc_dim(zp, 0);
    int C = dc_dim(zp, 1);
    int F = dc_dim(zp, 2);
    int N = dc_dim(zp, 3);
    int r2[3] = {B * C, F, N};
    mlx_array z2 = dc_reshape(zp, r2, 3, s);
    mlx_array w2 = dc_istft_bfn(st, z2, le_len, s);
    int T = dc_dim(w2, 1);
    int r3[3] = {B, C, T};
    wav = dc_reshape(w2, r3, 3, s);
    mlx_array_free(z2);
    mlx_array_free(w2);
  }
  mlx_array trimmed = dc_slice_last(wav, pad, pad + length, s);
  mlx_array_free(zp);
  mlx_array_free(wav);
  return trimmed;
}
