#include "apply.h"

#include <math.h>

static float max_abs_diff(mlx_array a, mlx_array b, mlx_stream s) {
  mlx_array d = dc_sub(a, b, s);
  mlx_array ad = dc_abs(d, s);
  mlx_array m = mlx_array_new();
  DC_CHECK(mlx_max(&m, ad, false, s));
  float v = dc_item_f32(m);
  mlx_array_free(d);
  mlx_array_free(ad);
  mlx_array_free(m);
  return v;
}

static mlx_array triangle_weight(int segment_length, mlx_stream s) {
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
  mlx_array_free(up);
  mlx_array_free(down);
  mlx_array_free(uf);
  mlx_array_free(df);
  mlx_array_free(w);
  mlx_array_free(wmax);
  return wn;
}

typedef mlx_array (*DcAddFn)(mlx_array src, mlx_array update, const int *start,
                             const int *stop, mlx_stream s);

static mlx_array identity_ola(mlx_array mix, float overlap, int batch_size,
                              DcAddFn addfn, mlx_stream s) {
  int batch = dc_dim(mix, 0);
  int channels = dc_dim(mix, 1);
  int length = dc_dim(mix, 2);
  int sources = 1;
  int sr = 44100;
  float segment = 7.8f;
  int segment_length = (int)((float)sr * segment);
  int stride = (int)((1.0f - overlap) * (float)segment_length);
  if (stride <= 0) {
    dc_die("invalid overlap");
  }
  int offsets[4096];
  int n_off = 0;
  for (int off = 0; off < length; off += stride) {
    if (n_off >= 4096) {
      dc_die("too many segments");
    }
    offsets[n_off++] = off;
  }
  mlx_array weight = triangle_weight(segment_length, s);
  int oshape[4] = {batch, sources, channels, length};
  mlx_array out = dc_zeros(oshape, 4, MLX_FLOAT32, s);
  mlx_array sum_weight = dc_zeros((int[]){length}, 1, MLX_FLOAT32, s);
  int max_b = batch_size < 1 ? 1 : batch_size;
  mlx_array batch_in[32];
  int batch_idx[32];
  int nb = 0;

  for (int i = 0; i <= n_off; i++) {
    int flush = (i == n_off) || (i < n_off && (length - offsets[i]) < segment_length);
    if (i < n_off && (length - offsets[i]) >= segment_length) {
      mlx_array ch = dc_slice_last(mix, offsets[i], offsets[i] + segment_length, s);
      batch_in[nb] = ch;
      batch_idx[nb] = i;
      nb++;
      if (nb < max_b) {
        continue;
      }
      flush = 1;
    }
    if (flush && nb > 0) {
      for (int k = 0; k < nb; k++) {
        mlx_array ch = batch_in[k];
        int r4[4] = {batch, sources, channels, segment_length};
        mlx_array ct = dc_reshape(ch, r4, 4, s);
        int wshape[4] = {1, 1, 1, segment_length};
        mlx_array wr = dc_reshape(weight, wshape, 4, s);
        mlx_array upd = dc_mul(ct, wr, s);
        int off = offsets[batch_idx[k]];
        int end = off + segment_length;
        int starts[4] = {0, 0, 0, off};
        int stops[4] = {batch, sources, channels, end};
        mlx_array out2 = addfn(out, upd, starts, stops, s);
        int wstarts[1] = {off};
        int wstops[1] = {end};
        mlx_array sw2 = addfn(sum_weight, weight, wstarts, wstops, s);
        mlx_array_free(out);
        mlx_array_free(sum_weight);
        out = out2;
        sum_weight = sw2;
        mlx_array_free(ct);
        mlx_array_free(wr);
        mlx_array_free(upd);
      }
      dc_eval2(out, sum_weight);
      for (int k = 0; k < nb; k++) {
        mlx_array_free(batch_in[k]);
      }
      nb = 0;
    }
    if (i < n_off && (length - offsets[i]) < segment_length) {
      int this_len = length - offsets[i];
      mlx_array ch = dc_slice_last(mix, offsets[i], offsets[i] + this_len, s);
      int r4[4] = {batch, sources, channels, this_len};
      mlx_array ct = dc_reshape(ch, r4, 4, s);
      mlx_array wsl = dc_slice_last(weight, 0, this_len, s);
      int wshape[4] = {1, 1, 1, this_len};
      mlx_array wr = dc_reshape(wsl, wshape, 4, s);
      mlx_array upd = dc_mul(ct, wr, s);
      int off = offsets[i];
      int end = off + this_len;
      int starts[4] = {0, 0, 0, off};
      int stops[4] = {batch, sources, channels, end};
      mlx_array out2 = addfn(out, upd, starts, stops, s);
      int wstarts[1] = {off};
      int wstops[1] = {end};
      mlx_array sw2 = addfn(sum_weight, wsl, wstarts, wstops, s);
      mlx_array_free(out);
      mlx_array_free(sum_weight);
      out = out2;
      sum_weight = sw2;
      mlx_array_free(ch);
      mlx_array_free(ct);
      mlx_array_free(wsl);
      mlx_array_free(wr);
      mlx_array_free(upd);
      dc_eval2(out, sum_weight);
    }
  }

  mlx_array w4 = dc_reshape(sum_weight, (int[]){1, 1, 1, length}, 4, s);
  mlx_array y = dc_div(out, w4, s);
  dc_eval(y);
  mlx_array_free(weight);
  mlx_array_free(out);
  mlx_array_free(sum_weight);
  mlx_array_free(w4);
  return y;
}

static mlx_array make_sine(int frames, int channels, mlx_stream s) {
  float *buf = (float *)malloc((size_t)channels * (size_t)frames * sizeof(float));
  if (!buf) {
    dc_die("oom sine");
  }
  for (int t = 0; t < frames; t++) {
    float x = (float)t / 44100.0f;
    float v = 0.5f * sinf(2.0f * (float)M_PI * 220.0f * x) +
              0.3f * sinf(2.0f * (float)M_PI * 440.0f * x);
    buf[t] = v;
  }
  if (channels > 1) {
    for (int t = 0; t < frames; t++) {
      buf[frames + t] = buf[(t + 100) % frames];
    }
  }
  float peak = 0.0f;
  int n = channels * frames;
  for (int i = 0; i < n; i++) {
    float a = fabsf(buf[i]);
    if (a > peak) {
      peak = a;
    }
  }
  if (peak > 0.0f) {
    float g = 0.9f / peak;
    for (int i = 0; i < n; i++) {
      buf[i] *= g;
    }
  }
  int shape[3] = {1, channels, frames};
  mlx_array a = mlx_array_new_data(buf, shape, 3, MLX_FLOAT32);
  free(buf);
  (void)s;
  return a;
}

static void run_case(const char *label, int seconds, float overlap, int batch_size,
                     mlx_stream s) {
  int frames = 44100 * seconds;
  mlx_array mix = make_sine(frames, 2, s);
  mlx_array y_safe = identity_ola(mix, overlap, batch_size, dc_slice_add, s);
  mlx_array y_scat = identity_ola(mix, overlap, batch_size, dc_slice_update_add, s);
  mlx_array ref = dc_reshape(mix, (int[]){1, 1, 2, frames}, 4, s);
  float err_safe = max_abs_diff(y_safe, ref, s);
  float err_scat = max_abs_diff(y_scat, ref, s);
  printf("test_ola %s overlap=%.2f batch=%d safe=%g scatter_add=%g\n", label,
         overlap, batch_size, err_safe, err_scat);
  if (err_safe > 1e-4f) {
    dc_die("safe OLA reconstruction %g exceeds 1e-4", err_safe);
  }
  mlx_array_free(mix);
  mlx_array_free(y_safe);
  mlx_array_free(y_scat);
  mlx_array_free(ref);
}

int main(void) {
  mlx_set_error_handler(dc_on_mlx_error, NULL, NULL);
  mlx_stream s = mlx_default_gpu_stream_new();
  run_case("20s", 20, 0.25f, 1, s);
  run_case("20s", 20, 0.25f, 8, s);
  run_case("20s", 20, 0.0f, 1, s);
  run_case("20s", 20, 0.5f, 8, s);
  run_case("60s", 60, 0.25f, 8, s);
  printf("test_ola ok\n");
  mlx_stream_free(s);
  return 0;
}
