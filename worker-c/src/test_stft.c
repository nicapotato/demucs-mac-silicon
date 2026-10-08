#include "stft.h"

#include <math.h>

int main(void) {
  mlx_set_error_handler(dc_on_mlx_error, NULL, NULL);
  mlx_stream s = mlx_default_gpu_stream_new();
  DcStft st = dc_stft_new(4096, 1024, s);
  int shape[2] = {2, 44100};
  mlx_array x = mlx_array_new();
  DC_CHECK(mlx_random_normal(&x, shape, 2, MLX_FLOAT32, 0, 1, mlx_array_empty, s));
  mlx_array z = dc_stft_bfn(&st, x, s);
  mlx_array y = dc_istft_bfn(&st, z, 44100, s);
  dc_eval(y);
  int T = dc_dim(y, 1);
  if (T != 44100) {
    dc_die("istft length %d != 44100", T);
  }
  mlx_array d = dc_sub(x, y, s);
  mlx_array ad = dc_abs(d, s);
  mlx_array m = mlx_array_new();
  DC_CHECK(mlx_mean(&m, ad, false, s));
  float mae = dc_item_f32(m);
  printf("test_stft ok mae=%g shape_z=[%d,%d,%d]\n", mae, dc_dim(z, 0),
         dc_dim(z, 1), dc_dim(z, 2));
  if (mae > 0.05f) {
    dc_die("stft roundtrip mae too high");
  }
  mlx_array_free(x);
  mlx_array_free(z);
  mlx_array_free(y);
  mlx_array_free(d);
  mlx_array_free(ad);
  mlx_array_free(m);
  dc_stft_free(&st);
  mlx_stream_free(s);
  return 0;
}
