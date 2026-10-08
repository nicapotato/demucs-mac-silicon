#include "metal_kernels.h"

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

int main(void) {
  mlx_set_error_handler(dc_on_mlx_error, NULL, NULL);
  mlx_stream s = mlx_default_gpu_stream_new();
  int shape[3] = {2, 8, 32};
  mlx_array x = mlx_array_new();
  DC_CHECK(mlx_random_normal(&x, shape, 3, MLX_FLOAT32, 0, 1, mlx_array_empty, s));
  mlx_array w = dc_full_f32((int[]){8}, 1, 1.0f, s);
  mlx_array b = dc_zeros((int[]){8}, 1, MLX_FLOAT32, s);
  mlx_array ref = dc_groupnorm(x, w, b, 4, 1e-5f, s);
  mlx_array refg = dc_gelu(ref, s);
  mlx_array fused = dc_fused_groupnorm_gelu(x, w, b, 4, 1e-5f, s);
  float d = max_abs_diff(refg, fused, s);
  if (d > 1e-4f) {
    dc_die("fused_groupnorm_gelu mismatch %g", d);
  }
  mlx_array x2 = mlx_array_new();
  int shape2[3] = {2, 8, 32};
  DC_CHECK(mlx_random_normal(&x2, shape2, 3, MLX_FLOAT32, 0, 1, mlx_array_empty, s));
  mlx_array w2 = dc_full_f32((int[]){8}, 1, 1.0f, s);
  mlx_array b2 = dc_zeros((int[]){8}, 1, MLX_FLOAT32, s);
  mlx_array gn = dc_groupnorm(x2, w2, b2, 1, 1e-5f, s);
  mlx_array glu = dc_glu_axis1(gn, s);
  mlx_array fglu = dc_fused_groupnorm_glu(x2, w2, b2, 1, 1e-5f, s);
  float d2 = max_abs_diff(glu, fglu, s);
  if (d2 > 1e-4f) {
    dc_die("fused_groupnorm_glu mismatch %g", d2);
  }
  int gshape[3] = {32, 6, 344};
  mlx_array xg = mlx_array_new();
  DC_CHECK(mlx_random_normal(&xg, gshape, 3, MLX_FLOAT32, 0, 1, mlx_array_empty, s));
  mlx_array wg = dc_full_f32((int[]){6}, 1, 1.0f, s);
  mlx_array bg = dc_zeros((int[]){6}, 1, MLX_FLOAT32, s);
  mlx_array fg = dc_fused_groupnorm_gelu(xg, wg, bg, 1, 1e-5f, s);
  mlx_array rg = dc_groupnorm(xg, wg, bg, 1, 1e-5f, s);
  mlx_array rgg = dc_gelu(rg, s);
  float dconv_gn = max_abs_diff(fg, rgg, s);
  printf("test_kernels dconv-shape fused_vs_ops max_abs=%g\n", dconv_gn);
  if (dconv_gn > 1e-5f) {
    dc_die("dconv-shape groupnorm drifted %g", dconv_gn);
  }

  printf("test_kernels ok gelu_diff=%g glu_diff=%g\n", d, d2);
  mlx_array_free(x);
  mlx_array_free(w);
  mlx_array_free(b);
  mlx_array_free(ref);
  mlx_array_free(refg);
  mlx_array_free(fused);
  mlx_array_free(x2);
  mlx_array_free(w2);
  mlx_array_free(b2);
  mlx_array_free(gn);
  mlx_array_free(glu);
  mlx_array_free(fglu);
  mlx_array_free(xg);
  mlx_array_free(wg);
  mlx_array_free(bg);
  mlx_array_free(fg);
  mlx_array_free(rg);
  mlx_array_free(rgg);
  mlx_stream_free(s);
  return 0;
}
