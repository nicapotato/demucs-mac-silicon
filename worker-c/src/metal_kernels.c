#include "metal_kernels.h"

mlx_array dc_gelu(mlx_array x, mlx_stream s) {
  /* 0.5 * x * (1 + erf(x / sqrt(2))) */
  mlx_array s2 = mlx_array_new_float(0.7071067811865475f);
  mlx_array xs = dc_mul(x, s2, s);
  mlx_array e = dc_erf(xs, s);
  mlx_array one = mlx_array_new_float(1.0f);
  mlx_array ep = dc_add(e, one, s);
  mlx_array half = mlx_array_new_float(0.5f);
  mlx_array xh = dc_mul(x, half, s);
  mlx_array y = dc_mul(xh, ep, s);
  mlx_array_free(s2);
  mlx_array_free(xs);
  mlx_array_free(e);
  mlx_array_free(one);
  mlx_array_free(ep);
  mlx_array_free(half);
  mlx_array_free(xh);
  return y;
}

mlx_array dc_glu_axis1(mlx_array x, mlx_stream s) {
  mlx_array a = dc_split_half(x, 1, 0, s);
  mlx_array b = dc_split_half(x, 1, 1, s);
  mlx_array sb = dc_sigmoid(b, s);
  mlx_array y = dc_mul(a, sb, s);
  mlx_array_free(a);
  mlx_array_free(b);
  mlx_array_free(sb);
  return y;
}

mlx_array dc_groupnorm(mlx_array x, mlx_array weight, mlx_array bias,
                       int num_groups, float eps, mlx_stream s) {
  int B = dc_dim(x, 0);
  int C = dc_dim(x, 1);
  if (C % num_groups != 0) {
    dc_die("channels %d not divisible by groups %d", C, num_groups);
  }
  int cpg = C / num_groups;
  int n = dc_ndim(x);
  int rshape[DC_MAX_DIM];
  rshape[0] = B;
  rshape[1] = num_groups;
  rshape[2] = cpg;
  for (int i = 2; i < n; i++) {
    rshape[i + 1] = dc_dim(x, i);
  }
  int rn = n + 1;
  mlx_array xr = dc_reshape(x, rshape, rn, s);
  int axes[DC_MAX_DIM];
  int naxes = 0;
  for (int i = 2; i < rn; i++) {
    axes[naxes++] = i;
  }
  mlx_array mean = dc_mean_axes(xr, axes, naxes, true, s);
  mlx_array var = dc_var_axes(xr, axes, naxes, true, s);
  mlx_array centered = dc_sub(xr, mean, s);
  mlx_array var_eps = dc_add_scalar(var, eps, s);
  mlx_array inv = dc_rsqrt(var_eps, s);
  mlx_array normed = dc_mul(centered, inv, s);
  int orig[DC_MAX_DIM];
  for (int i = 0; i < n; i++) {
    orig[i] = dc_dim(x, i);
  }
  mlx_array xout = dc_reshape(normed, orig, n, s);
  int wshape[DC_MAX_DIM];
  wshape[0] = 1;
  wshape[1] = C;
  for (int i = 2; i < n; i++) {
    wshape[i] = 1;
  }
  mlx_array w = dc_reshape(weight, wshape, n, s);
  mlx_array b = dc_reshape(bias, wshape, n, s);
  mlx_array scaled = dc_mul(xout, w, s);
  mlx_array y = dc_add(scaled, b, s);
  mlx_array_free(xr);
  mlx_array_free(mean);
  mlx_array_free(var);
  mlx_array_free(centered);
  mlx_array_free(var_eps);
  mlx_array_free(inv);
  mlx_array_free(normed);
  mlx_array_free(xout);
  mlx_array_free(w);
  mlx_array_free(b);
  mlx_array_free(scaled);
  return y;
}

mlx_array dc_fused_groupnorm_gelu(mlx_array x, mlx_array weight, mlx_array bias,
                                  int num_groups, float eps, mlx_stream s) {
  /* Custom Metal GN was wrong vs MLX ops on batched DConv shapes (B*Fr > 1). */
  mlx_array n = dc_groupnorm(x, weight, bias, num_groups, eps, s);
  mlx_array y = dc_gelu(n, s);
  mlx_array_free(n);
  return y;
}

mlx_array dc_fused_groupnorm_glu(mlx_array x, mlx_array weight, mlx_array bias,
                                 int num_groups, float eps, mlx_stream s) {
  mlx_array n = dc_groupnorm(x, weight, bias, num_groups, eps, s);
  mlx_array y = dc_glu_axis1(n, s);
  mlx_array_free(n);
  return y;
}
