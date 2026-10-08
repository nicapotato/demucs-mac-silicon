#ifndef DEMUCS_C_METAL_KERNELS_H
#define DEMUCS_C_METAL_KERNELS_H

#include "util.h"

mlx_array dc_gelu(mlx_array x, mlx_stream s);
mlx_array dc_glu_axis1(mlx_array x, mlx_stream s);
mlx_array dc_groupnorm(mlx_array x, mlx_array weight, mlx_array bias,
                       int num_groups, float eps, mlx_stream s);
mlx_array dc_fused_groupnorm_gelu(mlx_array x, mlx_array weight, mlx_array bias,
                                  int num_groups, float eps, mlx_stream s);
mlx_array dc_fused_groupnorm_glu(mlx_array x, mlx_array weight, mlx_array bias,
                                 int num_groups, float eps, mlx_stream s);

#endif
