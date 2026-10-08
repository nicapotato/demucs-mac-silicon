#ifndef DEMUCS_C_UTIL_H
#define DEMUCS_C_UTIL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlx/c/mlx.h"

#define DC_MAX_DIM 8
#define DC_MAX_SOURCES 8

#define DC_CHECK(expr)                                                         \
  do {                                                                         \
    if (expr) {                                                                \
      dc_die("%s failed at %s:%d", #expr, __FILE__, __LINE__);                 \
    }                                                                          \
  } while (0)

_Noreturn void dc_die(const char *fmt, ...);
void dc_on_mlx_error(const char *msg, void *data);

void dc_eval(mlx_array a);
void dc_eval2(mlx_array a, mlx_array b);
void dc_sync(mlx_stream s);

int dc_ndim(mlx_array a);
int dc_dim(mlx_array a, int axis);
size_t dc_size(mlx_array a);

mlx_array dc_copy(mlx_array a);
void dc_replace(mlx_array *dst, mlx_array src);

mlx_array dc_add(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_sub(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_mul(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_div(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_add_scalar(mlx_array a, float v, mlx_stream s);
mlx_array dc_mul_scalar(mlx_array a, float v, mlx_stream s);

mlx_array dc_reshape(mlx_array a, const int *shape, int ndim, mlx_stream s);
mlx_array dc_transpose(mlx_array a, const int *axes, int naxes, mlx_stream s);
mlx_array dc_contiguous(mlx_array a, mlx_stream s);
mlx_array dc_astype(mlx_array a, mlx_dtype dt, mlx_stream s);
mlx_array dc_expand_dims(mlx_array a, int axis, mlx_stream s);
mlx_array dc_broadcast_to(mlx_array a, const int *shape, int ndim, mlx_stream s);

mlx_array dc_slice(mlx_array a, const int *start, const int *stop, mlx_stream s);
mlx_array dc_slice_last(mlx_array a, int start, int stop, mlx_stream s);
mlx_array dc_slice_axis(mlx_array a, int axis, int start, int stop, mlx_stream s);
mlx_array dc_slice_update(mlx_array src, mlx_array update, const int *start,
                          const int *stop, mlx_stream s);
mlx_array dc_slice_update_add(mlx_array src, mlx_array update, const int *start,
                              const int *stop, mlx_stream s);
/* Read-add-write; avoids mlx_slice_update_add corruption on strided slices. */
mlx_array dc_slice_add(mlx_array src, mlx_array update, const int *start,
                       const int *stop, mlx_stream s);

mlx_array dc_pad_last(mlx_array a, int left, int right, const char *mode,
                      mlx_stream s);
mlx_array dc_pad1d(mlx_array a, int left, int right, const char *mode,
                   mlx_stream s);
mlx_array dc_pad_axis(mlx_array a, int axis, int left, int right,
                      const char *mode, mlx_stream s);

mlx_array dc_concat(mlx_array *items, int n, int axis, mlx_stream s);
mlx_array dc_stack(mlx_array *items, int n, int axis, mlx_stream s);
mlx_array dc_split_half(mlx_array a, int axis, int which, mlx_stream s);

mlx_array dc_zeros(const int *shape, int ndim, mlx_dtype dt, mlx_stream s);
mlx_array dc_full_f32(const int *shape, int ndim, float v, mlx_stream s);
mlx_array dc_arange(int n, mlx_dtype dt, mlx_stream s);
mlx_array dc_linspace_int(int start, int stop_exclusive, mlx_stream s);

mlx_array dc_mean_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                       mlx_stream s);
mlx_array dc_std_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                      mlx_stream s);
mlx_array dc_var_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                      mlx_stream s);
mlx_array dc_rsqrt(mlx_array a, mlx_stream s);
mlx_array dc_sqrt(mlx_array a, mlx_stream s);
mlx_array dc_sigmoid(mlx_array a, mlx_stream s);
mlx_array dc_sin(mlx_array a, mlx_stream s);
mlx_array dc_cos(mlx_array a, mlx_stream s);
mlx_array dc_erf(mlx_array a, mlx_stream s);
mlx_array dc_maximum(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_abs(mlx_array a, mlx_stream s);
mlx_array dc_real(mlx_array a, mlx_stream s);
mlx_array dc_imag(mlx_array a, mlx_stream s);
mlx_array dc_complex(mlx_array real, mlx_array imag, mlx_stream s);
mlx_array dc_matmul(mlx_array a, mlx_array b, mlx_stream s);
mlx_array dc_softmax_axis(mlx_array a, int axis, mlx_stream s);
mlx_array dc_take_axis(mlx_array a, mlx_array indices, int axis, mlx_stream s);
mlx_array dc_flip_axis(mlx_array a, int axis, mlx_stream s);
mlx_array dc_power(mlx_array a, mlx_array b, mlx_stream s);

mlx_array dc_center_trim(mlx_array a, int ref_len, mlx_stream s);
mlx_array dc_unflatten(mlx_array a, int axis, const int *shape, int nshape,
                       mlx_stream s);
mlx_array dc_flatten(mlx_array a, int start, int end, mlx_stream s);
mlx_array dc_squeeze_axis(mlx_array a, int axis, mlx_stream s);

mlx_array dc_require_weight(mlx_map_string_to_array map, const char *key);
bool dc_has_weight(mlx_map_string_to_array map, const char *key);

float dc_item_f32(mlx_array a);
void dc_print_shape(const char *name, mlx_array a);

#endif
