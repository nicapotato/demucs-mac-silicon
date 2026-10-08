#include "util.h"

#include <math.h>

_Noreturn void dc_die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("demucs-c: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  fflush(stderr);
  abort();
}

void dc_on_mlx_error(const char *msg, void *data) {
  (void)data;
  dc_die("mlx error: %s", msg ? msg : "(null)");
}

void dc_eval(mlx_array a) {
  mlx_vector_array v = mlx_vector_array_new_value(a);
  DC_CHECK(mlx_eval(v));
  mlx_vector_array_free(v);
}

void dc_eval2(mlx_array a, mlx_array b) {
  mlx_array items[2] = {a, b};
  mlx_vector_array v = mlx_vector_array_new_data(items, 2);
  DC_CHECK(mlx_eval(v));
  mlx_vector_array_free(v);
}

void dc_sync(mlx_stream s) { DC_CHECK(mlx_synchronize(s)); }

int dc_ndim(mlx_array a) { return (int)mlx_array_ndim(a); }

int dc_dim(mlx_array a, int axis) {
  int n = dc_ndim(a);
  if (axis < 0) {
    axis += n;
  }
  if (axis < 0 || axis >= n) {
    dc_die("axis %d out of range for ndim %d", axis, n);
  }
  return mlx_array_dim(a, axis);
}

size_t dc_size(mlx_array a) { return mlx_array_size(a); }

mlx_array dc_copy(mlx_array a) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_array_set(&r, a));
  return r;
}

void dc_replace(mlx_array *dst, mlx_array src) {
  mlx_array_free(*dst);
  *dst = src;
}

static mlx_array unary(int (*fn)(mlx_array *, const mlx_array, const mlx_stream),
                       mlx_array a, mlx_stream s, const char *name) {
  mlx_array r = mlx_array_new();
  if (fn(&r, a, s)) {
    dc_die("%s failed", name);
  }
  return r;
}

static mlx_array binary(int (*fn)(mlx_array *, const mlx_array, const mlx_array,
                                  const mlx_stream),
                        mlx_array a, mlx_array b, mlx_stream s, const char *name) {
  mlx_array r = mlx_array_new();
  if (fn(&r, a, b, s)) {
    dc_die("%s failed", name);
  }
  return r;
}

mlx_array dc_add(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_add, a, b, s, "mlx_add");
}
mlx_array dc_sub(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_subtract, a, b, s, "mlx_subtract");
}
mlx_array dc_mul(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_multiply, a, b, s, "mlx_multiply");
}
mlx_array dc_div(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_divide, a, b, s, "mlx_divide");
}

mlx_array dc_add_scalar(mlx_array a, float v, mlx_stream s) {
  mlx_array sc = mlx_array_new_float(v);
  mlx_array r = dc_add(a, sc, s);
  mlx_array_free(sc);
  return r;
}

mlx_array dc_mul_scalar(mlx_array a, float v, mlx_stream s) {
  mlx_array sc = mlx_array_new_float(v);
  mlx_array r = dc_mul(a, sc, s);
  mlx_array_free(sc);
  return r;
}

mlx_array dc_reshape(mlx_array a, const int *shape, int ndim, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_reshape(&r, a, shape, (size_t)ndim, s));
  return r;
}

mlx_array dc_transpose(mlx_array a, const int *axes, int naxes, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_transpose_axes(&r, a, axes, (size_t)naxes, s));
  return r;
}

mlx_array dc_contiguous(mlx_array a, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_contiguous(&r, a, false, s));
  return r;
}

mlx_array dc_astype(mlx_array a, mlx_dtype dt, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_astype(&r, a, dt, s));
  return r;
}

mlx_array dc_expand_dims(mlx_array a, int axis, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_expand_dims(&r, a, axis, s));
  return r;
}

mlx_array dc_broadcast_to(mlx_array a, const int *shape, int ndim, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_broadcast_to(&r, a, shape, (size_t)ndim, s));
  return r;
}

mlx_array dc_slice(mlx_array a, const int *start, const int *stop, mlx_stream s) {
  int n = dc_ndim(a);
  int strides[DC_MAX_DIM];
  for (int i = 0; i < n; i++) {
    strides[i] = 1;
  }
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_slice(&r, a, start, (size_t)n, stop, (size_t)n, strides,
                     (size_t)n, s));
  return r;
}

mlx_array dc_slice_last(mlx_array a, int start, int stop, mlx_stream s) {
  return dc_slice_axis(a, -1, start, stop, s);
}

mlx_array dc_slice_axis(mlx_array a, int axis, int start, int stop,
                        mlx_stream s) {
  int n = dc_ndim(a);
  if (axis < 0) {
    axis += n;
  }
  int starts[DC_MAX_DIM];
  int stops[DC_MAX_DIM];
  const int *shape = mlx_array_shape(a);
  for (int i = 0; i < n; i++) {
    starts[i] = 0;
    stops[i] = shape[i];
  }
  starts[axis] = start;
  stops[axis] = stop;
  return dc_slice(a, starts, stops, s);
}

mlx_array dc_slice_update(mlx_array src, mlx_array update, const int *start,
                          const int *stop, mlx_stream s) {
  int n = dc_ndim(src);
  int strides[DC_MAX_DIM];
  for (int i = 0; i < n; i++) {
    strides[i] = 1;
  }
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_slice_update(&r, src, update, start, (size_t)n, stop, (size_t)n,
                            strides, (size_t)n, s));
  return r;
}

mlx_array dc_slice_update_add(mlx_array src, mlx_array update, const int *start,
                              const int *stop, mlx_stream s) {
  int n = dc_ndim(src);
  int strides[DC_MAX_DIM];
  for (int i = 0; i < n; i++) {
    strides[i] = 1;
  }
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_slice_update_add(&r, src, update, start, (size_t)n, stop,
                                (size_t)n, strides, (size_t)n, s));
  return r;
}

mlx_array dc_slice_add(mlx_array src, mlx_array update, const int *start,
                       const int *stop, mlx_stream s) {
  mlx_array sl = dc_slice(src, start, stop, s);
  mlx_array sum = dc_add(sl, update, s);
  mlx_array r = dc_slice_update(src, sum, start, stop, s);
  mlx_array_free(sl);
  mlx_array_free(sum);
  return r;
}

mlx_array dc_pad_axis(mlx_array a, int axis, int left, int right,
                      const char *mode, mlx_stream s) {
  if (left == 0 && right == 0) {
    return dc_copy(a);
  }
  int n = dc_ndim(a);
  if (axis < 0) {
    axis += n;
  }
  int axes[1] = {axis};
  int low[1] = {left};
  int high[1] = {right};
  mlx_array zero = mlx_array_new_float(0.0f);
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_pad(&r, a, axes, 1, low, 1, high, 1, zero, mode, s));
  mlx_array_free(zero);
  return r;
}

mlx_array dc_pad_last(mlx_array a, int left, int right, const char *mode,
                      mlx_stream s) {
  return dc_pad_axis(a, -1, left, right, mode, s);
}

mlx_array dc_flip_axis(mlx_array a, int axis, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_flip_axis(&r, a, axis, s));
  return r;
}

mlx_array dc_pad1d(mlx_array a, int left, int right, const char *mode,
                   mlx_stream s) {
  if (strcmp(mode, "reflect") != 0) {
    return dc_pad_last(a, left, right, mode, s);
  }
  int length = dc_dim(a, -1);
  int padding_left = left;
  int padding_right = right;
  mlx_array x = dc_copy(a);
  int max_pad = padding_left > padding_right ? padding_left : padding_right;
  if (length <= max_pad) {
    int extra_pad = max_pad - length + 1;
    int extra_right = padding_right < extra_pad ? padding_right : extra_pad;
    int extra_left = extra_pad - extra_right;
    padding_left -= extra_left;
    padding_right -= extra_right;
    if (extra_left || extra_right) {
      mlx_array padded = dc_pad_last(x, extra_left, extra_right, "constant", s);
      mlx_array_free(x);
      x = padded;
      length = dc_dim(x, -1);
    }
  }
  if (padding_left) {
    mlx_array left_slice = dc_slice_last(x, 1, padding_left + 1, s);
    mlx_array left_ref = dc_flip_axis(left_slice, -1, s);
    mlx_array parts[2] = {left_ref, x};
    mlx_array cat = dc_concat(parts, 2, -1, s);
    mlx_array_free(left_slice);
    mlx_array_free(left_ref);
    mlx_array_free(x);
    x = cat;
  }
  if (padding_right) {
    int L = dc_dim(x, -1);
    mlx_array right_slice = dc_slice_last(x, L - padding_right - 1, L - 1, s);
    mlx_array right_ref = dc_flip_axis(right_slice, -1, s);
    mlx_array parts[2] = {x, right_ref};
    mlx_array cat = dc_concat(parts, 2, -1, s);
    mlx_array_free(right_slice);
    mlx_array_free(right_ref);
    mlx_array_free(x);
    x = cat;
  }
  return x;
}

mlx_array dc_concat(mlx_array *items, int n, int axis, mlx_stream s) {
  mlx_vector_array vec = mlx_vector_array_new_data(items, (size_t)n);
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_concatenate_axis(&r, vec, axis, s));
  mlx_vector_array_free(vec);
  return r;
}

mlx_array dc_stack(mlx_array *items, int n, int axis, mlx_stream s) {
  mlx_vector_array vec = mlx_vector_array_new_data(items, (size_t)n);
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_stack_axis(&r, vec, axis, s));
  mlx_vector_array_free(vec);
  return r;
}

mlx_array dc_split_half(mlx_array a, int axis, int which, mlx_stream s) {
  mlx_vector_array parts = mlx_vector_array_new();
  DC_CHECK(mlx_split(&parts, a, 2, axis, s));
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_vector_array_get(&r, parts, (size_t)which));
  mlx_vector_array_free(parts);
  return r;
}

mlx_array dc_zeros(const int *shape, int ndim, mlx_dtype dt, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_zeros(&r, shape, (size_t)ndim, dt, s));
  return r;
}

mlx_array dc_full_f32(const int *shape, int ndim, float v, mlx_stream s) {
  mlx_array val = mlx_array_new_float(v);
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_full(&r, shape, (size_t)ndim, val, MLX_FLOAT32, s));
  mlx_array_free(val);
  return r;
}

mlx_array dc_arange(int n, mlx_dtype dt, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_arange(&r, 0.0, (double)n, 1.0, dt, s));
  return r;
}

mlx_array dc_linspace_int(int start, int stop_exclusive, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_arange(&r, (double)start, (double)stop_exclusive, 1.0, MLX_INT32,
                      s));
  return r;
}

mlx_array dc_mean_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                       mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_mean_axes(&r, a, axes, (size_t)naxes, keepdims, s));
  return r;
}

mlx_array dc_std_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                      mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_std_axes(&r, a, axes, (size_t)naxes, keepdims, 0, s));
  return r;
}

mlx_array dc_var_axes(mlx_array a, const int *axes, int naxes, bool keepdims,
                      mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_var_axes(&r, a, axes, (size_t)naxes, keepdims, 0, s));
  return r;
}

mlx_array dc_rsqrt(mlx_array a, mlx_stream s) {
  return unary(mlx_rsqrt, a, s, "mlx_rsqrt");
}
mlx_array dc_sqrt(mlx_array a, mlx_stream s) {
  return unary(mlx_sqrt, a, s, "mlx_sqrt");
}
mlx_array dc_sigmoid(mlx_array a, mlx_stream s) {
  return unary(mlx_sigmoid, a, s, "mlx_sigmoid");
}
mlx_array dc_sin(mlx_array a, mlx_stream s) {
  return unary(mlx_sin, a, s, "mlx_sin");
}
mlx_array dc_cos(mlx_array a, mlx_stream s) {
  return unary(mlx_cos, a, s, "mlx_cos");
}
mlx_array dc_erf(mlx_array a, mlx_stream s) {
  return unary(mlx_erf, a, s, "mlx_erf");
}
mlx_array dc_abs(mlx_array a, mlx_stream s) {
  return unary(mlx_abs, a, s, "mlx_abs");
}
mlx_array dc_real(mlx_array a, mlx_stream s) {
  return unary(mlx_real, a, s, "mlx_real");
}
mlx_array dc_imag(mlx_array a, mlx_stream s) {
  return unary(mlx_imag, a, s, "mlx_imag");
}

mlx_array dc_maximum(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_maximum, a, b, s, "mlx_maximum");
}

mlx_array dc_complex(mlx_array real, mlx_array imag, mlx_stream s) {
  mlx_array j = mlx_array_new_complex(0.0f, 1.0f);
  mlx_array ji = dc_mul(imag, j, s);
  mlx_array r = dc_add(real, ji, s);
  mlx_array_free(j);
  mlx_array_free(ji);
  return r;
}

mlx_array dc_matmul(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_matmul, a, b, s, "mlx_matmul");
}

mlx_array dc_softmax_axis(mlx_array a, int axis, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_softmax_axis(&r, a, axis, true, s));
  return r;
}

mlx_array dc_take_axis(mlx_array a, mlx_array indices, int axis, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_take_axis(&r, a, indices, axis, s));
  return r;
}

mlx_array dc_power(mlx_array a, mlx_array b, mlx_stream s) {
  return binary(mlx_power, a, b, s, "mlx_power");
}

mlx_array dc_center_trim(mlx_array a, int ref_len, mlx_stream s) {
  int length = dc_dim(a, -1);
  int delta = length - ref_len;
  if (delta < 0) {
    dc_die("center_trim: tensor shorter than reference (%d < %d)", length,
           ref_len);
  }
  if (delta == 0) {
    return dc_copy(a);
  }
  int start = delta / 2;
  int end = length - (delta - start);
  return dc_slice_last(a, start, end, s);
}

mlx_array dc_unflatten(mlx_array a, int axis, const int *shape, int nshape,
                       mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_unflatten(&r, a, axis, shape, (size_t)nshape, s));
  return r;
}

mlx_array dc_flatten(mlx_array a, int start, int end, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_flatten(&r, a, start, end, s));
  return r;
}

mlx_array dc_squeeze_axis(mlx_array a, int axis, mlx_stream s) {
  mlx_array r = mlx_array_new();
  DC_CHECK(mlx_squeeze_axis(&r, a, axis, s));
  return r;
}

bool dc_has_weight(mlx_map_string_to_array map, const char *key) {
  mlx_array tmp = mlx_array_new();
  int rc = mlx_map_string_to_array_get(&tmp, map, key);
  mlx_array_free(tmp);
  return rc == 0;
}

mlx_array dc_require_weight(mlx_map_string_to_array map, const char *key) {
  mlx_array a = mlx_array_new();
  if (mlx_map_string_to_array_get(&a, map, key) != 0) {
    dc_die("missing weight '%s'", key);
  }
  return a;
}

float dc_item_f32(mlx_array a) {
  dc_eval(a);
  float v = 0.0f;
  DC_CHECK(mlx_array_item_float32(&v, a));
  return v;
}

void dc_print_shape(const char *name, mlx_array a) {
  int n = dc_ndim(a);
  fprintf(stderr, "%s rank=%d shape=[", name, n);
  for (int i = 0; i < n; i++) {
    fprintf(stderr, "%s%d", i ? "," : "", dc_dim(a, i));
  }
  fprintf(stderr, "]\n");
}
