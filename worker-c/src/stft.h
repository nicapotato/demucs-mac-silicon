#ifndef DEMUCS_C_STFT_H
#define DEMUCS_C_STFT_H

#include "config.h"

typedef struct {
  int n_fft;
  int hop;
  mlx_array window;
  mlx_array window_sq;
} DcStft;

DcStft dc_stft_new(int n_fft, int hop, mlx_stream s);
void dc_stft_free(DcStft *st);
mlx_array dc_stft_bfn(DcStft *st, mlx_array x, mlx_stream s);
mlx_array dc_istft_bfn(DcStft *st, mlx_array z, int length, mlx_stream s);
mlx_array dc_htdemucs_spec(DcStft *st, mlx_array mix, mlx_stream s);
mlx_array dc_htdemucs_ispec(DcStft *st, mlx_array z, int length, mlx_stream s);

#endif
