#ifndef DEMUCS_C_AUDIO_H
#define DEMUCS_C_AUDIO_H

#include "util.h"

typedef struct {
  float *data; /* interleaved is NOT used; planar (ch, frames) */
  int channels;
  int frames;
  int sample_rate;
} DcWav;

void dc_wav_free(DcWav *wav);
DcWav dc_wav_read(const char *path);
void dc_wav_resample(DcWav *wav, int target_sr);
void dc_wav_write_pcm16(const char *path, const float *planar, int channels,
                        int frames, int sample_rate);
void dc_mp3_write(const char *path, const float *planar, int channels,
                  int frames, int sample_rate, int bitrate_kbps);
mlx_array dc_wav_to_mx(const DcWav *wav, mlx_stream s);
void dc_mx_to_wav_planar(mlx_array a, float **out, int *channels, int *frames);

#endif
