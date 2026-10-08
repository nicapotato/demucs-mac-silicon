#include "audio.h"

#include <lame/lame.h>
#include <math.h>
#include <stdint.h>

static uint16_t u16le(const unsigned char *p) {
  return (uint16_t)(p[0] | (p[1] << 8));
}
static uint32_t u32le(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static void put_u16le(unsigned char *p, uint16_t v) {
  p[0] = (unsigned char)(v & 0xff);
  p[1] = (unsigned char)((v >> 8) & 0xff);
}
static void put_u32le(unsigned char *p, uint32_t v) {
  p[0] = (unsigned char)(v & 0xff);
  p[1] = (unsigned char)((v >> 8) & 0xff);
  p[2] = (unsigned char)((v >> 16) & 0xff);
  p[3] = (unsigned char)((v >> 24) & 0xff);
}

void dc_wav_free(DcWav *wav) {
  if (!wav) {
    return;
  }
  free(wav->data);
  wav->data = NULL;
  wav->channels = 0;
  wav->frames = 0;
}

DcWav dc_wav_read(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    dc_die("cannot open wav %s", path);
  }
  unsigned char hdr[12];
  if (fread(hdr, 1, 12, fp) != 12) {
    dc_die("short wav header %s", path);
  }
  if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
    dc_die("not a RIFF/WAVE file: %s", path);
  }
  int sample_rate = 0;
  int channels = 0;
  int bits = 0;
  int data_size = -1;
  long data_off = -1;
  while (1) {
    unsigned char ch[8];
    if (fread(ch, 1, 8, fp) != 8) {
      break;
    }
    uint32_t sz = u32le(ch + 4);
    if (memcmp(ch, "fmt ", 4) == 0) {
      unsigned char fmt[40];
      if (sz > sizeof(fmt) || fread(fmt, 1, sz, fp) != sz) {
        dc_die("bad fmt chunk in %s", path);
      }
      uint16_t format = u16le(fmt);
      channels = (int)u16le(fmt + 2);
      sample_rate = (int)u32le(fmt + 4);
      bits = (int)u16le(fmt + 14);
      if (format != 1 && format != 3) {
        dc_die("unsupported wav format %u in %s", format, path);
      }
      if (sz % 2) {
        fseek(fp, 1, SEEK_CUR);
      }
    } else if (memcmp(ch, "data", 4) == 0) {
      data_size = (int)sz;
      data_off = ftell(fp);
      fseek(fp, sz + (sz % 2), SEEK_CUR);
    } else {
      fseek(fp, sz + (sz % 2), SEEK_CUR);
    }
  }
  if (data_off < 0 || channels <= 0 || sample_rate <= 0) {
    dc_die("wav missing fmt/data: %s", path);
  }
  if (channels > 2) {
    dc_die("wav has %d channels, expected 1 or 2", channels);
  }
  int bpf = channels * (bits / 8);
  if (bpf <= 0) {
    dc_die("bad wav bytes-per-frame");
  }
  int frames = data_size / bpf;
  float *planar = (float *)malloc((size_t)channels * (size_t)frames * sizeof(float));
  if (!planar) {
    dc_die("oom wav buffer");
  }
  fseek(fp, data_off, SEEK_SET);
  if (bits == 16) {
    int16_t *tmp = (int16_t *)malloc((size_t)data_size);
    if (!tmp || fread(tmp, 1, (size_t)data_size, fp) != (size_t)data_size) {
      dc_die("failed reading wav pcm %s", path);
    }
    for (int t = 0; t < frames; t++) {
      for (int c = 0; c < channels; c++) {
        planar[c * frames + t] = (float)tmp[t * channels + c] / 32768.0f;
      }
    }
    free(tmp);
  } else if (bits == 32) {
    float *tmp = (float *)malloc((size_t)data_size);
    if (!tmp || fread(tmp, 1, (size_t)data_size, fp) != (size_t)data_size) {
      dc_die("failed reading wav f32 %s", path);
    }
    for (int t = 0; t < frames; t++) {
      for (int c = 0; c < channels; c++) {
        planar[c * frames + t] = tmp[t * channels + c];
      }
    }
    free(tmp);
  } else {
    dc_die("unsupported wav bit depth %d in %s", bits, path);
  }
  fclose(fp);
  if (channels == 1) {
    float *stereo = (float *)malloc((size_t)2 * (size_t)frames * sizeof(float));
    if (!stereo) {
      dc_die("oom stereo upsample");
    }
    memcpy(stereo, planar, (size_t)frames * sizeof(float));
    memcpy(stereo + frames, planar, (size_t)frames * sizeof(float));
    free(planar);
    planar = stereo;
    channels = 2;
  }
  DcWav wav = {planar, channels, frames, sample_rate};
  return wav;
}

void dc_wav_resample(DcWav *wav, int target_sr) {
  if (!wav || !wav->data) {
    dc_die("dc_wav_resample: empty wav");
  }
  if (target_sr <= 0) {
    dc_die("dc_wav_resample: bad target rate %d", target_sr);
  }
  if (wav->sample_rate == target_sr) {
    return;
  }
  if (wav->sample_rate <= 0 || wav->frames <= 0 || wav->channels <= 0) {
    dc_die("dc_wav_resample: bad source wav");
  }
  int64_t new_frames_i =
      ((int64_t)wav->frames * (int64_t)target_sr) / (int64_t)wav->sample_rate;
  if (new_frames_i <= 0 || new_frames_i > 1 << 28) {
    dc_die("dc_wav_resample: bad output length %lld", (long long)new_frames_i);
  }
  int new_frames = (int)new_frames_i;
  float *out =
      (float *)malloc((size_t)wav->channels * (size_t)new_frames * sizeof(float));
  if (!out) {
    dc_die("oom resample buffer");
  }
  double ratio = (double)wav->sample_rate / (double)target_sr;
  for (int c = 0; c < wav->channels; c++) {
    const float *src = wav->data + (size_t)c * (size_t)wav->frames;
    float *dst = out + (size_t)c * (size_t)new_frames;
    for (int t = 0; t < new_frames; t++) {
      double src_t = (double)t * ratio;
      int i0 = (int)src_t;
      int i1 = i0 + 1;
      if (i0 < 0) {
        i0 = 0;
      }
      if (i0 >= wav->frames) {
        i0 = wav->frames - 1;
      }
      if (i1 >= wav->frames) {
        i1 = wav->frames - 1;
      }
      float frac = (float)(src_t - (double)i0);
      dst[t] = src[i0] + frac * (src[i1] - src[i0]);
    }
  }
  free(wav->data);
  wav->data = out;
  wav->frames = new_frames;
  wav->sample_rate = target_sr;
}

static float dc_prevent_clip_scale(const float *planar, int channels, int frames) {
  float peak = 0.0f;
  for (int c = 0; c < channels; c++) {
    const float *ch = planar + (size_t)c * (size_t)frames;
    for (int t = 0; t < frames; t++) {
      float a = fabsf(ch[t]);
      if (a > peak) {
        peak = a;
      }
    }
  }
  float scale = 1.01f * peak;
  if (scale < 1.0f) {
    scale = 1.0f;
  }
  return scale;
}

static int16_t dc_s16_from_float(float v, float scale) {
  v = v / scale;
  if (v > 1.0f) {
    v = 1.0f;
  }
  if (v < -1.0f) {
    v = -1.0f;
  }
  int iv = (int)lrintf(v * 32767.0f);
  if (iv > 32767) {
    iv = 32767;
  }
  if (iv < -32768) {
    iv = -32768;
  }
  return (int16_t)iv;
}

void dc_wav_write_pcm16(const char *path, const float *planar, int channels,
                        int frames, int sample_rate) {
  FILE *fp = fopen(path, "wb");
  if (!fp) {
    dc_die("cannot write %s", path);
  }
  uint32_t data_bytes = (uint32_t)channels * (uint32_t)frames * 2u;
  unsigned char hdr[44];
  memset(hdr, 0, sizeof(hdr));
  memcpy(hdr, "RIFF", 4);
  put_u32le(hdr + 4, 36 + data_bytes);
  memcpy(hdr + 8, "WAVEfmt ", 8);
  put_u32le(hdr + 16, 16);
  put_u16le(hdr + 20, 1);
  put_u16le(hdr + 22, (uint16_t)channels);
  put_u32le(hdr + 24, (uint32_t)sample_rate);
  put_u32le(hdr + 28, (uint32_t)sample_rate * (uint32_t)channels * 2u);
  put_u16le(hdr + 32, (uint16_t)(channels * 2));
  put_u16le(hdr + 34, 16);
  memcpy(hdr + 36, "data", 4);
  put_u32le(hdr + 40, data_bytes);
  if (fwrite(hdr, 1, 44, fp) != 44) {
    dc_die("wav header write failed %s", path);
  }
  /* Match Python save_audio(clip='rescale'): wav / max(1.01*peak, 1). */
  float scale = dc_prevent_clip_scale(planar, channels, frames);
  const int tile_frames = 65536;
  int16_t *tile = (int16_t *)malloc((size_t)tile_frames * (size_t)channels *
                                    sizeof(int16_t));
  if (!tile) {
    dc_die("oom wav encode tile");
  }
  for (int t0 = 0; t0 < frames; t0 += tile_frames) {
    int n = frames - t0;
    if (n > tile_frames) {
      n = tile_frames;
    }
    for (int t = 0; t < n; t++) {
      for (int c = 0; c < channels; c++) {
        tile[t * channels + c] = dc_s16_from_float(
            planar[(size_t)c * (size_t)frames + (size_t)(t0 + t)], scale);
      }
    }
    size_t nbytes = (size_t)n * (size_t)channels * 2u;
    if (fwrite(tile, 1, nbytes, fp) != nbytes) {
      dc_die("wav data write failed %s", path);
    }
  }
  free(tile);
  fclose(fp);
}

void dc_mp3_write(const char *path, const float *planar, int channels,
                  int frames, int sample_rate, int bitrate_kbps) {
  if (!path || !planar) {
    dc_die("dc_mp3_write: empty args");
  }
  if (channels != 1 && channels != 2) {
    dc_die("dc_mp3_write: channels=%d, expected 1 or 2", channels);
  }
  if (frames <= 0 || sample_rate <= 0) {
    dc_die("dc_mp3_write: bad frames/rate");
  }
  if (bitrate_kbps <= 0) {
    dc_die("dc_mp3_write: bitrate must be > 0");
  }

  lame_t gfp = lame_init();
  if (!gfp) {
    dc_die("lame_init failed for %s", path);
  }
  lame_set_in_samplerate(gfp, sample_rate);
  lame_set_num_channels(gfp, channels);
  lame_set_brate(gfp, bitrate_kbps);
  lame_set_quality(gfp, 2); /* match Python lameenc.set_quality(2) */
  lame_set_mode(gfp, channels == 1 ? MONO : STEREO);
  if (lame_init_params(gfp) < 0) {
    lame_close(gfp);
    dc_die("lame_init_params failed for %s", path);
  }

  FILE *fp = fopen(path, "wb");
  if (!fp) {
    lame_close(gfp);
    dc_die("cannot write %s", path);
  }

  float scale = dc_prevent_clip_scale(planar, channels, frames);
  const int tile_frames = 65536;
  int16_t *tile = (int16_t *)malloc((size_t)tile_frames * (size_t)channels *
                                    sizeof(int16_t));
  /* LAME: mp3buf_size >= 1.25*samples + 7200 */
  int mp3_cap = (int)((double)tile_frames * 1.25) + 7200;
  unsigned char *mp3buf = (unsigned char *)malloc((size_t)mp3_cap);
  if (!tile || !mp3buf) {
    free(tile);
    free(mp3buf);
    fclose(fp);
    lame_close(gfp);
    dc_die("oom mp3 encode buffers");
  }

  for (int t0 = 0; t0 < frames; t0 += tile_frames) {
    int n = frames - t0;
    if (n > tile_frames) {
      n = tile_frames;
    }
    for (int t = 0; t < n; t++) {
      for (int c = 0; c < channels; c++) {
        tile[t * channels + c] = dc_s16_from_float(
            planar[(size_t)c * (size_t)frames + (size_t)(t0 + t)], scale);
      }
    }
    int nb = 0;
    if (channels == 2) {
      nb = lame_encode_buffer_interleaved(gfp, tile, n, mp3buf, mp3_cap);
    } else {
      nb = lame_encode_buffer(gfp, tile, NULL, n, mp3buf, mp3_cap);
    }
    if (nb < 0) {
      free(tile);
      free(mp3buf);
      fclose(fp);
      lame_close(gfp);
      remove(path);
      dc_die("lame encode failed (%d) for %s", nb, path);
    }
    if (nb > 0 && fwrite(mp3buf, 1, (size_t)nb, fp) != (size_t)nb) {
      free(tile);
      free(mp3buf);
      fclose(fp);
      lame_close(gfp);
      remove(path);
      dc_die("mp3 write failed %s", path);
    }
  }

  int nb = lame_encode_flush(gfp, mp3buf, mp3_cap);
  if (nb < 0) {
    free(tile);
    free(mp3buf);
    fclose(fp);
    lame_close(gfp);
    remove(path);
    dc_die("lame flush failed (%d) for %s", nb, path);
  }
  if (nb > 0 && fwrite(mp3buf, 1, (size_t)nb, fp) != (size_t)nb) {
    free(tile);
    free(mp3buf);
    fclose(fp);
    lame_close(gfp);
    remove(path);
    dc_die("mp3 flush write failed %s", path);
  }
  free(tile);
  free(mp3buf);
  fclose(fp);
  lame_close(gfp);
}

mlx_array dc_wav_to_mx(const DcWav *wav, mlx_stream s) {
  (void)s;
  int shape[3] = {1, wav->channels, wav->frames};
  mlx_array a = mlx_array_new_data(wav->data, shape, 3, MLX_FLOAT32);
  return a;
}

void dc_mx_to_wav_planar(mlx_array a, float **out, int *channels, int *frames) {
  int n = dc_ndim(a);
  if (n == 3) {
    /* (1, C, T) or (C, 1, T) — expect (B, C, T) with B=1 */
    if (dc_dim(a, 0) != 1) {
      dc_die("expected batch=1 audio, got %d", dc_dim(a, 0));
    }
    *channels = dc_dim(a, 1);
    *frames = dc_dim(a, 2);
  } else if (n == 2) {
    *channels = dc_dim(a, 0);
    *frames = dc_dim(a, 1);
  } else {
    dc_die("unexpected audio rank %d", n);
  }
  dc_eval(a);
  size_t nval = dc_size(a);
  float *buf = (float *)malloc(nval * sizeof(float));
  if (!buf) {
    dc_die("oom host audio");
  }
  const float *src = mlx_array_data_float32(a);
  memcpy(buf, src, nval * sizeof(float));
  *out = buf;
}
