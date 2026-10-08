#include "apply.h"
#include "audio.h"

#include "mlx/c/metal.h"

#include <libgen.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>

static int g_timing = 0;

static double dc_now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void dc_time_log(const char *stage, double t0) {
  if (!g_timing) {
    return;
  }
  fprintf(stderr, "dc_timing %s %.3fs\n", stage, dc_now_s() - t0);
}

static void dc_set_metallib_beside_exe(const char *argv0) {
  char exe[1024];
  uint32_t n = sizeof(exe);
  if (_NSGetExecutablePath(exe, &n) != 0) {
    snprintf(exe, sizeof(exe), "%s", argv0 ? argv0 : "");
  }
  char tmp[1024];
  snprintf(tmp, sizeof(tmp), "%s", exe);
  char *dir = dirname(tmp);
  char path[1200];
  snprintf(path, sizeof(path), "%s/mlx.metallib", dir);
  if (access(path, R_OK) == 0) {
    DC_CHECK(mlx_metal_set_metallib_path(path));
  }
}

static void mkdir_p(const char *path) {
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s", path);
  for (char *p = buf + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      mkdir(buf, 0755);
      *p = '/';
    }
  }
  mkdir(buf, 0755);
}

static void emit(int json, const char *fmt, ...) {
  if (!json) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  fputc('\n', stdout);
  fflush(stdout);
}

typedef struct {
  int json;
  const char *track;
} DcProgressCtx;

static void on_progress(int done, int total, void *user) {
  DcProgressCtx *ctx = (DcProgressCtx *)user;
  float pct = total > 0 ? (100.0f * (float)done / (float)total) : 0.0f;
  emit(ctx->json,
       "{\"event\":\"separating\",\"track\":\"%s\",\"done\":%d,\"total\":%d,"
       "\"pct\":%.2f}",
       ctx->track, done, total, pct);
}

static const char *basename_stem(const char *path) {
  const char *slash = strrchr(path, '/');
  const char *base = slash ? slash + 1 : path;
  return base;
}

static void strip_ext(char *name) {
  char *dot = strrchr(name, '.');
  if (dot) {
    *dot = 0;
  }
}

int main(int argc, char **argv) {
  mlx_set_error_handler(dc_on_mlx_error, NULL, NULL);
  dc_set_metallib_beside_exe(argv[0]);
  const char *timing_env = getenv("DC_TIMING");
  g_timing = timing_env && timing_env[0] && timing_env[0] != '0';
  double t_all = dc_now_s();

  const char *out_dir = "separated";
  const char *track_name = NULL;
  const char *model_name = "htdemucs_6s";
  const char *weights = NULL;
  const char *config = NULL;
  float overlap = 0.25f;
  int shifts = 1;
  int seed = 0;
  int use_seed = 0;
  int batch_size = 8;
  int split = 1;
  int progress_json = 0;
  int compile_fwd = 0;
  int write_mp3 = 0;
  int mp3_bitrate = 128;
  const char *tracks[64];
  int ntracks = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (strcmp(argv[i], "--track-name") == 0 && i + 1 < argc) {
      track_name = argv[++i];
    } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) &&
               i + 1 < argc) {
      model_name = argv[++i];
    } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
      weights = argv[++i];
    } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config = argv[++i];
    } else if (strcmp(argv[i], "--overlap") == 0 && i + 1 < argc) {
      overlap = strtof(argv[++i], NULL);
    } else if (strcmp(argv[i], "--shifts") == 0 && i + 1 < argc) {
      shifts = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = (int)strtol(argv[++i], NULL, 10);
      use_seed = 1;
    } else if ((strcmp(argv[i], "-b") == 0 ||
                strcmp(argv[i], "--batch-size") == 0) &&
               i + 1 < argc) {
      batch_size = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--no-split") == 0) {
      split = 0;
    } else if (strcmp(argv[i], "--progress-json") == 0) {
      progress_json = 1;
    } else if (strcmp(argv[i], "--compile") == 0) {
      compile_fwd = 1;
    } else if (strcmp(argv[i], "--list-models") == 0) {
      printf("htdemucs_6s\n");
      return 0;
    } else if (strcmp(argv[i], "--prefetch-tracks") == 0 && i + 1 < argc) {
      ++i; /* GUI always passes this; C worker has no prefetch. */
    } else if (strcmp(argv[i], "--mp3") == 0) {
      write_mp3 = 1;
    } else if (strcmp(argv[i], "--mp3-bitrate") == 0 && i + 1 < argc) {
      mp3_bitrate = (int)strtol(argv[++i], NULL, 10);
    } else if (argv[i][0] == '-') {
      dc_die("unknown flag %s", argv[i]);
    } else {
      if (ntracks >= 64) {
        dc_die("too many tracks");
      }
      tracks[ntracks++] = argv[i];
    }
  }
  if (ntracks == 0) {
    fprintf(stderr,
            "usage: demucs_mlx_worker [flags] track.wav\n"
            "  -n MODEL  --weights FILE.safetensors --config FILE.json\n"
            "  -o DIR --track-name NAME --overlap F --shifts N --seed N\n"
            "  -b N --no-split --progress-json --compile --mp3 --mp3-bitrate N\n"
            "  --prefetch-tracks N is accepted and ignored (GUI compatibility)\n");
    return 2;
  }
  if (track_name && ntracks != 1) {
    dc_die("--track-name requires exactly one track");
  }
  if (mp3_bitrate <= 0) {
    dc_die("--mp3-bitrate must be > 0");
  }

  char wbuf[1024];
  char cbuf[1024];
  const char *cache = getenv("DEMUCS_MLX_CACHE");
  if (!weights) {
    if (cache && cache[0]) {
      snprintf(wbuf, sizeof(wbuf), "%s/%s.safetensors", cache, model_name);
    } else {
      snprintf(wbuf, sizeof(wbuf), "models/%s.safetensors", model_name);
    }
    weights = wbuf;
  }
  if (!config) {
    if (cache && cache[0]) {
      snprintf(cbuf, sizeof(cbuf), "%s/%s.json", cache, model_name);
    } else {
      snprintf(cbuf, sizeof(cbuf), "models/%s.json", model_name);
    }
    config = cbuf;
  }

  emit(progress_json, "{\"event\":\"status\",\"stage\":\"loading_model\",\"model\":\"%s\"}",
       model_name);

  double t0 = dc_now_s();
  mlx_stream stream = mlx_default_gpu_stream_new();
  DcWeights wts = dc_weights_load(weights, config, stream);
  DcModel model;
  dc_model_init(&model, &wts);
  if (compile_fwd) {
    dc_model_enable_compile(&model);
  }
  dc_time_log("load_model", t0);

  emit(progress_json,
       "{\"event\":\"status\",\"stage\":\"model_ready\",\"model\":\"%s\","
       "\"device\":\"gpu\","
       "\"sources\":[\"drums\",\"bass\",\"other\",\"vocals\",\"guitar\",\"piano\"]}",
       model_name);

  DcApplyOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.shifts = shifts;
  opts.split = split;
  opts.overlap = overlap;
  opts.transition_power = 1.0f;
  opts.segment = wts.cfg.segment;
  opts.batch_size = batch_size;
  opts.seed = (unsigned)seed;
  opts.use_seed = use_seed;

  for (int t = 0; t < ntracks; t++) {
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", track_name ? track_name : basename_stem(tracks[t]));
    strip_ext(stem);
    char dest[1024];
    snprintf(dest, sizeof(dest), "%s/%s", out_dir, stem);
    mkdir_p(dest);
    emit(progress_json, "{\"event\":\"loading\",\"track\":\"%s\",\"out\":\"%s\"}",
         tracks[t], dest);

    DcProgressCtx pctx;
    pctx.json = progress_json;
    pctx.track = tracks[t];
    opts.on_progress = on_progress;
    opts.progress_user = &pctx;
    emit(progress_json,
         "{\"event\":\"separating\",\"track\":\"%s\",\"done\":0,\"total\":0,"
         "\"pct\":0.00}",
         tracks[t]);

    t0 = dc_now_s();
    DcWav wav = dc_wav_read(tracks[t]);
    if (wav.sample_rate != wts.cfg.samplerate) {
      fprintf(stderr, "demucs-c: resampling %d -> %d\n", wav.sample_rate,
              wts.cfg.samplerate);
      dc_wav_resample(&wav, wts.cfg.samplerate);
    }
    mlx_array mix = dc_wav_to_mx(&wav, stream);
    dc_time_log("wav_read", t0);
    t0 = dc_now_s();
    mlx_array stems = dc_apply_model(&model, mix, &opts);
    dc_eval(stems);
    dc_sync(stream);
    dc_time_log("apply", t0);

    int S = wts.cfg.n_sources;
    int C = dc_dim(stems, 2);
    int Tlen = dc_dim(stems, 3);
    t0 = dc_now_s();
    for (int si = 0; si < S; si++) {
      mlx_array sl = dc_slice_axis(stems, 1, si, si + 1, stream);
      int r[3] = {1, C, Tlen};
      mlx_array s2 = dc_reshape(sl, r, 3, stream);
      float *planar = NULL;
      int ch = 0, fr = 0;
      dc_mx_to_wav_planar(s2, &planar, &ch, &fr);
      char path[1200];
      const char *ext = write_mp3 ? "mp3" : "wav";
      snprintf(path, sizeof(path), "%s/%s.%s", dest, wts.cfg.sources[si], ext);
      if (write_mp3) {
        dc_mp3_write(path, planar, ch, fr, wts.cfg.samplerate, mp3_bitrate);
      } else {
        dc_wav_write_pcm16(path, planar, ch, fr, wts.cfg.samplerate);
      }
      emit(progress_json, "{\"event\":\"writing\",\"stem\":\"%s\",\"path\":\"%s\"}",
           wts.cfg.sources[si], path);
      free(planar);
      mlx_array_free(sl);
      mlx_array_free(s2);
    }
    dc_time_log(write_mp3 ? "mp3_write" : "wav_write", t0);
    emit(progress_json,
         "{\"event\":\"done\",\"track\":\"%s\",\"out\":\"%s\",\"sources\":"
         "[\"drums\",\"bass\",\"other\",\"vocals\",\"guitar\",\"piano\"],\"format\":\"%s\"}",
         tracks[t], dest, write_mp3 ? "mp3" : "wav");
    mlx_array_free(mix);
    mlx_array_free(stems);
    dc_wav_free(&wav);
  }

  dc_model_free(&model);
  dc_weights_free(&wts);
  mlx_stream_free(stream);
  dc_time_log("total", t_all);
  return 0;
}
