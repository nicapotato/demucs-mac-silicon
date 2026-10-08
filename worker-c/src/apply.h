#ifndef DEMUCS_C_APPLY_H
#define DEMUCS_C_APPLY_H

#include "htdemucs.h"

typedef void (*DcProgressFn)(int done, int total, void *user);

typedef struct {
  int shifts;
  int split;
  float overlap;
  float transition_power;
  float segment;
  int batch_size;
  unsigned seed;
  int use_seed;
  DcProgressFn on_progress;
  void *progress_user;
} DcApplyOpts;

mlx_array dc_apply_model(DcModel *m, mlx_array mix, const DcApplyOpts *opts);

#endif
