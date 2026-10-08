#ifndef DEMUCS_C_TRANSFORMER_H
#define DEMUCS_C_TRANSFORMER_H

#include "layers.h"

void dc_crosstransformer(mlx_array *x_io, mlx_array *xt_io, const DcWeights *w,
                         const DcConfig *cfg, mlx_stream s);

#endif
