# demucs-mac-silicon

Apple Silicon MLX worker for Demucs stem separation. Frozen `demucs_mlx_worker` is
consumed by [demucs-ui-app](https://github.com/nicapotato/demucs-ui-app).

This is the MLX inference package (fork of [demucs-mlx](https://github.com/ssmall256/demucs-mlx)).
The standalone GUI lives in **demucs-ui-app**. Windows CUDA lives in
[demucs-windows](https://github.com/nicapotato/demucs-windows). Version: [`project.conf`](project.conf).

## Features

- **~73x realtime** on Apple Silicon — 2.6x faster than Demucs with PyTorch MPS
- **Bit-exact parity** with upstream Demucs stems (within floating-point tolerance)
- Custom fused Metal kernels (GroupNorm+GELU, GroupNorm+GLU, OLA)
- No PyTorch required at inference time
- Automatic resampling — input files at any sample rate are resampled to the model rate
- Audio I/O via [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- STFT/iSTFT via [mlx-spectro](https://github.com/ssmall256/mlx-spectro)

## Requirements

- Python >= 3.10
- macOS with Apple Silicon

## Install / CLI

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

On first run, demucs-mlx loads cached MLX weights if available. To bootstrap a missing
model, install the conversion extra:

```bash
uv sync --extra convert
```

Options:

```
-n, --name          Model name (default: htdemucs)
-o, --out           Output directory (default: separated)
--shifts            Number of random shifts (default: 1)
--seed              Optional RNG seed for reproducible shifts (default: none)
--overlap           Overlap ratio (default: 0.25)
-b, --batch-size    Batch size (default: 8)
--write-workers     Concurrent writer threads (default: 1)
--list-models       List available models
-v, --verbose       Verbose logging
```

## Frozen worker

```bash
make worker
# → dist/worker/demucs_mlx_worker/demucs_mlx_worker
```

CI (`make ci`) runs tests and uploads `demucs-mlx-worker-mac-arm64`. Product zips and
itch.io shipping are in demucs-ui-app.

## Python usage

```python
from demucs_mlx import Separator

separator = Separator()
origin, stems = separator.separate_audio_file("song.wav")

for name, audio in stems.items():
    print(f"{name}: {audio.shape}")
```

## Models

| Model | Sources | Description |
|-------|---------|-------------|
| `htdemucs` | 4 | Hybrid Transformer Demucs (default) |
| `htdemucs_ft` | 4 | Fine-tuned HTDemucs |
| `htdemucs_6s` | 6 | 6-source (adds piano, guitar) |
| `hdemucs_mmi` | 4 | Hybrid Demucs MMI |
| `mdx` | 4 | Music Demixing model |
| `mdx_extra` | 4 | MDX with extra training |

## Documentation

- API reference: `docs/api.md`
- Development workflow: `docs/development.md`
- Platform notes: `docs/platform.md`

## License

MIT. Based on [Demucs](https://github.com/adefossez/demucs) by Meta Research. See `LICENSE` for details.
