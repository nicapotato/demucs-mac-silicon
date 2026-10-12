# Platform notes

This project is uv-first on macOS Apple Silicon (MLX). The raylib GUI lives in
[demucs-ui-app](https://github.com/nicapotato/demucs-ui-app). Windows CUDA inference
lives in [demucs-windows](https://github.com/nicapotato/demucs-windows).

## macOS (Apple Silicon)

- Apple Silicon is supported via MLX (worker-py) and mlx-c (worker-c).
- Audio I/O is handled natively by mlx-audio-io (no FFmpeg required).
- worker-c `--mp3` needs Homebrew `lame` (and `mpg123` for lame >= 4): `make native-deps`.

Typical flow:

```bash
make sync
make models-fetch
uv run demucs-mlx-worker /path/to/audio.wav -o out
```

Freeze the standalone worker (consumed by demucs-ui-app):

```bash
make worker-pyinstaller    # -> dist/worker/demucs_mlx_worker/
```

Env vars:

| Variable | Purpose |
|----------|---------|
| `DEMUCS_MLX_PYTHON` | Python that has the project installed (freeze) |
| `DEMUCS_MLX_WORKER` | Path to a frozen worker binary (GUI) |
| `DEMUCS_MLX_CACHE_DIR` | Upstream weights cache (safetensors + config) |
| `DEMUCS_MLX_CACHE` | GUI's name: weights dir; `<dir>/mlx` is used if `<dir>` holds a C export |
| `DEMUCS_MLX_NO_DOWNLOAD` | `1`: never download weights |
| `DEMUCS_MLX_WORKER_DIST` | Override freeze output dir (default `dist/worker`) |

Weights are downloaded by upstream demucs-mlx on first use (SHA-256 checked). `make models-fetch`
prefetches them; worker-c reads its own export via `make models-export`. See
[upgrading.md](upgrading.md).

## Linux

- Python >= 3.10 required.

```bash
uv sync
uv run demucs-mlx-worker /path/to/audio.wav
```

The raylib GUI is not packaged for Linux.
