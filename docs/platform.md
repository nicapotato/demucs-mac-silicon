# Platform notes

This project is uv-first on macOS Apple Silicon (MLX). The raylib GUI lives in
[demucs-ui-app](https://github.com/nicapotato/demucs-ui-app). Windows CUDA inference
lives in [demucs-windows](https://github.com/nicapotato/demucs-windows).

## macOS (Apple Silicon)

- Apple Silicon is supported via MLX.
- Audio I/O is handled natively by mlx-audio-io (no FFmpeg required).

Typical flow:

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

Freeze the standalone worker (consumed by demucs-ui-app):

```bash
uv sync --extra convert
DEMUCS_MLX_PYTHON=.venv/bin/python bash scripts/freeze_worker.sh
```

Env vars:

| Variable | Purpose |
|----------|---------|
| `DEMUCS_MLX_PYTHON` | Python that has `demucs_mlx` (freeze / GUI) |
| `DEMUCS_MLX_WORKER` | Path to frozen MLX worker binary |
| `DEMUCS_MLX_CACHE` | Directory with `htdemucs_6s_mlx.pkl` |
| `DEMUCS_MLX_WORKER_DIST` | Override freeze output dir (default `dist/worker`) |

## Linux

- Python >= 3.10 required.

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

The raylib GUI is not packaged for Linux.
