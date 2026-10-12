# demucs-mac-silicon

Apple Silicon Demucs workers, consumed by [demucs-ui-app](https://github.com/nicapotato/demucs-ui-app):

| Worker | What it is |
|---|---|
| **worker-py** (`worker_py/`) | Thin wrapper over **upstream [demucs-mlx](https://github.com/ssmall256/demucs-mlx), pinned exactly**. Adds the GUI contract (JSON progress events, MP3, `--track-name`). Frozen with PyInstaller. |
| **worker-c** (`worker-c/`) | Independent C implementation of htdemucs_6s on [mlx-c](https://github.com/ml-explore/mlx-c). ~30 MB, no Python. |

The standalone GUI lives in **demucs-ui-app**. Windows CUDA lives in
[demucs-windows](https://github.com/nicapotato/demucs-windows). Version: [`project.conf`](project.conf).

demucs-mlx is no longer vendored or forked here. See **[docs/upgrading.md](docs/upgrading.md)**
for how upstream changes are tracked and how to be sure they work.

## Requirements

- macOS on Apple Silicon, Python >= 3.10, [uv](https://docs.astral.sh/uv/)
- CMake + Xcode command line tools (worker-c only)
- Homebrew `lame` + `mpg123` for worker-c MP3 (`make native-deps`)

## Quick start

```bash
make sync                 # uv sync --extra dev --extra freeze
make native-deps          # brew install lame mpg123 (worker-c --mp3)
make models-fetch         # upstream weights -> models/mlx-cache (digest-checked)
uv run demucs-mlx-worker song.wav -o out -n htdemucs_6s
```

Also `python -m worker_py ...`. Options:

```
-n, --name            Model name (default: htdemucs_6s)
-o, --out             Output directory (default: separated)
--track-name NAME     Output subfolder (default: input stem)
--shifts N            Random shifts (default: 1)       --seed N
--overlap F           Overlap ratio (default: 0.25)    --segment SECONDS
-b, --batch-size      int or auto (default: auto)
--no-split            Disable chunked inference
--attention fp16|fp32 Attention kernel precision (upstream default: fp16)
--compile/--no-compile
--mp3 [--mp3-bitrate N]   MP3 stems instead of WAV
--progress-json       Line-delimited JSON events on stdout (the GUI contract)
--prefetch-tracks N   Accepted and ignored (GUI compatibility)
--offline             Never download weights
--list-models
```

JSON events (`worker_py/events.py`; worker-c emits the same stream):
`status` (`loading_model`, `model_ready`), `loading`, `separating` (`done`, `total`, `pct`),
`writing`, `done`, `error`.

Environment:

| Variable | Purpose |
|---|---|
| `DEMUCS_MLX_CACHE_DIR` | Upstream weights cache (wins) |
| `DEMUCS_MLX_CACHE` | GUI's historic name. Used as the cache dir, or `<dir>/mlx` if `<dir>` holds a C-worker export |
| `DEMUCS_MLX_NO_DOWNLOAD=1` | Never download weights (same as `--offline`) |
| `DEMUCS_MLX_PYTHON` | Python used by `scripts/freeze_worker.sh` |
| `DEMUCS_MLX_WORKER_DIST` | Freeze output dir (default `dist/worker`) |

## Build

```bash
make worker-pyinstaller   # frozen worker-py  -> dist/worker/demucs_mlx_worker/
make worker               # worker-c + metallib -> dist/worker/demucs_mlx_worker (file)
make models-export        # C weights/config/keys + models/export_manifest.json
```

## Verify and upgrade upstream

```bash
make verify          # lint, drift check, C build+tests, frozen build, parity + contract tests
make upgrade-check   # after bumping the demucs-mlx pin: relock, contract check, re-record, verify
```

A weekly workflow (`upstream-watch`) opens a PR when PyPI has a newer demucs-mlx.

## Release

Bump `VERSION` in `project.conf`, then:

```bash
make release-watch
```

Runs the same gate, then tags `v$VERSION` and publishes the frozen worker-py zip, the
worker-c zip, upstream-format weights (`htdemucs_6s.safetensors` + config) and
`export_manifest.json` (pinned upstream version and hashes) on a GitHub Release.

## Documentation

- Upstream tracking: `docs/upgrading.md`
- Development: `docs/development.md`
- Release: `docs/release.md`
- Platform notes: `docs/platform.md`
- Python API (upstream's): `docs/api.md`
- Patches we want upstream: `upstream-patches/README.md`

## License

MIT. Built on [demucs-mlx](https://github.com/ssmall256/demucs-mlx) and
[Demucs](https://github.com/adefossez/demucs) by Meta Research. See `LICENSE`.
