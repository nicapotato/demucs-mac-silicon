# Development

uv-first. `uv.lock` pins everything; `demucs-mlx` is pinned with `==` in `pyproject.toml`.

## Setup

```bash
make sync            # uv sync --extra dev --extra freeze
make native-deps     # brew install lame mpg123 (worker-c --mp3)
make models-fetch    # weights -> models/mlx-cache
```

## Everyday

```bash
make test-fast       # ruff + pyright + tests that need no weights/binaries
make test            # full pytest (model / worker-c / frozen tests skip if artifacts missing)
make verify          # build everything, then run every gate (nothing skips)
```

Format: `uv run ruff format worker_py` (CI checks `ruff check`).

## Layout

| Path | Role |
|---|---|
| `worker_py/` | GUI-facing wrapper over upstream: `cli`, `events`, `progress`, `mp3`, `cache` |
| `worker_py/c_export.py`, `contract.py` | export for worker-c and the drift/contract checks |
| `worker_py/parity.py` | run two workers on a clip and compare |
| `worker-c/` | C worker (mlx-c); `src/test_*.c` unit tests |
| `scripts/` | `export_safetensors.py`, `dump_goldens.py`, `freeze_worker.sh` |
| `models/` | `mlx-cache/` (upstream weights, ignored), C export + tracked `*.keys.txt`, `*.json`, `export_manifest.json` |
| `goldens/htdemucs_6s/` | numeric goldens for the C worker (only `meta.json` is tracked) |
| `upstream-patches/` | changes we want upstream |

## Goldens

`make goldens` regenerates `goldens/htdemucs_6s/` from the pinned upstream. It reaches into
upstream model internals, so it may need a fix after a bump. The end-to-end py-vs-c gate is
the main check; goldens localise a failure.

## Upgrading upstream

See [upgrading.md](upgrading.md).
