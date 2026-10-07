# Development

This project is uv-first and uses `uv.lock` to pin dependencies.

## Setup

```bash
uv lock
uv sync --extra dev
```

## Lint and type check

```bash
uv run ruff check demucs_mlx tests
uv run pyright
```

## Format

```bash
uv run ruff format demucs_mlx
```

## Tests

```bash
make test
```

## Build

```bash
uv run python -m build
```

## Release

Bump `VERSION` in `project.conf` (and `MODELS_RELEASE_TAG` only if the MLX pickle
must change), then:

```bash
make release-watch
```

This tags `v$VERSION` and stores the frozen worker + pickle on a GitHub Release.
