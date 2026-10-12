# Release

1. `make verify` (CI runs the same gate before publishing).
2. Bump `VERSION` in `project.conf`.
3. `make release-watch` (dispatches `.github/workflows/release.yml`).

The workflow: reads the version and pinned upstream, runs `ci.yml` (`make verify`), checks the
downloaded weights hash against `models/export_manifest.json`, builds worker-c and the frozen
worker-py, then publishes on tag `v$VERSION`:

| Asset | Purpose |
|---|---|
| `demucs-mlx-worker-py-mac-arm64-vX.Y.Z.zip` | frozen worker_py (PyInstaller onedir) |
| `demucs-mlx-worker-c-mac-arm64-vX.Y.Z.zip` | worker-c binary + `mlx.metallib` |
| `htdemucs_6s.safetensors`, `htdemucs_6s_config.json` | upstream-format weights for the Python worker / offline bundles (`DEMUCS_MLX_CACHE_DIR` layout) |
| `htdemucs_6s.c.safetensors`, `htdemucs_6s.json` | C-worker export (same tensors, no `model_0.` prefix). Rename the safetensors to `htdemucs_6s.safetensors` next to the json. Not stored in git (~105 MB). |
| `export_manifest.json` | pinned demucs-mlx version, hashes, mlx / mlx-c pairing |
| `SHA256SUMS` | checksums |

The old converted pickle (`htdemucs_6s_mlx.pkl`) and the immutable `MODELS_RELEASE_TAG` pin are
gone: upstream now distributes digest-checked safetensors, and the manifest records the exact
hash each release was validated against.

Upstream release history lives at https://github.com/ssmall256/demucs-mlx/releases.
