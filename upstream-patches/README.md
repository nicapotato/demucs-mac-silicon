# Upstream patches

Changes we want in [ssmall256/demucs-mlx](https://github.com/ssmall256/demucs-mlx) so
`worker_py` does not have to patch upstream internals at runtime.

| Patch | Replaces | Status |
|-------|----------|--------|
| `0001-progress-callback.patch` | `worker_py/progress.py` (tqdm + `apply_model` shim) | not submitted yet |

Generated against **v1.6.1** (`git apply --check` clean; its tests pass).

## 0001: `progress_callback`

Adds `progress_callback: Callable[[int, int], None]` to `apply_model(...)` and
`Separator(...)`. It is called as `(done_segments, total_segments)`: `(0, total)` at the
start of each forward pass (one per shift, per bag member), then after every batch.
Independent of the tqdm `progress` flag, so it works with default `shifts=1`.

Why: upstream's `apply_model` forces `progress=False` on its per-shift recursion, so
with the default `shifts=1` no progress can be observed at all. Our GUI needs it.

### Submit

```bash
gh repo fork ssmall256/demucs-mlx --clone=false
git clone git@github.com:<you>/demucs-mlx.git && cd demucs-mlx
git checkout -b progress-callback v1.6.1
git am /path/to/0001-progress-callback.patch   # or: git apply + commit
gh pr create --repo ssmall256/demucs-mlx --title "Add progress_callback to apply_model and Separator"
```

### After it ships

1. Bump the pin in `pyproject.toml` to the release containing it.
2. `worker_py/cli.py` already prefers it: `progress.supports_native_callback()` detects
   `Separator(progress_callback=...)`.
3. Delete the shim in `worker_py/progress.py` and `tests/test_progress_shim.py`.
