"""Model-cache directory resolution.

The GUI sets ``DEMUCS_MLX_CACHE`` (historic name). Upstream demucs-mlx reads
``DEMUCS_MLX_CACHE_DIR`` and stores ``<model>.safetensors`` + ``<model>_config.json``.

The C worker's export uses the *same* ``<model>.safetensors`` filename with
different tensor keys (no ``model_0.`` prefix) plus ``<model>.json``. If the GUI
points both workers at one directory, upstream would see the C export as a broken
cache and overwrite it. So when the directory holds a C export, the Python worker
uses a ``mlx/`` subdirectory instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

UPSTREAM_ENV = "DEMUCS_MLX_CACHE_DIR"
LEGACY_ENV = "DEMUCS_MLX_CACHE"
SUBDIR = "mlx"


def has_c_export(path: Path) -> bool:
    """True if ``path`` holds C-worker exports (``X.safetensors`` + ``X.json``)."""
    if not path.is_dir():
        return False
    for st in path.glob("*.safetensors"):
        stem = st.stem
        if (path / f"{stem}.json").is_file() and not (path / f"{stem}_config.json").exists():
            return True
    return False


def resolve_cache_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Directory upstream demucs-mlx should use, or None to keep its default."""
    env = os.environ if env is None else env
    explicit = (env.get(UPSTREAM_ENV) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    legacy = (env.get(LEGACY_ENV) or "").strip()
    if not legacy:
        return None
    path = Path(legacy).expanduser()
    return path / SUBDIR if has_c_export(path) else path


def apply_cache_env(env: dict[str, str] | None = None) -> Path | None:
    """Export the resolved cache dir as ``DEMUCS_MLX_CACHE_DIR`` (in place)."""
    target = os.environ if env is None else env
    resolved = resolve_cache_dir(target)
    if resolved is not None:
        target[UPSTREAM_ENV] = str(resolved)
    return resolved
