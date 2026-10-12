"""What worker-c depends on from upstream, and how we detect drift.

worker-c never runs upstream code. It depends on:

* the exported weights (tensor names, shapes, values)  -> ``models/htdemucs_6s.keys.txt``
* the model config                                     -> ``models/htdemucs_6s.json``
* default inference parameters (overlap, shifts, split, segment)
* the MLX version its mlx-c build embeds

``models/export_manifest.json`` records these for the currently pinned upstream.
``tests/test_contract.py`` fails when any of them drifts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "export_manifest.json"
KEYS_PATH = ROOT / "models" / "htdemucs_6s.keys.txt"
CONFIG_PATH = ROOT / "models" / "htdemucs_6s.json"
MODEL_NAME = "htdemucs_6s"
#: Prefix upstream's cache puts on every tensor (a bag of one model).
UPSTREAM_PREFIX = "model_0."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upstream_version() -> str:
    return metadata.version("demucs-mlx")


def mlx_version() -> str:
    return metadata.version("mlx")


def cmake_mlx_c_tag(root: Path = ROOT) -> str:
    text = (root / "worker-c" / "CMakeLists.txt").read_text()
    match = re.search(r"mlx-c\.git\s+GIT_TAG\s+(\S+?)\)", text)
    if not match:
        raise RuntimeError("could not find the mlx-c GIT_TAG in worker-c/CMakeLists.txt")
    return match.group(1)


def built_mlx_version(root: Path = ROOT) -> str | None:
    """MLX version embedded by the local worker-c build (None if not built)."""
    header = root / "worker-c" / "build" / "_deps" / "mlx-src" / "mlx" / "version.h"
    if not header.is_file():
        return None
    text = header.read_text()
    parts = [re.search(rf"MLX_VERSION_{k}\s+(\d+)", text) for k in ("MAJOR", "MINOR", "PATCH")]
    if not all(parts):
        return None
    return ".".join(m.group(1) for m in parts if m)


def c_worker_defaults(root: Path = ROOT) -> dict[str, Any]:
    """Defaults hard-coded in worker-c/src/main.c."""
    text = (root / "worker-c" / "src" / "main.c").read_text()

    def grab(pattern: str) -> str:
        m = re.search(pattern, text)
        if not m:
            raise RuntimeError(f"main.c: pattern {pattern!r} not found")
        return m.group(1)

    return {
        "overlap": float(grab(r"float\s+overlap\s*=\s*([0-9.]+)f?;")),
        "shifts": int(grab(r"int\s+shifts\s*=\s*(\d+);")),
        "split": bool(int(grab(r"int\s+split\s*=\s*(\d+);"))),
    }


def upstream_defaults() -> dict[str, Any]:
    """Defaults of upstream ``Separator`` (the Python reference behavior)."""
    from demucs_mlx import Separator

    params = inspect.signature(Separator.__init__).parameters
    return {
        "overlap": float(params["overlap"].default),
        "shifts": int(params["shifts"].default),
        "split": bool(params["split"].default),
    }


def tensor_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def strip_upstream_prefix(key: str) -> str:
    return key[len(UPSTREAM_PREFIX) :] if key.startswith(UPSTREAM_PREFIX) else key


def build_manifest(
    *,
    cache_file: Path,
    export_file: Path,
    num_tensors: int,
    root: Path = ROOT,
) -> dict[str, Any]:
    previous = load_manifest() or {}
    built = built_mlx_version(root) or previous.get("mlx_c", {}).get("embedded_mlx")
    return {
        "schema": 1,
        "model": MODEL_NAME,
        "upstream": {"package": "demucs-mlx", "version": upstream_version()},
        "mlx_python": mlx_version(),
        "mlx_c": {"tag": cmake_mlx_c_tag(root), "embedded_mlx": built},
        "source_cache_sha256": sha256_file(cache_file),
        "export": {
            "safetensors_sha256": sha256_file(export_file),
            "num_tensors": num_tensors,
            "keys_sha256": hashlib.sha256(KEYS_PATH.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        },
        "defaults": c_worker_defaults(root),
    }


def load_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text())


def write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def check_against_committed(
    res: dict[str, Any], *, allow_version_change: bool = False
) -> list[str]:
    """Compare a fresh ``export_model`` result with the committed manifest + files.

    ``allow_version_change`` is for the upgrade flow only: a bumped demucs-mlx may
    differ from the recorded version (``make models-export`` then re-records it), but
    tensors, config and weight bytes must still match.
    """
    problems: list[str] = []
    manifest = load_manifest()
    if manifest is None:
        return ["models/export_manifest.json is missing (run: make models-export)"]

    fresh_keys = tensor_lines(Path(res["keys"]))
    committed_keys = tensor_lines(KEYS_PATH)
    if fresh_keys != committed_keys:
        fresh, committed = set(fresh_keys), set(committed_keys)
        added = sorted(fresh - committed)[:5]
        removed = sorted(committed - fresh)[:5]
        problems.append(f"tensor names/shapes changed; added={added} removed={removed}")

    if Path(res["config"]).read_text() != CONFIG_PATH.read_text():
        problems.append("model config json changed")

    exp = manifest["export"]
    if sha256_file(Path(res["safetensors"])) != exp["safetensors_sha256"]:
        problems.append("exported weight bytes changed (values or dtype)")
    if not allow_version_change and manifest["upstream"]["version"] != upstream_version():
        problems.append(
            f"manifest recorded demucs-mlx {manifest['upstream']['version']}, "
            f"installed {upstream_version()} (re-run make models-export to record it)"
        )
    return problems
