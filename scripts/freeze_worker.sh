#!/usr/bin/env bash
# Freeze demucs-mlx CLI into a standalone onedir worker with PyInstaller.
# Intentionally excludes torch/demucs (conversion-only deps). Inference uses
# pre-bundled MLX weights (DEMUCS_MLX_CACHE / models/).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${DEMUCS_MLX_PYTHON:-$REPO/.venv/bin/python}"
DIST="${DEMUCS_MLX_WORKER_DIST:-$REPO/dist/worker}"
SPEC_WORK="$REPO/dist/pyi"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing python at $PYTHON — create repo .venv first" >&2
  exit 1
fi

"$PYTHON" -c "import PyInstaller" 2>/dev/null || "$PYTHON" -m pip install pyinstaller

rm -rf "$DIST" "$SPEC_WORK"
mkdir -p "$SPEC_WORK"

ENTRY="$SPEC_WORK/demucs_mlx_worker.py"
cat >"$ENTRY" <<'PY'
"""Frozen demucs-mlx worker entry.

mlx-audio-io refuses to load its native .so unless the on-disk bytes match the
wheel RECORD hash and pass codesign --verify. PyInstaller / ad-hoc codesign of
the .app changes those bytes, so we skip those two checks when frozen. OS/arch/
Python/MLX compatibility checks still run.
"""
import sys


def _patch_mlx_audio_io_preflight() -> None:
    if not (getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")):
        return
    import mlx_audio_io._native_loader as nl

    nl.verify_record_hash = lambda native_path: None  # noqa: ARG005
    nl.verify_codesign = lambda native_path: None  # noqa: ARG005


_patch_mlx_audio_io_preflight()

from demucs_mlx.separate import main

raise SystemExit(main())
PY

cd "$SPEC_WORK"
# --paths so editable installs still resolve the source package
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name demucs_mlx_worker \
  --paths "$REPO" \
  --collect-all mlx \
  --collect-all mlx_audio_io \
  --collect-all mlx_spectro \
  --collect-submodules demucs_mlx \
  --hidden-import lameenc \
  --hidden-import demucs_mlx.separate \
  --hidden-import demucs_mlx.model_converter \
  --hidden-import demucs_mlx.apply_mlx \
  --hidden-import demucs_mlx.audio \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module torchaudio \
  --exclude-module demucs \
  --exclude-module tensorflow \
  --exclude-module jax \
  --distpath "$DIST" \
  --workpath "$SPEC_WORK/build" \
  "$ENTRY"

BIN="$DIST/demucs_mlx_worker/demucs_mlx_worker"
test -x "$BIN"
echo "Worker frozen at $BIN"
du -sh "$DIST/demucs_mlx_worker"
