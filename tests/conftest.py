"""Shared fixtures. Model/worker-dependent tests skip cleanly when artifacts are absent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "models" / "mlx-cache"
C_WORKER = ROOT / "worker-c" / "build" / "demucs_mlx_worker"
# `make verify` freezes into dist/worker-py (dist/worker also holds the C worker copy).
FROZEN_CANDIDATES = [
    ROOT / "dist" / "worker-py" / "demucs_mlx_worker" / "demucs_mlx_worker",
    ROOT / "dist" / "worker" / "demucs_mlx_worker" / "demucs_mlx_worker",
]
SAMPLERATE = 44100


def pytest_configure(config):
    # Never let a test fall through to a network download of weights.
    os.environ.setdefault("DEMUCS_MLX_CACHE_DIR", str(CACHE_DIR))
    os.environ.setdefault("DEMUCS_MLX_NO_DOWNLOAD", "1")


def _has_model() -> bool:
    return (CACHE_DIR / "htdemucs_6s.safetensors").is_file() and (
        CACHE_DIR / "htdemucs_6s_config.json"
    ).is_file()


@pytest.fixture(scope="session")
def model_cache() -> Path:
    if not _has_model():
        pytest.skip(f"no htdemucs_6s weights in {CACHE_DIR} (run: make models-fetch)")
    return CACHE_DIR


@pytest.fixture(scope="session")
def c_worker() -> Path:
    if not C_WORKER.is_file():
        pytest.skip("worker-c not built (run: make worker-c)")
    return C_WORKER


@pytest.fixture(scope="session")
def frozen_worker() -> Path:
    for candidate in FROZEN_CANDIDATES:
        if candidate.is_file():
            return candidate
    pytest.skip("frozen worker not built (run: make worker-py-verify-build)")


@pytest.fixture(scope="session")
def clip_wav(tmp_path_factory) -> Path:
    """Deterministic 12 s stereo clip: a few tones plus noise bursts (>1 segment)."""
    from worker_py.c_export import write_wav_pcm16

    rng = np.random.default_rng(1234)
    t = np.arange(SAMPLERATE * 12) / SAMPLERATE
    left = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 659 * t)
    burst = (np.sin(2 * np.pi * 2 * t) > 0.6) * rng.standard_normal(t.size) * 0.1
    right = 0.2 * np.sin(2 * np.pi * 330 * t + 0.3) + burst
    audio = np.stack([left + burst * 0.5, right]).astype(np.float32)
    path = tmp_path_factory.mktemp("clips") / "clip.wav"
    write_wav_pcm16(path, audio, SAMPLERATE)
    return path
