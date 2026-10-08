"""Unit tests for C-worker export helpers (no model weights required)."""
from __future__ import annotations

import math

import numpy as np

from demucs_mlx.c_export import flatten_params, read_wav_float, save_safetensors, si_sdr, write_wav_pcm16


def test_flatten_params_nested():
    tree = {"a": {"b": [1, 2]}, "c": 3}
    # flatten_params only keeps mx.array leaves; ints are ignored.
    out = flatten_params(tree)
    assert out == {}


def test_si_sdr_identical():
    x = np.ones((2, 100), dtype=np.float32)
    v = si_sdr(x, x)
    assert math.isfinite(v)
    assert v > 50.0


def test_save_safetensors_noncontiguous(tmp_path):
    import mlx.core as mx
    from safetensors.numpy import load_file

    x = mx.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(mx.float32)
    y = mx.transpose(x, (0, 3, 2, 1))
    path = tmp_path / "t.safetensors"
    save_safetensors(path, {"y": y})
    loaded = load_file(str(path))
    ref = np.ascontiguousarray(np.array(y))
    assert loaded["y"].shape == ref.shape
    assert float(np.max(np.abs(loaded["y"] - ref))) == 0.0


def test_wav_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((2, 1024)).astype(np.float32) * 0.1
    path = tmp_path / "t.wav"
    write_wav_pcm16(path, audio, 44100)
    back, sr = read_wav_float(path)
    assert sr == 44100
    assert back.shape == audio.shape
    assert float(np.max(np.abs(back - audio))) < 2.0 / 32768.0 + 1e-4


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_flatten_params_nested()
    test_si_sdr_identical()
    with tempfile.TemporaryDirectory() as d:
        test_save_safetensors_noncontiguous(Path(d))
        test_wav_roundtrip(Path(d))
    print("test_c_export ok")
