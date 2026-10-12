"""MP3 stem encoding via lameenc (upstream demucs-mlx only writes WAV/FLAC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rescale_no_clip(wav: np.ndarray) -> np.ndarray:
    """Same 'rescale' mode as upstream save_audio: divide by max(1.01 * peak, 1)."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav / max(1.01 * peak, 1.0)


def to_pcm16_interleaved(wav: np.ndarray) -> tuple[bytes, int]:
    """(channels, frames) float32 -> interleaved int16 bytes and channel count."""
    wav = rescale_no_clip(wav.astype(np.float32, copy=False))
    if wav.ndim == 1:
        pcm = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
        return pcm.tobytes(), 1
    pcm = np.clip(wav.T * 32767.0, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(pcm).tobytes(), int(wav.shape[0])


def save_mp3(wav: np.ndarray, path: Path, samplerate: int, bitrate: int = 128) -> None:
    import lameenc

    data, channels = to_pcm16_interleaved(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(bitrate))
    encoder.set_in_sample_rate(int(samplerate))
    encoder.set_channels(channels)
    encoder.set_quality(2)
    out = encoder.encode(data) + encoder.flush()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
