"""Flatten MLX HTDemucs weights and dump C-worker config / goldens."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .mlx_convert import BagOfModelsMLX
from .mlx_htdemucs import HTDemucsMLX


def flatten_params(obj: Any, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested MLX parameter trees to dotted keys."""
    out: dict[str, mx.array] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_params(value, path))
        return out
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_params(value, path))
        return out
    if isinstance(obj, mx.array):
        if not prefix:
            raise ValueError("parameter array missing key")
        out[prefix] = obj
        return out
    return out


def unwrap_htdemucs(model: Any) -> HTDemucsMLX:
    if isinstance(model, BagOfModelsMLX):
        if len(model.models) != 1:
            raise ValueError(
                f"C worker supports a single HTDemucs, got bag of {len(model.models)}"
            )
        inner = model.models[0]
    else:
        inner = model
    if not isinstance(inner, HTDemucsMLX):
        raise TypeError(f"expected HTDemucsMLX, got {type(inner).__name__}")
    return inner


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def model_config_dict(model: HTDemucsMLX) -> dict[str, Any]:
    args, kwargs = model._init_args_kwargs
    kwargs = dict(kwargs)
    kwargs["segment"] = float(model.segment)
    return {
        "model": "htdemucs_6s",
        "model_class": type(model).__name__,
        "args": _jsonable(list(args)),
        "kwargs": _jsonable(dict(kwargs)),
        "sources": list(model.sources),
        "samplerate": int(model.samplerate),
        "segment": float(model.segment),
        "nfft": int(model.nfft),
        "hop_length": int(model.hop_length),
        "audio_channels": int(model.audio_channels),
        "channels": int(model.channels),
        "depth": int(model.depth),
        "kernel_size": int(model.kernel_size),
        "stride": int(model.stride),
        "context": int(model.context),
        "cac": bool(model.cac),
        "use_train_segment": bool(model.use_train_segment),
        "wiener_iters": int(model.wiener_iters),
        "valid_length": int(model.valid_length(1)),
        "channels_time": kwargs.get("channels_time"),
        "growth": kwargs.get("growth"),
        "time_stride": kwargs.get("time_stride"),
        "context_enc": kwargs.get("context_enc"),
        "norm_groups": kwargs.get("norm_groups"),
        "norm_starts": kwargs.get("norm_starts"),
        "dconv_mode": kwargs.get("dconv_mode"),
        "dconv_depth": kwargs.get("dconv_depth"),
        "dconv_comp": kwargs.get("dconv_comp"),
        "dconv_init": kwargs.get("dconv_init"),
        "rewrite": kwargs.get("rewrite"),
        "freq_emb": kwargs.get("freq_emb"),
        "emb_scale": kwargs.get("emb_scale"),
        "emb_smooth": kwargs.get("emb_smooth"),
        "t_layers": kwargs.get("t_layers"),
        "t_heads": kwargs.get("t_heads"),
        "t_hidden_scale": kwargs.get("t_hidden_scale"),
        "t_emb": kwargs.get("t_emb"),
        "t_max_period": kwargs.get("t_max_period"),
        "t_weight_pos_embed": kwargs.get("t_weight_pos_embed"),
        "t_norm_in": kwargs.get("t_norm_in"),
        "t_norm_first": kwargs.get("t_norm_first"),
        "t_norm_out": kwargs.get("t_norm_out"),
        "t_layer_scale": kwargs.get("t_layer_scale"),
        "t_gelu": kwargs.get("t_gelu"),
        "t_cross_first": kwargs.get("t_cross_first"),
        "bottom_channels": kwargs.get("bottom_channels"),
    }


def save_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, value in tensors.items():
        mx.eval(value)
        # safetensors writes the raw buffer; non-contiguous MLX views
        # (transposes) would silently save the wrong memory order.
        arrays[key] = np.ascontiguousarray(np.array(value, copy=True))
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to export C-worker weights. "
            "Install with: uv pip install safetensors"
        ) from exc
    save_file(arrays, str(path))


def write_keys_manifest(path: Path, tensors: dict[str, mx.array]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key in sorted(tensors):
        value = tensors[key]
        mx.eval(value)
        shape = ",".join(str(int(d)) for d in value.shape)
        lines.append(f"{key}\t{shape}\t{value.dtype}")
    path.write_text("\n".join(lines) + "\n")


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB. Arrays are (channels, time) or (time,)."""
    ref_f = ref.reshape(-1).astype(np.float64)
    est_f = est.reshape(-1).astype(np.float64)
    if ref_f.size != est_f.size:
        raise ValueError(f"si_sdr size mismatch {ref_f.size} vs {est_f.size}")
    alpha = np.dot(est_f, ref_f) / (np.dot(ref_f, ref_f) + eps)
    e_target = alpha * ref_f
    e_noise = est_f - e_target
    num = np.dot(e_target, e_target) + eps
    den = np.dot(e_noise, e_noise) + eps
    return float(10.0 * math.log10(num / den))


def write_wav_pcm16(path: Path, audio: np.ndarray, samplerate: int) -> None:
    """Write (channels, frames) float32 in [-1, 1] as stereo/mono PCM16 WAV."""
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    if pcm.ndim == 1:
        pcm = pcm[None, :]
    if pcm.shape[0] not in (1, 2):
        raise ValueError(f"expected 1 or 2 channels, got {pcm.shape}")
    interleaved = np.ascontiguousarray(
        (pcm.T * 32767.0).astype(np.int16)
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(pcm.shape[0]))
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(interleaved.tobytes())


def read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV to (channels, frames) float32."""
    import wave

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    pcm = pcm.reshape(-1, nch).T
    return pcm, int(sr)
