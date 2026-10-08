#!/usr/bin/env python3
"""Dump HTDemucs_6s intermediate tensors for C numeric gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x, copy=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--name", default="htdemucs_6s")
    parser.add_argument(
        "-o",
        "--out",
        default=str(ROOT / "goldens" / "htdemucs_6s"),
        help="Output directory for golden tensors",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Clip length in seconds (default: model.segment)",
    )
    args = parser.parse_args(argv)

    from demucs_mlx.c_export import (
        flatten_params,
        model_config_dict,
        save_safetensors,
        unwrap_htdemucs,
        write_wav_pcm16,
    )
    from demucs_mlx.mlx_utils import center_trim
    from demucs_mlx.model_converter import get_mlx_model

    model = unwrap_htdemucs(get_mlx_model(args.name))
    if hasattr(model, "eval"):
        model.eval()

    mx.random.seed(int(args.seed))
    sr = int(model.samplerate)
    seconds = float(model.segment if args.seconds is None else args.seconds)
    frames = int(sr * seconds)
    mix = mx.random.normal((1, 2, frames)).astype(mx.float32)
    mx.eval(mix)

    goldens: dict[str, mx.array] = {"mix": mix[0]}
    length = mix.shape[-1]
    length_pre_pad = None
    if model.use_train_segment:
        training_length = int(model.segment * model.samplerate)
        if mix.shape[-1] < training_length:
            length_pre_pad = mix.shape[-1]
            mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length_pre_pad)])

    z = model._spec(mix)
    mag = model._magnitude(z)
    goldens["spec_real"] = mx.real(z)
    goldens["spec_imag"] = mx.imag(z)
    goldens["mag"] = mag

    x = mag
    B, C, Fq, T = x.shape
    mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
    std = mx.std(x, axis=(1, 2, 3), keepdims=True)
    x = (x - mean) / (1e-5 + std)
    xt = mix
    meant = mx.mean(xt, axis=(1, 2), keepdims=True)
    stdt = mx.std(xt, axis=(1, 2), keepdims=True)
    xt = (xt - meant) / (1e-5 + stdt)
    goldens["x_norm"] = x
    goldens["xt_norm"] = xt

    saved = []
    saved_t = []
    lengths = []
    lengths_t = []
    for idx, encode in enumerate(model.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(model.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = model.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                saved_t.append(xt)
            else:
                inject = xt
        x = encode(x, inject)
        if idx == 0 and model.freq_emb is not None:
            frs = mx.arange(x.shape[-2], dtype=mx.int32)
            emb = model.freq_emb(frs).transpose(1, 0)[None, :, :, None]
            x = x + model.freq_emb_scale * emb
        saved.append(x)
        goldens[f"enc_{idx}"] = mx.array(x)
        goldens[f"tenc_{idx}"] = mx.array(xt)

    if model.crosstransformer:
        if model.bottom_channels:
            b, c, f, t = x.shape
            x = x.reshape(b, c, f * t)
            x = model.channel_upsampler(x)
            x = x.reshape(b, model.bottom_channels, f, t)
            xt = model.channel_upsampler_t(xt)
        x, xt = model.crosstransformer(x, xt)
        if model.bottom_channels:
            x = x.reshape(b, model.bottom_channels, f * t)
            x = model.channel_downsampler(x)
            x = x.reshape(b, c, f, t)
            xt = model.channel_downsampler_t(xt)
        goldens["transformer_x"] = mx.array(x)
        goldens["transformer_xt"] = mx.array(xt)

    offset = model.depth - len(model.tdecoder)
    for idx, decode in enumerate(model.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        if idx >= offset:
            tdec = model.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip_t = saved_t.pop(-1)
                xt, _ = tdec(xt, skip_t, length_t)
        goldens[f"dec_{idx}"] = x
        goldens[f"tdec_{idx}"] = xt

    S = len(model.sources)
    x = x.reshape(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]
    zout = model._mask(z, x)
    goldens["zout_real"] = mx.real(zout)
    goldens["zout_imag"] = mx.imag(zout)
    if model.use_train_segment:
        x = model._ispec(zout, training_length)
    else:
        x = model._ispec(zout, length)
    goldens["ispec"] = x

    actual_length = xt.shape[-1]
    xt = xt.reshape(B, S, -1, actual_length)
    xt = xt * stdt[:, None] + meant[:, None]
    x = center_trim(x, xt)
    x = xt + x
    if model.use_train_segment:
        x = x[..., :training_length]
    else:
        x = x[..., :length]
    if length_pre_pad:
        x = x[..., :length_pre_pad]
    goldens["stems"] = x[0]
    mx.eval(*goldens.values())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors(out_dir / "goldens.safetensors", goldens)
    write_wav_pcm16(out_dir / "mix.wav", _to_numpy(goldens["mix"]), sr)
    meta = model_config_dict(model)
    meta["seed"] = int(args.seed)
    meta["mix_frames"] = int(frames)
    meta["param_count"] = len(flatten_params(model.parameters()))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_dir / 'goldens.safetensors'} ({len(goldens)} tensors)")
    print(f"wrote {out_dir / 'mix.wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
