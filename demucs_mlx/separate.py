"""MLX-only CLI for Demucs stem separation."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .mlx_registry import MLX_MODEL_REGISTRY

# When True, emit one JSON object per line on stdout (no tqdm / print noise).
_PROGRESS_JSON = False


def _emit(event: str, **fields: tp.Any) -> None:
    if not _PROGRESS_JSON:
        return
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _save_mp3(wav: np.ndarray, path: Path, samplerate: int, bitrate: int = 128) -> None:
    """Encode float32 (channels, frames) audio to MP3 via lameenc."""
    import lameenc

    from .audio import _prevent_clip_numpy

    clipped = _prevent_clip_numpy(wav.astype(np.float32, copy=False), mode="rescale")
    if clipped.ndim == 1:
        channels = 1
        pcm = np.clip(clipped * 32767.0, -32768, 32767).astype(np.int16)
        interleaved = pcm.tobytes()
    else:
        channels = int(clipped.shape[0])
        # (channels, frames) -> interleaved int16
        pcm = np.clip(clipped.T * 32767.0, -32768, 32767).astype(np.int16)
        interleaved = np.ascontiguousarray(pcm).tobytes()

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(bitrate))
    encoder.set_in_sample_rate(int(samplerate))
    encoder.set_channels(channels)
    encoder.set_quality(2)
    mp3_data = encoder.encode(interleaved)
    mp3_data += encoder.flush()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mp3_data)


class _AsyncWriter:
    def __init__(
        self,
        maxsize: int = 4,
        workers: int = 1,
        *,
        clip: str = "rescale",
        bits_per_sample: int = 16,
        as_float: bool = False,
        fmt: str = "wav",
        mp3_bitrate: int = 128,
    ):
        if workers <= 0:
            raise ValueError("workers must be > 0")
        if fmt not in ("wav", "mp3"):
            raise ValueError("fmt must be 'wav' or 'mp3'")
        self._queue: "queue.Queue[tp.Optional[tuple]]" = queue.Queue(maxsize=maxsize)
        self._error: tp.Optional[BaseException] = None
        self._workers = int(workers)
        self._clip = clip
        self._bits_per_sample = int(bits_per_sample)
        self._as_float = bool(as_float)
        self._fmt = fmt
        self._mp3_bitrate = int(mp3_bitrate)
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name=f"demucs-writer-{i}")
            for i in range(self._workers)
        ]
        for thread in self._threads:
            thread.start()

    def _run(self) -> None:
        from .audio import save_audio

        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._queue.task_done()
                    break
                wav, path, samplerate = item
                if self._fmt == "mp3":
                    _save_mp3(wav, path, samplerate, bitrate=self._mp3_bitrate)
                else:
                    save_audio(
                        wav,
                        path,
                        samplerate=samplerate,
                        clip=self._clip,
                        bits_per_sample=self._bits_per_sample,
                        as_float=self._as_float,
                    )
                _emit("writing", stem=path.stem, path=str(path))
            except BaseException as exc:  # propagate after join
                self._error = exc
            finally:
                if item is not None:
                    self._queue.task_done()

    def submit(self, wav: np.ndarray, path: Path, samplerate: int) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((wav, path, samplerate))

    def close(self) -> None:
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._error is not None:
            raise self._error


def _list_models() -> int:
    for name in sorted(MLX_MODEL_REGISTRY.keys()):
        desc = MLX_MODEL_REGISTRY[name].get("description", "")
        if desc:
            print(f"{name}\t{desc}")
        else:
            print(name)
    return 0


def _load_audio(path: Path, model):
    import mlx.core as mx
    import mlx_audio_io as mac

    audio_mx, sr = mac.load(str(path), sr=model.samplerate, dtype="float32")
    wav = audio_mx.T
    src_channels = wav.shape[0]
    tgt_channels = model.audio_channels
    if src_channels != tgt_channels:
        if tgt_channels == 1:
            wav = mx.mean(wav, axis=0, keepdims=True)
        elif src_channels == 1 and tgt_channels > 1:
            wav = mx.broadcast_to(wav, (tgt_channels, wav.shape[1]))
        elif src_channels > tgt_channels:
            wav = wav[:tgt_channels, :]
        else:
            raise ValueError(
                f"Audio has {src_channels} channels but model expects {tgt_channels}."
            )
    return wav


def _iter_prefetched_audio(
    tracks: tp.Sequence[str],
    model,
    *,
    prefetch: int,
) -> tp.Iterator[tuple[Path, tp.Any]]:
    if prefetch <= 0:
        for track in tracks:
            path = Path(track)
            yield path, _load_audio(path, model)
        return

    q: queue.Queue = queue.Queue(maxsize=max(1, int(prefetch)))
    done = threading.Event()
    paths = [Path(track) for track in tracks]

    def _producer() -> None:
        try:
            for path in paths:
                if done.is_set():
                    break
                try:
                    wav = _load_audio(path, model)
                except BaseException as exc:
                    q.put((path, None, exc))
                    break
                q.put((path, wav, None))
        finally:
            q.put(None)

    thread = threading.Thread(target=_producer, daemon=True, name="demucs-audio-prefetch")
    thread.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            path, wav, exc = item
            if exc is not None:
                raise exc
            assert wav is not None
            yield path, wav
    finally:
        done.set()
        thread.join()


def _separate_one(
    path: Path,
    wav,
    model,
    out_dir: Path,
    shifts: int,
    seed: tp.Optional[int],
    overlap: float,
    segment: tp.Optional[float],
    split: bool,
    batch_size: int,
    verbose: bool,
    writer: _AsyncWriter,
    fmt: str,
    track_name: tp.Optional[str],
) -> Path:
    import mlx.core as mx

    from .apply_mlx import apply_model

    stem_name = track_name if track_name else path.stem
    track_out = out_dir / stem_name
    track_out.mkdir(parents=True, exist_ok=True)
    ext = "mp3" if fmt == "mp3" else "wav"

    _emit("loading", track=str(path), out=str(track_out))
    if verbose and not _PROGRESS_JSON:
        print(f"Loading audio: {path}")

    mix = wav[None, ...]

    def _on_progress(done: int, total: int) -> None:
        pct = (100.0 * done / total) if total > 0 else 0.0
        _emit("separating", track=str(path), done=done, total=total, pct=round(pct, 2))

    if verbose and not _PROGRESS_JSON:
        print("Running MLX separation...")
    _emit("separating", track=str(path), done=0, total=0, pct=0.0)

    estimates = apply_model(
        model,
        mix,
        shifts=shifts,
        seed=seed,
        split=split,
        overlap=overlap,
        segment=segment,
        batch_size=batch_size,
        progress=verbose and not _PROGRESS_JSON,
        progress_callback=_on_progress if _PROGRESS_JSON else None,
    )
    mx.eval(estimates)

    stem_paths = [track_out / f"{s}.{ext}" for s in model.sources]
    stems = np.asarray(estimates[0])
    for stem_idx, stem_path in enumerate(stem_paths):
        stem = np.ascontiguousarray(stems[stem_idx], dtype=np.float32)
        writer.submit(stem, stem_path, samplerate=model.samplerate)
        if verbose and not _PROGRESS_JSON:
            print(f"Wrote: {stem_path}")
    return track_out


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    global _PROGRESS_JSON

    parser = argparse.ArgumentParser(
        prog="demucs-mlx",
        description="MLX-only Demucs stem separation",
    )
    parser.add_argument("tracks", nargs="*", help="Audio files to separate")
    parser.add_argument("-n", "--name", default="htdemucs", help="Model name")
    parser.add_argument("-o", "--out", default="separated", help="Output directory")
    parser.add_argument(
        "--track-name",
        default=None,
        help="Override output subfolder name (default: input stem)",
    )
    parser.add_argument("--segment", type=float, default=None, help="Segment length in seconds")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    parser.add_argument("--shifts", type=int, default=1, help="Number of random shifts")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible shifts",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument(
        "--write-workers",
        type=int,
        default=1,
        help="Number of concurrent audio writer threads",
    )
    parser.add_argument(
        "--prefetch-tracks",
        type=int,
        default=2,
        help="Number of prefetched decoded tracks",
    )
    parser.add_argument("--no-split", action="store_true", help="Disable chunked inference")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Write MP3 stems instead of WAV (via lameenc)",
    )
    parser.add_argument("--mp3-bitrate", type=int, default=128, help="MP3 bitrate kbps")
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit line-delimited JSON progress events on stdout",
    )

    args = parser.parse_args(argv)
    _PROGRESS_JSON = bool(args.progress_json)
    # MLX GPU streams are thread-local; never decode on a background thread when
    # driving the GUI worker (or any --progress-json run).
    if _PROGRESS_JSON:
        args.prefetch_tracks = 0

    if args.list_models:
        return _list_models()

    if not args.tracks:
        parser.print_help(sys.stderr)
        return 2
    if args.shifts < 0:
        raise SystemExit("--shifts must be >= 0")
    if not (0.0 <= float(args.overlap) < 1.0):
        raise SystemExit("--overlap must be in [0, 1)")
    if args.segment is not None and float(args.segment) <= 0:
        raise SystemExit("--segment must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.write_workers <= 0:
        raise SystemExit("--write-workers must be > 0")
    if args.prefetch_tracks < 0:
        raise SystemExit("--prefetch-tracks must be >= 0")
    if args.mp3_bitrate <= 0:
        raise SystemExit("--mp3-bitrate must be > 0")
    if args.track_name is not None and len(args.tracks) != 1:
        raise SystemExit("--track-name requires exactly one input track")

    if args.name not in MLX_MODEL_REGISTRY:
        known = ", ".join(sorted(MLX_MODEL_REGISTRY.keys()))
        raise SystemExit(f"Unknown model '{args.name}'. Available: {known}")

    fmt = "mp3" if args.mp3 else "wav"
    try:
        _emit("status", stage="loading_model", model=args.name)
        if args.verbose and not _PROGRESS_JSON:
            print(f"Loading MLX model: {args.name}")
        from .model_converter import get_mlx_model

        model = get_mlx_model(args.name)
        if hasattr(model, "eval"):
            model.eval()
        _emit("status", stage="model_ready", model=args.name, sources=list(model.sources))

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = _AsyncWriter(
            maxsize=max(8, args.write_workers * 4),
            workers=args.write_workers,
            fmt=fmt,
            mp3_bitrate=args.mp3_bitrate,
        )
        try:
            track_iter = _iter_prefetched_audio(
                args.tracks, model, prefetch=args.prefetch_tracks
            )
            if not _PROGRESS_JSON:
                track_iter = tqdm(
                    track_iter,
                    total=len(args.tracks),
                    desc="Tracks",
                    unit="track",
                )
            for path, wav in track_iter:
                track_out = _separate_one(
                    path,
                    wav,
                    model,
                    out_dir,
                    shifts=args.shifts,
                    seed=args.seed,
                    overlap=args.overlap,
                    segment=args.segment,
                    split=not args.no_split,
                    batch_size=args.batch_size,
                    verbose=args.verbose,
                    writer=writer,
                    fmt=fmt,
                    track_name=args.track_name,
                )
                _emit(
                    "done",
                    track=str(path),
                    out=str(track_out),
                    sources=list(model.sources),
                    format=fmt,
                )
        finally:
            writer.close()
    except BaseException as exc:
        _emit("error", message=str(exc))
        if _PROGRESS_JSON:
            return 1
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
