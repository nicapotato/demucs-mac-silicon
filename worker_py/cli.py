"""demucs-mlx-worker: GUI-facing CLI, a thin wrapper over upstream ``demucs_mlx``.

Owns only what the GUI needs and upstream does not provide:

* ``--progress-json`` line events (see :mod:`worker_py.events`)
* ``--mp3`` / ``--mp3-bitrate`` output
* ``--track-name`` (output subfolder override)
* ``DEMUCS_MLX_CACHE`` -> upstream ``DEMUCS_MLX_CACHE_DIR`` mapping

All inference goes through upstream's public ``Separator`` API.
"""

from __future__ import annotations

import argparse
import os
import sys
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import events
from .cache import apply_cache_env


def _batch_size(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-size must be an integer or 'auto'") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be > 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demucs-mlx-worker",
        description="Demucs stem separation worker (wrapper over demucs-mlx)",
    )
    p.add_argument("tracks", nargs="*", help="Audio files to separate")
    p.add_argument("-n", "--name", default="htdemucs_6s", help="Model name")
    p.add_argument("-o", "--out", default="separated", help="Output directory")
    p.add_argument("--track-name", default=None, help="Output subfolder (default: input stem)")
    p.add_argument("--segment", type=float, default=None, help="Segment length in seconds")
    p.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    p.add_argument("--shifts", type=int, default=1, help="Number of random shifts")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for shifts")
    p.add_argument("-b", "--batch-size", type=_batch_size, default="auto")
    p.add_argument("--no-split", action="store_true", help="Disable chunked inference")
    p.add_argument("--attention", choices=["fp32", "fp16"], default=None)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--write-workers", type=int, default=1, help="Concurrent stem writers")
    p.add_argument(
        "--prefetch-tracks",
        type=int,
        default=0,
        help="Accepted for GUI compatibility; the worker decodes serially",
    )
    p.add_argument("--mp3", action="store_true", help="Write MP3 stems instead of WAV")
    p.add_argument("--mp3-bitrate", type=int, default=128, help="MP3 bitrate (kbps)")
    p.add_argument("--progress-json", action="store_true", help="Emit JSON events on stdout")
    p.add_argument("--offline", action="store_true", help="Never download model weights")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _validate(args: argparse.Namespace) -> None:
    if args.shifts < 0:
        raise SystemExit("--shifts must be >= 0")
    if not (0.0 <= args.overlap < 1.0):
        raise SystemExit("--overlap must be in [0, 1)")
    if args.segment is not None and args.segment <= 0:
        raise SystemExit("--segment must be > 0")
    if args.write_workers <= 0:
        raise SystemExit("--write-workers must be > 0")
    if args.mp3_bitrate <= 0:
        raise SystemExit("--mp3-bitrate must be > 0")
    if args.track_name is not None and len(args.tracks) != 1:
        raise SystemExit("--track-name requires exactly one input track")


def _write_stems(
    stems: dict[str, tp.Any],
    track_out: Path,
    *,
    samplerate: int,
    mp3: bool,
    bitrate: int,
    workers: int,
) -> None:
    import numpy as np
    from demucs_mlx import save_audio

    from .mp3 import save_mp3

    ext = "mp3" if mp3 else "wav"

    def write(name: str) -> Path:
        path = track_out / f"{name}.{ext}"
        arr = np.ascontiguousarray(stems[name], dtype=np.float32)
        if mp3:
            save_mp3(arr, path, samplerate, bitrate=bitrate)
        else:
            save_audio(arr, path, samplerate=samplerate)
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(name, pool.submit(write, name)) for name in stems]
        for name, fut in futures:
            events.emit("writing", stem=name, path=str(fut.result()))


def _separate_track(
    sep: tp.Any,
    path: Path,
    args: argparse.Namespace,
    router: tp.Any = None,
) -> Path:
    from . import progress

    name = args.track_name or path.stem
    track_out = Path(args.out) / name
    track_out.mkdir(parents=True, exist_ok=True)
    events.emit("loading", track=str(path), out=str(track_out))
    events.emit("separating", track=str(path), done=0, total=0, pct=0.0)

    last_pct = [0.0]

    def on_progress(done: int, total: int) -> None:
        # With shifts the total is an estimate that can shift by a segment: never
        # let the bar go backwards.
        pct = max(last_pct[0], (100.0 * done / total) if total > 0 else 0.0)
        last_pct[0] = pct
        events.emit("separating", track=str(path), done=done, total=total, pct=round(pct, 2))

    expected_bars = max(args.shifts, 1) * len(getattr(sep.model, "models", [sep.model]))
    if router is not None:
        with router.track(expected_bars, on_progress):
            _, stems = sep.separate_audio_file(path)
    elif events.enabled():
        with progress.capture_progress(expected_bars, on_progress):
            _, stems = sep.separate_audio_file(path)
    else:
        _, stems = sep.separate_audio_file(path)

    _write_stems(
        stems,
        track_out,
        samplerate=sep.samplerate,
        mp3=args.mp3,
        bitrate=args.mp3_bitrate,
        workers=args.write_workers,
    )
    events.emit(
        "done",
        track=str(path),
        out=str(track_out),
        sources=list(stems),
        format="mp3" if args.mp3 else "wav",
    )
    return track_out


def _run(args: argparse.Namespace) -> int:
    from demucs_mlx import Separator

    from . import progress

    # Prefer upstream's native callback when this demucs-mlx has one; otherwise
    # fall back to the runtime shim in worker_py.progress.
    native = events.enabled() and progress.supports_native_callback()
    router = progress.NativeRouter() if native else None
    events.emit("status", stage="loading_model", model=args.name)
    kwargs: dict[str, tp.Any] = dict(
        model=args.name,
        shifts=args.shifts,
        overlap=args.overlap,
        split=not args.no_split,
        segment=args.segment,
        seed=args.seed,
        batch_size=args.batch_size,
        compile=args.compile,
        attention_precision=args.attention,
        progress=False,
    )
    if router is not None:
        kwargs["progress_callback"] = router
    sep = Separator(**kwargs)
    events.emit("status", stage="model_ready", model=args.name, sources=list(sep.model.sources))
    for track in args.tracks:
        _separate_track(sep, Path(track), args, router)
    return 0


def main(argv: tp.Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events.set_enabled(args.progress_json)

    if args.list_models:
        from demucs_mlx import list_models

        for group, names in list_models().items():
            for name in sorted(names):
                print(f"{name}\t{group}")
        return 0
    if not args.tracks:
        build_parser().print_help(sys.stderr)
        return 2
    _validate(args)

    if args.offline:
        os.environ["DEMUCS_MLX_NO_DOWNLOAD"] = "1"
    apply_cache_env()

    try:
        return _run(args)
    except BaseException as exc:  # noqa: BLE001 - report everything to the GUI
        events.emit("error", message=str(exc))
        if events.enabled() and isinstance(exc, Exception):
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
