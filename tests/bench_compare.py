#!/usr/bin/env python3
"""Benchmark harness for the Python MLX worker (the C-worker high bar)."""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _worker_dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total / (1024.0 * 1024.0)


def _make_clip(path: Path, seconds: float, samplerate: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(samplerate * seconds)
    audio = rng.standard_normal((2, n), dtype=np.float32) * 0.1
    from demucs_mlx.c_export import write_wav_pcm16

    write_wav_pcm16(path, audio, samplerate)


def _run_cli(
    tracks: list[Path],
    out_dir: Path,
    *,
    name: str,
    seed: int,
    overlap: float,
    batch_size: int,
    extra: list[str],
    frozen_bin: Path | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str]
    if frozen_bin is not None:
        cmd = [str(frozen_bin)]
    else:
        cmd = [sys.executable, "-m", "demucs_mlx"]
    cmd += [
        "-n",
        name,
        "-o",
        str(out_dir),
        "--seed",
        str(seed),
        "--overlap",
        str(overlap),
        "-b",
        str(batch_size),
        "--shifts",
        "1",
        "--progress-json",
        *extra,
        *[str(t) for t in tracks],
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed ({proc.returncode}): {proc.stderr[-2000:]}\n{proc.stdout[-1000:]}"
        )
    return {"wall_s": wall, "cmd": cmd, "stdout": proc.stdout, "stderr": proc.stderr}


def _load_stems(out_dir: Path, track_stem: str) -> dict[str, np.ndarray]:
    from demucs_mlx.c_export import read_wav_float

    track_dir = out_dir / track_stem
    stems = {}
    for src in SOURCES:
        wav_path = track_dir / f"{src}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        audio, _sr = read_wav_float(wav_path)
        stems[src] = audio
    return stems


def _quality(ref: dict[str, np.ndarray], est: dict[str, np.ndarray]) -> dict:
    from demucs_mlx.c_export import si_sdr

    per: dict[str, dict[str, float]] = {}
    max_abs = 0.0
    for src in SOURCES:
        a = ref[src]
        b = est[src]
        n = min(a.shape[-1], b.shape[-1])
        a = a[..., :n]
        b = b[..., :n]
        diff = np.abs(a - b)
        src_max = float(diff.max()) if diff.size else 0.0
        rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
        per[src] = {"max_abs": src_max, "rmse": rmse, "si_sdr_db": si_sdr(a, b)}
        max_abs = max(max_abs, src_max)
    return {"per_stem": per, "max_abs": max_abs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--name", default="htdemucs_6s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--durations",
        default="10",
        help="Comma-separated clip lengths in seconds (default: 10)",
    )
    parser.add_argument(
        "--frozen",
        default="",
        help="Optional path to frozen demucs_mlx_worker binary",
    )
    parser.add_argument(
        "--c-worker",
        default="",
        help="Optional path to C worker binary to compare against Python",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Pass --compile to workers (Python mx.compile / C mlx_compile)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(ROOT / "goldens" / "bench"),
        help="Output JSON + clip directory",
    )
    args = parser.parse_args(argv)

    from demucs_mlx.model_converter import get_mlx_model

    model = get_mlx_model(args.name)
    sr = int(model.samplerate)
    durations = [float(x) for x in str(args.durations).split(",") if x.strip()]
    out_root = Path(args.out)
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    frozen_bin = Path(args.frozen) if args.frozen else None
    if frozen_bin is not None and not frozen_bin.is_file():
        raise SystemExit(f"frozen worker not found: {frozen_bin}")
    c_bin = Path(args.c_worker) if args.c_worker else None
    if c_bin is not None and not c_bin.is_file():
        raise SystemExit(f"C worker not found: {c_bin}")

    results: dict = {
        "model": args.name,
        "samplerate": sr,
        "seed": args.seed,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "mlx_version": None,
        "python_rss_mb": None,
        "runs": [],
    }
    try:
        import mlx

        results["mlx_version"] = getattr(mlx, "__version__", None)
    except ImportError:
        pass

    for dur in durations:
        clip = clips_dir / f"clip_{int(dur)}s.wav"
        _make_clip(clip, dur, sr, seed=args.seed + int(dur))
        py_out = out_root / f"py_{int(dur)}s"
        if py_out.exists():
            shutil.rmtree(py_out)

        walls: list[float] = []
        for i in range(args.warmup + args.iters):
            run_out = py_out if i == args.warmup + args.iters - 1 else out_root / f"tmp_{int(dur)}s"
            if run_out.exists():
                shutil.rmtree(run_out)
            info = _run_cli(
                [clip],
                run_out,
                name=args.name,
                seed=args.seed,
                overlap=args.overlap,
                batch_size=args.batch_size,
                extra=["--compile"] if args.compile else [],
                frozen_bin=None,
            )
            if i >= args.warmup:
                walls.append(float(info["wall_s"]))

        walls.sort()
        med = statistics.median(walls)
        py_stems = _load_stems(py_out, clip.stem)
        entry: dict = {
            "duration_s": dur,
            "python_cli": {
                "walls_s": walls,
                "median_s": med,
                "rt_factor": dur / med if med > 0 else None,
                "cold_s": walls[0] if walls else None,
            },
        }

        if frozen_bin is not None:
            fr_out = out_root / f"frozen_{int(dur)}s"
            if fr_out.exists():
                shutil.rmtree(fr_out)
            t0 = time.perf_counter()
            _run_cli(
                [clip],
                fr_out,
                name=args.name,
                seed=args.seed,
                overlap=args.overlap,
                batch_size=args.batch_size,
                extra=[],
                frozen_bin=frozen_bin,
            )
            fr_wall = time.perf_counter() - t0
            fr_stems = _load_stems(fr_out, clip.stem)
            entry["frozen"] = {
                "wall_s": fr_wall,
                "rt_factor": dur / fr_wall if fr_wall > 0 else None,
                "quality_vs_python": _quality(py_stems, fr_stems),
                "on_disk_mb": _worker_dir_size_mb(frozen_bin.parent),
            }

        if c_bin is not None:
            c_walls: list[float] = []
            c_out = out_root / f"c_{int(dur)}s"
            for i in range(args.warmup + args.iters):
                run_out = c_out if i == args.warmup + args.iters - 1 else out_root / f"tmp_c_{int(dur)}s"
                if run_out.exists():
                    shutil.rmtree(run_out)
                t0 = time.perf_counter()
                _run_cli(
                    [clip],
                    run_out,
                    name=args.name,
                    seed=args.seed,
                    overlap=args.overlap,
                    batch_size=args.batch_size,
                    extra=["--compile"] if args.compile else [],
                    frozen_bin=c_bin,
                )
                c_wall = time.perf_counter() - t0
                if i >= args.warmup:
                    c_walls.append(c_wall)
            c_walls.sort()
            c_med = statistics.median(c_walls)
            c_stems = _load_stems(c_out, clip.stem)
            entry["c_worker"] = {
                "walls_s": c_walls,
                "median_s": c_med,
                "cold_s": c_walls[0] if c_walls else None,
                "rt_factor": dur / c_med if c_med > 0 else None,
                "quality_vs_python": _quality(py_stems, c_stems),
                "on_disk_mb": _worker_dir_size_mb(c_bin),
            }

        results["runs"].append(entry)
        extra = ""
        if "c_worker" in entry and entry["c_worker"].get("median_s"):
            cmed = float(entry["c_worker"]["median_s"])
            extra = f"  c median {cmed:.3f}s ({dur / cmed:.1f}x RT)"
        print(
            f"{dur:.0f}s audio: python median {med:.3f}s ({dur / med:.1f}x RT){extra}"
            if med
            else f"{dur:.0f}s audio: no timings"
        )

    results["python_rss_mb"] = _peak_rss_mb()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bench.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_root / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
