"""Run two workers on the same clip and compare their stems (the parity gate).

Used by ``tests/test_parity.py`` (pass/fail) and ``tests/bench_compare.py`` (timings).
worker-py is the reference; worker-c and the frozen binary are candidates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .c_export import read_wav_float, si_sdr
from .events import validate

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")
MODELS_DIR = ROOT / "models"
MLX_CACHE = MODELS_DIR / "mlx-cache"

PY_WORKER = [sys.executable, "-m", "worker_py"]


@dataclass
class RunResult:
    wall_s: float
    events: list[dict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


def worker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["DEMUCS_MLX_CACHE_DIR"] = str(MLX_CACHE)  # python workers (upstream cache)
    env["DEMUCS_MLX_CACHE"] = str(MODELS_DIR)  # C worker (its export dir)
    env["DEMUCS_MLX_NO_DOWNLOAD"] = "1"
    env.update(extra or {})
    return env


def run_worker(
    cmd: list[str],
    clip: Path,
    out_dir: Path,
    *,
    track_name: str = "clip",
    seed: int = 0,
    overlap: float = 0.25,
    batch_size: int = 8,
    shifts: int = 0,
    extra: list[str] | None = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = [
        *cmd,
        str(clip),
        "-n",
        "htdemucs_6s",
        "-o",
        str(out_dir),
        "--track-name",
        track_name,
        "--seed",
        str(seed),
        "--overlap",
        str(overlap),
        "-b",
        str(batch_size),
        "--shifts",
        str(shifts),
        "--progress-json",
        "--prefetch-tracks",
        "0",
        *(extra or []),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(full, cwd=str(ROOT), capture_output=True, text=True, env=worker_env())
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed ({proc.returncode}): {full}\n"
            f"stderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-1000:]}"
        )
    return RunResult(wall, parse_events(proc.stdout), proc.stdout, proc.stderr)


def parse_events(stdout: str) -> list[dict]:
    out = []
    for line in stdout.splitlines():
        if '"event"' in line:
            out.append(json.loads(line))
    return out


def event_problems(events: list[dict]) -> list[str]:
    """GUI-contract violations in an event stream (shared by worker-py and worker-c)."""
    problems: list[str] = []
    for ev in events:
        problems += validate(ev)
    names = [e["event"] for e in events]
    for needed in ("status", "loading", "separating", "writing", "done"):
        if needed not in names:
            problems.append(f"missing event {needed!r}")
    if "done" in names and names[-1] != "done":
        problems.append("done must be the last event")
    if "status" in names:
        stages = [e["stage"] for e in events if e["event"] == "status"]
        if stages[:2] != ["loading_model", "model_ready"]:
            problems.append(f"status stages out of order: {stages}")
    pcts = [e["pct"] for e in events if e["event"] == "separating"]
    if pcts != sorted(pcts):
        problems.append("separating pct is not monotonic")
    done = [e for e in events if e["event"] == "done"]
    if done and list(done[0]["sources"]) != list(SOURCES):
        problems.append(f"unexpected sources {done[0]['sources']}")
    written = [e["stem"] for e in events if e["event"] == "writing"]
    if sorted(written) != sorted(SOURCES):
        problems.append(f"writing events {written} != sources")
    return problems


def load_stems(out_dir: Path, track: str, ext: str = "wav") -> dict[str, np.ndarray]:
    stems = {}
    for src in SOURCES:
        path = out_dir / track / f"{src}.{ext}"
        if not path.exists():
            raise FileNotFoundError(path)
        stems[src], _sr = read_wav_float(path)
    return stems


def quality(ref: dict[str, np.ndarray], est: dict[str, np.ndarray]) -> dict:
    per: dict[str, dict[str, float]] = {}
    max_abs = 0.0
    for src in SOURCES:
        n = min(ref[src].shape[-1], est[src].shape[-1])
        a, b = ref[src][..., :n], est[src][..., :n]
        diff = np.abs(a - b)
        src_max = float(diff.max()) if diff.size else 0.0
        rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
        per[src] = {"max_abs": src_max, "rmse": rmse, "si_sdr_db": si_sdr(a, b)}
        max_abs = max(max_abs, src_max)
    return {"per_stem": per, "max_abs": max_abs}
