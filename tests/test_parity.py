"""Parity gates: worker-py is the reference; worker-c and the frozen binary must agree.

Thresholds come from a measured baseline (demucs-mlx 1.6.1, mlx 0.32.3, M-series, 12 s
synthetic clip, shifts=0 so the comparison is RNG-free): py vs c max |diff| = 1.8e-4
(~6 LSB of PCM16) and per-stem SI-SDR from 38 dB (near-silent stems, dominated by PCM16
rounding) to 87 dB. Gates leave headroom for normal fp noise but catch real regressions.
Re-measure with `make bench` when an upstream bump legitimately moves numbers, and
say so in the PR.
"""
from __future__ import annotations

import pytest

from worker_py import parity

MAX_ABS_PY_VS_C = 1e-3
MIN_SI_SDR_DB = 30.0
MAX_ABS_PY_VS_FROZEN = 5e-5  # same code, different packaging: should be ~identical

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def py_run(model_cache, clip_wav, tmp_path_factory):
    out = tmp_path_factory.mktemp("py")
    res = parity.run_worker(parity.PY_WORKER, clip_wav, out)
    return res, out


def test_py_event_stream_is_valid_gui_contract(py_run):
    res, _ = py_run
    assert parity.event_problems(res.events) == []


def test_py_progress_reaches_100_percent(py_run):
    res, _ = py_run
    seps = [e for e in res.events if e["event"] == "separating"]
    assert seps[-1]["pct"] == 100.0
    assert seps[-1]["done"] == seps[-1]["total"] > 0


def test_py_stdout_is_only_json_events(py_run):
    res, _ = py_run
    for line in res.stdout.splitlines():
        assert line.startswith("{") and '"event"' in line, f"stray stdout line: {line!r}"


def test_py_writes_all_stems_under_track_name(py_run):
    _, out = py_run
    stems = parity.load_stems(out, "clip")
    assert set(stems) == set(parity.SOURCES)
    assert all(a.shape[0] == 2 and a.shape[1] > 44100 * 10 for a in stems.values())


def test_py_shifts_progress_covers_all_passes(model_cache, clip_wav, tmp_path):
    res = parity.run_worker(parity.PY_WORKER, clip_wav, tmp_path, shifts=2, batch_size=1)
    seps = [e for e in res.events if e["event"] == "separating" and e["total"]]
    assert seps[-1]["done"] == seps[-1]["total"]
    assert seps[-1]["total"] >= 2 * 2  # >= 2 segments per pass, 2 passes
    assert parity.event_problems(res.events) == []


def test_py_mp3_output(model_cache, clip_wav, tmp_path):
    res = parity.run_worker(parity.PY_WORKER, clip_wav, tmp_path, extra=["--mp3"])
    done = [e for e in res.events if e["event"] == "done"][0]
    assert done["format"] == "mp3"
    for src in parity.SOURCES:
        f = tmp_path / "clip" / f"{src}.mp3"
        assert f.stat().st_size > 1000


def test_py_error_event_and_exit_code(model_cache, tmp_path):
    import subprocess

    proc = subprocess.run(
        [*parity.PY_WORKER, str(tmp_path / "missing.wav"), "-o", str(tmp_path), "--progress-json"],
        capture_output=True, text=True, env=parity.worker_env(), cwd=str(parity.ROOT),
    )
    assert proc.returncode == 1
    assert any('"event": "error"' in ln for ln in proc.stdout.splitlines())


def test_c_event_stream_is_valid_gui_contract(c_worker, clip_wav, tmp_path):
    res = parity.run_worker([str(c_worker)], clip_wav, tmp_path)
    assert parity.event_problems(res.events) == []


def test_c_matches_py(c_worker, py_run, clip_wav, tmp_path):
    _, py_out = py_run
    parity.run_worker([str(c_worker)], clip_wav, tmp_path / "c")
    q = parity.quality(parity.load_stems(py_out, "clip"), parity.load_stems(tmp_path / "c", "clip"))
    assert q["max_abs"] < MAX_ABS_PY_VS_C, q
    for stem, m in q["per_stem"].items():
        assert m["si_sdr_db"] > MIN_SI_SDR_DB, (stem, m)


def test_frozen_matches_py_and_contract(frozen_worker, py_run, clip_wav, tmp_path):
    res = parity.run_worker([str(frozen_worker)], clip_wav, tmp_path / "f")
    assert parity.event_problems(res.events) == []
    _, py_out = py_run
    q = parity.quality(parity.load_stems(py_out, "clip"), parity.load_stems(tmp_path / "f", "clip"))
    assert q["max_abs"] < MAX_ABS_PY_VS_FROZEN, q


# C worker vs the committed goldens (upstream-computed intermediates). Baseline on 1.6.1:
# conv encoders ~1e-6, transformer ~4e-4, final stems ~7e-4. ~10x headroom.
C_COMPARE_LIMITS = {
    "mag": 1e-3, "x_norm": 1e-4, "xt_norm": 1e-4,
    "transformer_x": 5e-3, "transformer_xt": 5e-3, "transformer_from_golden_enc": 5e-3,
    "stems": 7e-3,
}
C_COMPARE_DEFAULT_LIMIT = 1e-4  # tenc_*/enc_*


def test_c_compare_against_goldens(c_worker):
    import re
    import subprocess

    models = parity.MODELS_DIR
    goldens = parity.ROOT / "goldens" / "htdemucs_6s"
    needed = [
        models / "htdemucs_6s.safetensors", models / "htdemucs_6s.json",
        goldens / "goldens.safetensors", goldens / "mix.wav",
    ]
    if not all(p.is_file() for p in needed):
        pytest.skip("export weights + goldens first (make models-export goldens)")
    proc = subprocess.run(
        [str(c_worker.parent / "test_compare"), *map(str, needed)],
        capture_output=True, text=True, cwd=str(parity.ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    seen = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"test_compare (\S+) (?:max_abs=|max_abs x=)([0-9.eE+-]+)", line)
        if m:
            seen[m.group(1)] = float(m.group(2))
    assert "stems" in seen, proc.stdout
    for name, value in seen.items():
        assert value < C_COMPARE_LIMITS.get(name, C_COMPARE_DEFAULT_LIMIT), (name, value)
