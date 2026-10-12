"""Pins the runtime progress shim against the installed upstream ``apply_model``.

If this fails after an upstream bump, either upstream changed how it creates its tqdm
bar / recurses per shift (fix worker_py/progress.py), or it shipped a native
``progress_callback`` (then delete the shim; see upstream-patches/README.md).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from demucs_mlx import apply_mlx

from worker_py import progress


class _IdentityModel:
    samplerate = 44_100
    segment = 1.0
    sources = ["mix"]

    def valid_length(self, length: int) -> int:
        return int(length)

    def __call__(self, x):
        return x[:, None, :, :]


def _mix(seconds: float):
    return mx.array(np.zeros((1, 2, int(seconds * 44_100)), dtype=np.float32))


def _run(shifts: int, expected_bars: int, seconds: float = 3.0):
    events: list[tuple[int, int]] = []
    with progress.capture_progress(expected_bars, lambda d, t: events.append((d, t))):
        apply_mlx.apply_model(  # looked up at call time, like Separator does
            _IdentityModel(), _mix(seconds), shifts=shifts, seed=0, split=True,
            overlap=0.0, batch_size=1,
        )
    return events


@pytest.mark.parametrize("shifts, bars", [(0, 1), (1, 1), (2, 2)])
def test_reports_every_segment_even_with_shifts(shifts, bars):
    events = _run(shifts, bars)
    assert events, "no progress observed: upstream changed its progress plumbing"
    done, total = events[-1]
    assert done == total, "progress must finish at 100%"
    assert total >= 3 * bars, "every pass contributes its segments"
    dones = [d for d, _ in events]
    assert dones == sorted(dones), "progress must be monotonic"


def test_patches_are_restored():
    import tqdm
    before = (tqdm.tqdm, apply_mlx.apply_model)
    _run(1, 1)
    assert (tqdm.tqdm, apply_mlx.apply_model) == before


def test_patches_restored_on_error():
    import tqdm

    before = tqdm.tqdm
    with pytest.raises(RuntimeError), progress.capture_progress(1, lambda *_: None):
        raise RuntimeError("boom")
    assert tqdm.tqdm is before


def test_aggregator_is_monotonic_across_bars():
    seen: list[tuple[int, int]] = []
    agg = progress.Aggregator(2, lambda d, t: seen.append((d, t)))
    agg.start(4)
    agg.update(4)
    agg.close()
    agg.start(4)
    agg.update(2)
    agg.update(2)
    agg.close()
    assert seen[0] == (0, 8)
    assert seen[-1] == (8, 8)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_native_router_folds_per_pass_updates():
    seen: list[tuple[int, int]] = []
    router = progress.NativeRouter()
    router(0, 5)  # no destination: ignored
    with router.track(2, lambda d, t: seen.append((d, t))):
        for _ in range(2):
            router(0, 3)
            router(1, 3)
            router(3, 3)
    assert seen[-1] == (6, 6)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
