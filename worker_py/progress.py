"""Segment progress for upstream ``apply_model`` without forking it.

Upstream demucs-mlx 1.6.x has no progress callback. With ``progress=True`` its
``apply_model`` does ``from tqdm import tqdm`` *at call time* and calls
``bar.update(n)`` per batch of segments. Two small, temporary patches:

1. ``tqdm.tqdm`` becomes a shim that forwards bar updates to our callback.
2. ``apply_mlx.apply_model`` is wrapped so the *leaf* calls (``shifts == 0``) run
   with ``progress=True``. Upstream forces ``progress=False`` on the per-shift
   recursion, so with the default ``shifts=1`` no bar would exist at all.

Both are restored on exit. One bar appears per (shift x sub-model) leaf call.

This leans on an upstream internal, so ``tests/test_progress_shim.py`` pins the
behavior against the installed version. Once upstream ships a real callback (see
``upstream-patches/``), ``supports_native_callback()`` flips to True and this
module is no longer used.
"""

from __future__ import annotations

import contextlib
import inspect
import typing as tp

#: callback(done_segments, total_segments) -- totals cover every shift/model bar
ProgressCallback = tp.Callable[[int, int], None]


class Aggregator:
    """Fold per-bar (done, total) updates into one monotonic (done, total)."""

    def __init__(self, expected_bars: int, callback: ProgressCallback):
        self.expected_bars = max(1, int(expected_bars))
        self.callback = callback
        self.bars_started = 0
        self.finished_total = 0  # sum of totals of closed bars
        self.cur_total = 0
        self.cur_done = 0
        self.last: tuple[int, int] = (0, 0)

    def start(self, total: int) -> None:
        if self.cur_total:  # previous bar was never closed explicitly
            self.finished_total += self.cur_total
        self.bars_started += 1
        self.cur_total = max(0, int(total or 0))
        self.cur_done = 0
        self._report()

    def update(self, n: int = 1) -> None:
        self.cur_done = min(self.cur_total, self.cur_done + int(n))
        self._report()

    def close(self) -> None:
        if self.cur_total:
            self.cur_done = self.cur_total
            self._report()
            self.finished_total += self.cur_total
            self.cur_total = 0
            self.cur_done = 0

    def _report(self) -> None:
        remaining_bars = max(0, self.expected_bars - self.bars_started)
        total = self.finished_total + self.cur_total * (1 + remaining_bars)
        done = self.finished_total + self.cur_done
        pair = (min(done, total), total)
        if pair != self.last:
            self.last = pair
            self.callback(*pair)


class _Bar:
    """Minimal tqdm stand-in: only what apply_model touches."""

    def __init__(self, agg: Aggregator, total: int | None = None, **_kwargs: tp.Any):
        self._agg = agg
        self._closed = False
        agg.start(total or 0)

    def update(self, n: int = 1) -> None:
        self._agg.update(n)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._agg.close()

    def set_description(self, *_a: tp.Any, **_k: tp.Any) -> None:
        pass

    def set_postfix(self, *_a: tp.Any, **_k: tp.Any) -> None:
        pass

    def __enter__(self) -> _Bar:
        return self

    def __exit__(self, *_exc: tp.Any) -> None:
        self.close()


@contextlib.contextmanager
def capture_progress(expected_bars: int, callback: ProgressCallback) -> tp.Iterator[Aggregator]:
    """Route ``apply_model(progress=True)`` bar updates to ``callback``."""
    import tqdm as tqdm_module

    agg = Aggregator(expected_bars, callback)
    original = tqdm_module.tqdm

    def factory(*args: tp.Any, **kwargs: tp.Any) -> _Bar:
        total = kwargs.get("total", args[1] if len(args) > 1 else None)
        return _Bar(agg, total=total)

    from demucs_mlx import apply_mlx

    original_apply = apply_mlx.apply_model
    signature = inspect.signature(original_apply)

    def apply_with_leaf_progress(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if int(bound.arguments.get("shifts") or 0) == 0:
            bound.arguments["progress"] = True
        return original_apply(*bound.args, **bound.kwargs)

    tqdm_module.tqdm = factory  # type: ignore[assignment]
    apply_mlx.apply_model = apply_with_leaf_progress  # type: ignore[assignment]
    try:
        yield agg
    finally:
        apply_mlx.apply_model = original_apply  # type: ignore[assignment]
        tqdm_module.tqdm = original  # type: ignore[assignment]


class NativeRouter:
    """Adapter for upstream's native ``progress_callback(done, total)`` (per pass).

    Passed once as ``Separator(progress_callback=router)``; :meth:`track` selects the
    per-track destination and folds per-pass updates into one monotonic stream.
    """

    def __init__(self) -> None:
        self._agg: Aggregator | None = None

    def __call__(self, done: int, total: int) -> None:
        agg = self._agg
        if agg is None:
            return
        if done == 0:
            agg.start(total)
            return
        agg.update(done - agg.cur_done)
        if done >= total:
            agg.close()

    @contextlib.contextmanager
    def track(self, expected_bars: int, callback: ProgressCallback) -> tp.Iterator[Aggregator]:
        self._agg = Aggregator(expected_bars, callback)
        try:
            yield self._agg
        finally:
            self._agg = None


def supports_native_callback() -> bool:
    """True once upstream ``Separator`` accepts ``progress_callback``."""
    from demucs_mlx import Separator

    return "progress_callback" in inspect.signature(Separator.__init__).parameters
