"""Line-delimited JSON progress events: the GUI contract (demucs-ui-app).

One JSON object per stdout line. The GUI greps for the substring ``"event"`` and
parses the line, so keep these names and fields stable. Changing them is a
breaking change for demucs-ui-app and the C worker (which emits the same stream).
"""

from __future__ import annotations

import json
import sys
import typing as tp

#: event name -> required fields (besides "event")
EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "status": ("stage",),  # stage: loading_model | model_ready (+ model, sources)
    "loading": ("track", "out"),
    "separating": ("track", "done", "total", "pct"),
    "writing": ("stem", "path"),
    "done": ("track", "out", "sources", "format"),
    "error": ("message",),
}

_enabled = False


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    return _enabled


def emit(event: str, **fields: tp.Any) -> None:
    if not _enabled:
        return
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def validate(line_obj: dict) -> list[str]:
    """Return a list of contract violations for one parsed event (empty = ok)."""
    problems: list[str] = []
    name = line_obj.get("event")
    if name not in EVENT_FIELDS:
        return [f"unknown event {name!r}"]
    for field in EVENT_FIELDS[name]:
        if field not in line_obj:
            problems.append(f"{name}: missing field {field!r}")
    return problems
