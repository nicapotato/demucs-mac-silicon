# Python API

`worker_py` adds no Python API of its own beyond the CLI (`worker_py.cli.main`) and the helper
modules listed in [development.md](development.md). Programmatic use is **upstream's API**,
pinned in `pyproject.toml`:

```python
from demucs_mlx import Separator, save_audio, list_models

separator = Separator(model="htdemucs_6s")
origin, stems = separator.separate_audio_file("song.wav")
for name, audio in stems.items():
    save_audio(audio, f"{name}.wav", samplerate=separator.samplerate)
```

Reference: https://github.com/ssmall256/demucs-mlx#python-usage (match the version in
`pyproject.toml`; the defaults differ between releases, for example batch size is `auto` and
attention is fp16 by default since 1.5/1.6).

## Worker CLI contract

See the README for flags and `worker_py/events.py` for the JSON event schema. These are the
interface demucs-ui-app depends on; `tests/test_parity.py` enforces them for worker-py,
worker-c and the frozen binary.
