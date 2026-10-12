"""worker-c contract: detect upstream/mlx drift before it ships.

Cheap checks run everywhere. ``test_export_matches_committed_manifest`` needs the
weights in models/mlx-cache and is skipped without them.
"""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from worker_py import contract


def test_manifest_exists_and_is_current_schema():
    manifest = contract.load_manifest()
    assert manifest is not None, "run: make models-export"
    assert manifest["schema"] == 1
    assert manifest["model"] == "htdemucs_6s"


def test_manifest_records_installed_upstream():
    manifest = contract.load_manifest()
    assert manifest["upstream"]["version"] == contract.upstream_version(), (
        "demucs-mlx was bumped: re-run `make models-export` and commit the diff "
        "(models/export_manifest.json, keys.txt) after `make verify` passes"
    )


def test_pyproject_pins_upstream_exactly():
    text = (contract.ROOT / "pyproject.toml").read_text()
    pins = re.findall(r'"demucs-mlx==([^"]+)"', text)
    assert pins, "demucs-mlx must be pinned with == in pyproject.toml"
    assert set(pins) == {contract.upstream_version()}


def test_committed_artifacts_match_manifest():
    manifest = contract.load_manifest()
    exp = manifest["export"]
    assert hashlib.sha256(contract.KEYS_PATH.read_bytes()).hexdigest() == exp["keys_sha256"]
    assert hashlib.sha256(contract.CONFIG_PATH.read_bytes()).hexdigest() == exp["config_sha256"]
    assert len(contract.tensor_lines(contract.KEYS_PATH)) == exp["num_tensors"]


def test_keys_manifest_has_no_upstream_prefix():
    for line in contract.tensor_lines(contract.KEYS_PATH):
        assert not line.startswith(contract.UPSTREAM_PREFIX)


def test_c_worker_defaults_match_upstream():
    c = contract.c_worker_defaults()
    assert c == contract.upstream_defaults()
    assert c == contract.load_manifest()["defaults"]


def _decode(value):
    """Upstream encodes Fractions as {"__type__": "fraction", ...}."""
    if isinstance(value, dict) and value.get("__type__") == "fraction":
        return value["numerator"] / value["denominator"]
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def test_upstream_model_kwargs_match_our_config(model_cache):
    """Every constructor kwarg upstream publishes must equal what worker-c was built for."""
    upstream = _decode(json.loads((model_cache / "htdemucs_6s_config.json").read_text()))
    assert upstream["num_models"] == 1
    assert upstream["per_model_classes"] == ["HTDemucsMLX"]
    ours = json.loads(contract.CONFIG_PATH.read_text())["kwargs"]
    for key, value in upstream["per_model_kwargs"][0].items():
        assert ours[key] == value, f"config kwarg {key!r}: upstream {value!r} vs ours {ours[key]!r}"


def test_mlx_c_tag_matches_manifest_and_mlx_minor():
    manifest = contract.load_manifest()
    assert contract.cmake_mlx_c_tag() == manifest["mlx_c"]["tag"], (
        "mlx-c tag changed in worker-c/CMakeLists.txt; re-run `make models-export`"
    )
    embedded = manifest["mlx_c"]["embedded_mlx"]
    assert embedded, "unknown mlx version embedded by mlx-c; build worker-c then export"
    built = contract.built_mlx_version()
    if built is not None:
        assert built == embedded
    # mlx-c bundles its own MLX. Python MLX and C MLX must share major.minor.
    assert embedded.split(".")[:2] == contract.mlx_version().split(".")[:2], (
        f"python mlx {contract.mlx_version()} vs mlx-c embedded mlx {embedded}"
    )


@pytest.mark.model
def test_export_matches_committed_manifest(model_cache, tmp_path):
    from worker_py.c_export import export_model

    res = export_model("htdemucs_6s", tmp_path)
    assert contract.check_against_committed(res) == []
    assert contract.sha256_file(res["cache_file"]) == contract.load_manifest()[
        "source_cache_sha256"
    ]
