from __future__ import annotations

from worker_py import cache


def test_no_env_keeps_upstream_default():
    assert cache.resolve_cache_dir({}) is None


def test_upstream_env_wins(tmp_path):
    env = {"DEMUCS_MLX_CACHE_DIR": str(tmp_path / "a"), "DEMUCS_MLX_CACHE": str(tmp_path / "b")}
    assert cache.resolve_cache_dir(env) == tmp_path / "a"


def test_legacy_env_used_as_is(tmp_path):
    assert cache.resolve_cache_dir({"DEMUCS_MLX_CACHE": str(tmp_path)}) == tmp_path


def test_c_export_dir_is_not_clobbered(tmp_path):
    (tmp_path / "htdemucs_6s.safetensors").write_bytes(b"x")
    (tmp_path / "htdemucs_6s.json").write_text("{}")
    assert cache.has_c_export(tmp_path)
    assert cache.resolve_cache_dir({"DEMUCS_MLX_CACHE": str(tmp_path)}) == tmp_path / "mlx"


def test_upstream_cache_dir_is_not_a_c_export(tmp_path):
    (tmp_path / "htdemucs_6s.safetensors").write_bytes(b"x")
    (tmp_path / "htdemucs_6s_config.json").write_text("{}")
    assert not cache.has_c_export(tmp_path)


def test_apply_cache_env_sets_upstream_var(tmp_path):
    env = {"DEMUCS_MLX_CACHE": str(tmp_path)}
    cache.apply_cache_env(env)
    assert env["DEMUCS_MLX_CACHE_DIR"] == str(tmp_path)
