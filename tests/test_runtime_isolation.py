from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from research_engine import extractor, grounding, logged_search, paths, telemetry_observer


def _clear_legacy_path_overrides(monkeypatch) -> None:
    for module, name in (
        (logged_search, "CALL_LOG"),
        (extractor, "CACHE_DIR"),
        (extractor, "READER_TELEMETRY_LOG"),
        (extractor, "BLOCKED_LOG_PATH"),
        (telemetry_observer, "MASTER_LOG"),
        (telemetry_observer, "CALL_LOG"),
        (grounding, "TELEMETRY_PATH"),
    ):
        if name in vars(module):
            monkeypatch.delattr(module, name)


def test_env_vars_point_at_tmp_under_pytest() -> None:
    production_data = Path.home() / ".research_engine"
    production_cache = paths.package_path("cache")

    for env_var, production_path in (
        (paths.DATA_DIR_ENV, production_data),
        (paths.CACHE_DIR_ENV, production_cache),
    ):
        resolved = Path(os.environ[env_var]).resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        assert resolved.is_relative_to(temp_root)
        assert resolved != production_path.resolve()


def test_log_paths_follow_env_at_call_time(tmp_path: Path, monkeypatch) -> None:
    _clear_legacy_path_overrides(monkeypatch)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))

    assert logged_search.call_log_path().is_relative_to(tmp_path)
    assert extractor.reader_telemetry_path().is_relative_to(tmp_path)
    assert extractor.blocked_log_path().is_relative_to(tmp_path)
    assert telemetry_observer.master_log_path().is_relative_to(tmp_path)
    assert telemetry_observer.call_log_path().is_relative_to(tmp_path)
    assert grounding.telemetry_path().is_relative_to(tmp_path)


def test_cache_dir_follows_env_at_call_time(tmp_path: Path, monkeypatch) -> None:
    _clear_legacy_path_overrides(monkeypatch)
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(tmp_path))
    assert extractor.cache_dir() == tmp_path
    assert paths.cache_dir() == tmp_path

    monkeypatch.delenv(paths.CACHE_DIR_ENV)
    assert paths.cache_dir() == paths.package_path("cache")


def test_append_call_writes_under_env_dir(tmp_path: Path, monkeypatch) -> None:
    _clear_legacy_path_overrides(monkeypatch)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))

    logged_search._append_call({"lane": "test", "ok": True})

    assert (tmp_path / "research-call-log.jsonl").exists()


def test_no_module_level_resolved_paths() -> None:
    names = (
        "CACHE_DIR",
        "READER_TELEMETRY_LOG",
        "BLOCKED_LOG_PATH",
        "CALL_LOG",
        "MASTER_LOG",
        "TELEMETRY_PATH",
    )
    pattern = re.compile(rf"^({'|'.join(names)})\\s*=")
    modules = (logged_search, extractor, telemetry_observer, grounding)

    for module in modules:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not pattern.search(source), module.__name__
