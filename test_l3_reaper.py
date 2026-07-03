from __future__ import annotations

import importlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_reaper(monkeypatch, tmp_path, *, idle_s: str = "300"):
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WEBREAD_L3_IDLE_S", idle_s)

    l3_guard = importlib.import_module("research_engine.l3_guard")
    importlib.reload(l3_guard)

    l3_reaper = importlib.import_module("research_engine.l3_reaper")
    return importlib.reload(l3_reaper)


def _write_last_request(tmp_path: Path, value: float) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_request").write_text(str(value), encoding="utf-8")


def _write_last_reap(tmp_path: Path, value: float) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_reap").write_text(str(value), encoding="utf-8")


def test_reap_skips_recent_request(monkeypatch, tmp_path) -> None:
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 900.0)

    with patch.object(l3_reaper.subprocess, "run") as run_mock:
        assert l3_reaper.reap(now=1_000.0) is False

    run_mock.assert_not_called()


def test_script_entrypoint_runs_reap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WEBREAD_L3_IDLE_S", "300")
    monkeypatch.setenv("WEBREAD_MLX_PORT", "8083")
    monkeypatch.setenv("WEBREAD_SIDECAR_STOP_CMD", "stop-sidecar --now")
    l3_guard = importlib.import_module("research_engine.l3_guard")
    importlib.reload(l3_guard)
    _write_last_request(tmp_path, 600.0)

    with patch("subprocess.run") as run_mock, patch("os.kill") as kill_mock, patch(
        "time.time",
        return_value=1_234.5,
    ), patch.object(l3_guard, "log_attempt") as log_mock:
        run_mock.side_effect = [
            subprocess.CompletedProcess(["stop-sidecar", "--now"], 0, "", ""),
            subprocess.CompletedProcess(["lsof", "-ti", "tcp:8083"], 0, "", ""),
        ]
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(Path(__file__).with_name("l3_reaper.py")), run_name="__main__")

    assert excinfo.value.code == 0
    assert run_mock.call_count == 2
    kill_mock.assert_not_called()
    assert (tmp_path / "last_reap").read_text(encoding="utf-8").strip() == "1234.5"
    log_mock.assert_called_once_with(
        decision="reaped",
        reason="idle",
        rung="reaper",
        reaped=True,
    )


def test_reap_stops_idle_processes_and_logs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_STOP_CMD", "stop-sidecar --now")
    monkeypatch.setenv("WEBREAD_MLX_PORT", "9090")
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 600.0)

    with patch.object(
        l3_reaper.subprocess,
        "run",
        side_effect=[
            subprocess.CompletedProcess(["stop-sidecar", "--now"], 0, "", ""),
            subprocess.CompletedProcess(["lsof", "-ti", "tcp:9090"], 0, "123\n456\n", ""),
        ],
    ) as run_mock, patch.object(l3_reaper.os, "kill") as kill_mock:
        assert l3_reaper.reap(now=1_000.0) is True

    assert run_mock.call_args_list == [
        call(
            ["stop-sidecar", "--now"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ),
        call(
            ["lsof", "-ti", "tcp:9090"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ),
    ]
    assert kill_mock.call_args_list == [
        call(123, l3_reaper.signal.SIGTERM),
        call(456, l3_reaper.signal.SIGTERM),
    ]
    lines = (tmp_path / "l3.log").read_text(encoding="utf-8").splitlines()
    assert lines
    assert json.loads(lines[-1]) == {
        "decision": "reaped",
        "reason": "idle",
        "reaped": True,
        "rung": "reaper",
    }


def test_reap_skips_when_already_reaped_since_last_request(monkeypatch, tmp_path) -> None:
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 600.0)
    _write_last_reap(tmp_path, 700.0)

    with patch.object(l3_reaper.subprocess, "run") as run_mock:
        assert l3_reaper.reap(now=1_000.0) is False

    run_mock.assert_not_called()
    assert not (tmp_path / "l3.log").exists()


def test_reap_runs_once_when_new_activity_happened_since_last_reap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_STOP_CMD", "stop-sidecar --now")
    monkeypatch.setenv("WEBREAD_MLX_PORT", "8083")
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 600.0)
    _write_last_reap(tmp_path, 500.0)

    with patch.object(l3_reaper.time, "time", return_value=1_234.5):
        with patch.object(
            l3_reaper.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["stop-sidecar", "--now"], 0, "", ""),
                subprocess.CompletedProcess(["lsof", "-ti", "tcp:8083"], 0, "", ""),
            ],
        ) as run_mock, patch.object(l3_reaper.os, "kill") as kill_mock:
            assert l3_reaper.reap(now=1_000.0) is True

    assert run_mock.call_count == 2
    kill_mock.assert_not_called()
    assert (tmp_path / "last_reap").read_text(encoding="utf-8").strip() == "1234.5"


def test_reap_skips_when_last_request_is_missing(monkeypatch, tmp_path) -> None:
    l3_reaper = _load_reaper(monkeypatch, tmp_path)

    with patch.object(l3_reaper.subprocess, "run") as run_mock, patch.object(
        l3_reaper.os,
        "kill",
    ) as kill_mock:
        assert l3_reaper.reap(now=1_000.0) is False

    run_mock.assert_not_called()
    kill_mock.assert_not_called()


def test_reap_skips_fresh_l3_lock(monkeypatch, tmp_path) -> None:
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 600.0)
    lock_path = tmp_path / "l3.lock"
    lock_path.write_text("950.0", encoding="utf-8")
    os.utime(lock_path, (950.0, 950.0))

    with patch.object(l3_reaper.subprocess, "run") as run_mock:
        assert l3_reaper.reap(now=1_000.0) is False

    run_mock.assert_not_called()


def test_reap_proceeds_when_l3_lock_is_stale(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_STOP_CMD", "stop-sidecar --now")
    monkeypatch.setenv("WEBREAD_MLX_PORT", "8083")
    monkeypatch.setenv("WEBREAD_L3_STALE_LOCK_S", "60")
    l3_reaper = _load_reaper(monkeypatch, tmp_path)
    _write_last_request(tmp_path, 600.0)
    lock_path = tmp_path / "l3.lock"
    lock_path.write_text("800.0", encoding="utf-8")
    os.utime(lock_path, (800.0, 800.0))

    with patch.object(
        l3_reaper.subprocess,
        "run",
        side_effect=[
            subprocess.CompletedProcess(["stop-sidecar", "--now"], 0, "", ""),
            subprocess.CompletedProcess(["lsof", "-ti", "tcp:8083"], 0, "", ""),
        ],
    ) as run_mock, patch.object(l3_reaper.os, "kill") as kill_mock:
        assert l3_reaper.reap(now=1_000.0) is True

    assert run_mock.call_count == 2
    kill_mock.assert_not_called()


@pytest.mark.parametrize(
    ("port_value", "expected"),
    [
        ("8083", 8083),
        ("abc", None),
        ("0", None),
        ("-1", None),
        ("", None),
    ],
)
def test_mlx_port_validates_real_env_values(monkeypatch, tmp_path, port_value: str, expected: int | None) -> None:
    monkeypatch.setenv("WEBREAD_MLX_PORT", port_value)
    l3_reaper = _load_reaper(monkeypatch, tmp_path)

    assert l3_reaper._mlx_port() == expected
