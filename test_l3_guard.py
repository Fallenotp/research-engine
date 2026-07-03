from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

l3_guard = importlib.import_module("research_engine.l3_guard")


def _vm_stat_output(
    *,
    free_pages: int,
    inactive_pages: int,
    purgeable_pages: int,
    page_size: int = 4096,
) -> str:
    return "\n".join(
        [
            f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)",
            f"Pages free:                               {free_pages}.",
            "Pages active:                             12345.",
            f"Pages inactive:                           {inactive_pages}.",
            "Pages speculative:                        6789.",
            f"Pages purgeable:                          {purgeable_pages}.",
        ]
    )


def _run_command_factory(
    *,
    free_pages: int = 1_000_000,
    inactive_pages: int = 1_000_000,
    purgeable_pages: int = 1_000_000,
    pressure: int = 1,
    swap_used: str = "512.00M",
) -> callable:
    vm_stat_output = _vm_stat_output(
        free_pages=free_pages,
        inactive_pages=inactive_pages,
        purgeable_pages=purgeable_pages,
    )

    def fake_run_command(args: list[str]) -> str:
        if args == ["vm_stat"]:
            return vm_stat_output
        if args == ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]:
            return str(pressure)
        if args == ["sysctl", "-n", "vm.swapusage"]:
            return f"total = 8192.00M  used = {swap_used}  free = 7680.00M"
        raise AssertionError(f"unexpected command: {args}")

    return fake_run_command


def _disk_usage_with_free_gb(free_gb: float) -> SimpleNamespace:
    free_bytes = int(free_gb * 1024 * 1024 * 1024)
    return SimpleNamespace(total=free_bytes * 2, used=free_bytes, free=free_bytes)


def _patch_healthy_system(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(l3_guard, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("WEBREAD_L3_KILL", raising=False)
    monkeypatch.delenv("WEBREAD_L3_MIN_FREE_MB", raising=False)
    monkeypatch.delenv("WEBREAD_L3_MAX_SWAP_MB", raising=False)


def test_preflight_all_clear(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (True, None)


def test_preflight_blocks_low_ram(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(
            free_pages=100_000,
            inactive_pages=100_000,
            purgeable_pages=100_000,
        ),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_LOW_RAM")


def test_preflight_blocks_critical_pressure(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(pressure=4),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_PRESSURE")


def test_preflight_blocks_high_swap(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(swap_used="17408.00M"),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_PRESSURE")


def test_preflight_blocks_low_disk(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(5),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_LOW_DISK")


def test_preflight_blocks_killswitch_env(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)
    monkeypatch.setenv("WEBREAD_L3_KILL", "1")

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(
            free_pages=1,
            inactive_pages=1,
            purgeable_pages=1,
        ),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_KILLSWITCH")


def test_preflight_blocks_killswitch_file(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "l3.off").write_text("", encoding="utf-8")

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_KILLSWITCH")


def test_preflight_blocks_open_breaker(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "l3_breaker.json").write_text(
        json.dumps({"consecutive_failures": 3, "last_fail_time": 1_000.0}),
        encoding="utf-8",
    )

    with patch.object(l3_guard.time, "time", return_value=1_500.0), patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_BREAKER")


def test_preflight_checks_killswitch_before_ram(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)
    monkeypatch.setenv("WEBREAD_L3_KILL", "1")

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(
            free_pages=1,
            inactive_pages=1,
            purgeable_pages=1,
        ),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_KILLSWITCH")


def test_preflight_blocks_on_probe_failure(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=subprocess.CalledProcessError(1, ["vm_stat"]),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_PROBE_FAIL")


def test_preflight_killswitch_wins_when_probes_fail(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)
    monkeypatch.setenv("WEBREAD_L3_KILL", "1")

    with patch.object(
        l3_guard,
        "_run_command",
        side_effect=subprocess.CalledProcessError(1, ["vm_stat"]),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_KILLSWITCH")


def test_record_failure_opens_breaker_and_success_resets(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(l3_guard.time, "time", side_effect=[100.0, 200.0, 300.0]):
        l3_guard.record_failure()
        l3_guard.record_failure()
        l3_guard.record_failure()

    with patch.object(l3_guard.time, "time", return_value=500.0), patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (False, "L3_BLOCKED_BREAKER")

    l3_guard.record_success()

    with patch.object(l3_guard.time, "time", return_value=600.0), patch.object(
        l3_guard,
        "_run_command",
        side_effect=_run_command_factory(),
    ), patch.object(
        l3_guard.shutil,
        "disk_usage",
        return_value=_disk_usage_with_free_gb(50),
    ):
        assert l3_guard.preflight() == (True, None)


def test_record_failure_updates_count_under_breaker_lock(monkeypatch, tmp_path) -> None:
    _patch_healthy_system(monkeypatch, tmp_path)

    with patch.object(l3_guard.time, "time", side_effect=[100.0, 200.0]):
        l3_guard.record_failure()
        l3_guard.record_failure()

    breaker_state = json.loads((tmp_path / "l3_breaker.json").read_text(encoding="utf-8"))
    assert breaker_state["consecutive_failures"] == 2
    assert breaker_state["last_fail_time"] == 200.0
    assert l3_guard._breaker_lock_path() == tmp_path / "l3_breaker.lock"
    assert (tmp_path / "l3_breaker.lock").exists()


def test_log_attempt_writes_parseable_json_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(l3_guard, "CACHE_DIR", tmp_path)

    l3_guard.log_attempt(
        url="https://x",
        decision="blocked",
        reason="L3_BLOCKED_LOW_RAM",
        rung="steel_stagehand",
        free_mb=123.0,
        elapsed_ms=5,
    )

    log_path = tmp_path / "l3.log"
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["url"] == "https://x"
    assert payload["decision"] == "blocked"
    assert payload["reason"] == "L3_BLOCKED_LOW_RAM"
    assert payload["rung"] == "steel_stagehand"
    assert payload["free_mb"] == 123.0
    assert payload["elapsed_ms"] == 5

    l3_guard.log_attempt(
        url="https://x",
        decision="blocked",
        reason="L3_BLOCKED_LOW_RAM",
        rung="steel_stagehand",
        free_mb=123.0,
        elapsed_ms=5,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
