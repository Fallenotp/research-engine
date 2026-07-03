from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "webread"
CACHE_DIR = Path(os.environ.get("WEBREAD_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()

L3_BLOCKED_KILLSWITCH = "L3_BLOCKED_KILLSWITCH"
L3_BLOCKED_BREAKER = "L3_BLOCKED_BREAKER"
L3_BLOCKED_PROBE_FAIL = "L3_BLOCKED_PROBE_FAIL"
L3_BLOCKED_LOW_RAM = "L3_BLOCKED_LOW_RAM"
L3_BLOCKED_PRESSURE = "L3_BLOCKED_PRESSURE"
L3_BLOCKED_LOW_DISK = "L3_BLOCKED_LOW_DISK"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def _load_breaker_state() -> dict[str, float | int]:
    path = _cache_path("l3_breaker.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"consecutive_failures": 0, "last_fail_time": 0.0}
    if not isinstance(payload, dict):
        return {"consecutive_failures": 0, "last_fail_time": 0.0}
    return {
        "consecutive_failures": int(payload.get("consecutive_failures", 0) or 0),
        "last_fail_time": float(payload.get("last_fail_time", 0.0) or 0.0),
    }


def _write_breaker_state(state: dict[str, float | int]) -> None:
    _ensure_cache_dir()
    path = _cache_path("l3_breaker.json")
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state), encoding="utf-8")
    tmp_path.replace(path)


def _breaker_lock_path() -> Path:
    return _cache_path("l3_breaker.lock")


@contextmanager
def _breaker_update_lock():
    _ensure_cache_dir()
    lock_handle = None
    if fcntl is not None:
        try:
            lock_handle = _breaker_lock_path().open("a+", encoding="utf-8")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError:
            if lock_handle is not None:
                lock_handle.close()
            lock_handle = None
    try:
        yield
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()


def _breaker_open(now: float | None = None) -> bool:
    state = _load_breaker_state()
    last_fail_time = float(state["last_fail_time"])
    current_time = time.time() if now is None else now
    return int(state["consecutive_failures"]) >= 3 and current_time < (last_fail_time + 3600)


def _vm_stat_value(vm_stat_output: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", vm_stat_output, re.MULTILINE)
    if not match:
        raise ValueError(f"missing vm_stat field: {label}")
    return int(match.group(1))


def _freeable_mb() -> float:
    vm_stat_output = _run_command(["vm_stat"])
    page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
    if not page_size_match:
        raise ValueError("missing vm_stat page size")
    page_size = int(page_size_match.group(1))
    freeable_pages = (
        _vm_stat_value(vm_stat_output, "Pages free")
        + _vm_stat_value(vm_stat_output, "Pages inactive")
        + _vm_stat_value(vm_stat_output, "Pages purgeable")
    )
    return freeable_pages * page_size / (1024 * 1024)


def _memory_pressure_level() -> int:
    return int(_run_command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))


def _swap_used_mb() -> float:
    swap_output = _run_command(["sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used =\s*([0-9.]+)([KMGT])", swap_output)
    if not match:
        raise ValueError("missing vm.swapusage used value")
    value = float(match.group(1))
    unit = match.group(2)
    scale = {
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
    }[unit]
    return value * scale


def killswitch_on() -> bool:
    return os.environ.get("WEBREAD_L3_KILL") == "1" or _cache_path("l3.off").exists()


def record_failure() -> None:
    with _breaker_update_lock():
        state = _load_breaker_state()
        _write_breaker_state(
            {
                "consecutive_failures": int(state["consecutive_failures"]) + 1,
                "last_fail_time": time.time(),
            }
        )


def record_success() -> None:
    with _breaker_update_lock():
        _write_breaker_state({"consecutive_failures": 0, "last_fail_time": 0.0})


def log_attempt(**fields) -> None:
    _ensure_cache_dir()
    payload = json.dumps(fields, sort_keys=True) + "\n"
    fd = os.open(_cache_path("l3.log"), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(payload)


def preflight() -> tuple[bool, str | None]:
    if killswitch_on():
        return False, L3_BLOCKED_KILLSWITCH

    if _breaker_open():
        return False, L3_BLOCKED_BREAKER

    try:
        min_free_mb = int(os.environ.get("WEBREAD_L3_MIN_FREE_MB", "5120"))
        if _freeable_mb() < min_free_mb:
            return False, L3_BLOCKED_LOW_RAM

        if _memory_pressure_level() != 1:
            return False, L3_BLOCKED_PRESSURE

        max_swap_mb = int(os.environ.get("WEBREAD_L3_MAX_SWAP_MB", "16384"))
        if _swap_used_mb() > max_swap_mb:
            return False, L3_BLOCKED_PRESSURE

        free_disk_gb = shutil.disk_usage(Path.home()).free / (1024 * 1024 * 1024)
        if free_disk_gb < 10:
            return False, L3_BLOCKED_LOW_DISK
    except (subprocess.SubprocessError, OSError, ValueError):
        return False, L3_BLOCKED_PROBE_FAIL

    return True, None
