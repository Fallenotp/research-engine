# Install: launchctl load ~/Library/LaunchAgents/com.cleo.webread-l3-reaper.plist
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_engine import l3_guard, paths

logger = logging.getLogger(__name__)

WEBREAD_L3_IDLE_S = float(os.environ.get("WEBREAD_L3_IDLE_S", "300"))
STALE_LOCK_S = float(os.environ.get("WEBREAD_L3_STALE_LOCK_S", "600"))
WEBREAD_MLX_PORT = os.environ.get("WEBREAD_MLX_PORT", "8083")


def _read_timestamp(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _last_request_age(now: float) -> float | None:
    last_request = _read_timestamp(l3_guard.CACHE_DIR / "last_request")
    if last_request is None:
        return None
    return now - last_request


def _run_stop_command(env_name: str, default: str) -> None:
    try:
        subprocess.run(
            shlex.split(os.environ.get(env_name, default)),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        paths.safe_log(
            logger,
            logging.WARNING,
            "reaper stop command %s failed; the reap continues and is still recorded as done: %s",
            env_name,
            exc,
        )


def _mlx_port() -> int | None:
    try:
        port = int(WEBREAD_MLX_PORT)
    except ValueError:
        return None
    return port if port > 0 else None


def _stop_model_processes() -> None:
    port = _mlx_port()
    if port is None:
        return
    try:
        completed = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except Exception:
        return
    for raw_pid in completed.stdout.splitlines():
        pid_text = raw_pid.strip()
        if not pid_text:
            continue
        try:
            os.kill(int(pid_text), signal.SIGTERM)
        except Exception:
            continue


def _fresh_l3_lock_exists(now: float) -> bool:
    try:
        lock_mtime = (l3_guard.CACHE_DIR / "l3.lock").stat().st_mtime
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return now - lock_mtime < STALE_LOCK_S


def _write_last_reap() -> None:
    try:
        l3_guard.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (l3_guard.CACHE_DIR / "last_reap").write_text(str(time.time()), encoding="utf-8")
    except OSError as exc:
        paths.safe_log(
            logger,
            logging.WARNING,
            "last-reap timestamp could not be written; next run may reap early: %s",
            exc,
        )


def reap(now: float | None = None) -> bool:
    current_time = time.time() if now is None else now
    last_request_age = _last_request_age(current_time)
    if last_request_age is None:
        return False
    if last_request_age < WEBREAD_L3_IDLE_S:
        return False
    if _fresh_l3_lock_exists(current_time):
        return False
    last_request = _read_timestamp(l3_guard.CACHE_DIR / "last_request")
    last_reap = _read_timestamp(l3_guard.CACHE_DIR / "last_reap")
    if last_reap is not None and last_request is not None and last_request <= last_reap:
        return False

    # Reap only the idle L3 helpers: stop the stagehand sidecar and terminate the local MLX model.
    _run_stop_command("WEBREAD_SIDECAR_STOP_CMD", "pkill -f webread-stagehand-sidecar")
    _stop_model_processes()
    _write_last_reap()
    l3_guard.log_attempt(decision="reaped", reason="idle", rung="reaper", reaped=True)
    return True


def main() -> bool:
    return reap()


if __name__ == "__main__":
    main()
    raise SystemExit(0)
