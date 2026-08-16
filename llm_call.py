from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from . import paths


CODEX_BIN = Path("/opt/homebrew/bin/codex")
CLAUDE_HOME = paths.optional_path(paths.CLAUDE_HOME_ENV) or Path.home()
CODEX_DEFAULT_MODEL = "gpt-5.4-mini"
SONNET_MODEL = "sonnet"
OPUS_MODEL = "opus"
AGY_CLI = "agy-cli-1"
AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
GEMINI_TIMEOUT_SECONDS = 180
BACKENDS = ("codex", "sonnet", "opus", "gemini")


def llm_complete_with_backend(
    prompt: str,
    *,
    backend: str,
    timeout: int | float = 120,
    model: str | None = None,
) -> tuple[str, str]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported llm backend: {backend}")
    return _run_backend(prompt, backend=backend, timeout=timeout, model=model), backend


def llm_complete(
    prompt: str,
    *,
    timeout: int | float = 120,
    prefer: str | None = None,
) -> tuple[str, str]:
    for backend in _backend_order(prefer):
        try:
            text = _run_backend(prompt, backend=backend, timeout=timeout)
        except Exception:  # noqa: BLE001 - backend fallback boundary
            continue
        if text:
            return text, backend
    raise RuntimeError("no llm backend")


def _backend_order(prefer: str | None) -> list[str]:
    if prefer is None:
        return ["codex", "sonnet"]
    ordered = list(BACKENDS)
    if prefer in BACKENDS:
        ordered.remove(prefer)
        ordered.insert(0, prefer)
    return ordered


def _run_backend(
    prompt: str,
    *,
    backend: str,
    timeout: int | float,
    model: str | None = None,
) -> str:
    if backend == "codex":
        return _run_codex(prompt, timeout=timeout, model=model or CODEX_DEFAULT_MODEL)
    if backend == "sonnet":
        return _run_sonnet(prompt, timeout=timeout)
    if backend == "opus":
        return _run_opus(prompt, timeout=timeout)
    return _run_gemini(prompt, timeout=timeout)


def _run_codex(
    prompt: str,
    *,
    timeout: int | float,
    model: str = CODEX_DEFAULT_MODEL,
) -> str:
    if not _is_executable(CODEX_BIN):
        raise FileNotFoundError(str(CODEX_BIN))
    proc = subprocess.Popen(
        [
            str(CODEX_BIN),
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-m",
            model,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise
    completed = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return (completed.stdout or "").strip()


def _run_sonnet(prompt: str, *, timeout: int | float) -> str:
    claude_bin = Path(paths.require_executable(paths.CLAUDE_BIN_ENV, "claude"))
    if not _is_executable(claude_bin):
        raise FileNotFoundError(str(claude_bin))
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(CLAUDE_HOME)
    completed = subprocess.run(
        [
            str(claude_bin),
            "-p",
            prompt,
            "--model",
            SONNET_MODEL,
            "--no-session-persistence",
            "--disable-slash-commands",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return (completed.stdout or "").strip()


def _run_opus(prompt: str, *, timeout: int | float) -> str:
    claude_bin = Path(paths.require_executable(paths.CLAUDE_BIN_ENV, "claude"))
    if not _is_executable(claude_bin):
        raise FileNotFoundError(str(claude_bin))
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(CLAUDE_HOME)
    completed = subprocess.run(
        [
            str(claude_bin),
            "-p",
            prompt,
            "--model",
            OPUS_MODEL,
            "--no-session-persistence",
            "--disable-slash-commands",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return (completed.stdout or "").strip()


def _run_gemini(prompt: str, *, timeout: int | float) -> str:
    agy = _agy_binary()
    if agy is None:
        raise FileNotFoundError(AGY_CLI)
    completed = subprocess.run(
        [agy, AGY_SKIP_PERMISSIONS_FLAG, "--print", _agy_prompt_arg(prompt)],
        capture_output=True,
        text=True,
        timeout=max(float(timeout), float(GEMINI_TIMEOUT_SECONDS)),
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return (completed.stdout or "").strip()


def _agy_binary() -> str | None:
    return paths.executable(paths.AGY_BIN_ENV, AGY_CLI, "agy-cli-2", "agy")


def _agy_prompt_arg(prompt: str) -> str:
    if prompt.lstrip().startswith("-"):
        return f"Prompt:\n{prompt}"
    return prompt


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


if __name__ == "__main__":
    output, backend = llm_complete("Reply with exactly: OK")
    print(output)
    print(f"backend={backend}")
