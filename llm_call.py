from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path

from research_engine import paths

CODEX_DEFAULT_MODEL = "gpt-5.4-mini"
SONNET_MODEL = "sonnet"
OPUS_MODEL = "opus"
AGY_CLI = "agy-cli-1"
AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
GEMINI_TIMEOUT_SECONDS = 180
BACKENDS = ("codex", "sonnet", "opus", "gemini")
logger = logging.getLogger(__name__)


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
    failures: list[str] = []
    for backend in _backend_order(prefer):
        try:
            text = _run_backend(prompt, backend=backend, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - backend fallback boundary
            failure = f"{backend}: {type(exc).__name__}: {exc}"
            logger.warning("llm backend failed: %s", failure)
            failures.append(failure)
            continue
        if text:
            return text, backend
    detail = "; ".join(failures) if failures else "no backends attempted"
    raise RuntimeError(f"no llm backend; failures: {detail}")


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
    codex_bin = paths.require_executable(paths.CODEX_BIN_ENV, "codex")
    proc = subprocess.Popen(
        [
            codex_bin,
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
    claude_bin = paths.require_executable(paths.CLAUDE_BIN_ENV, "claude")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(paths.optional_path(paths.CLAUDE_HOME_ENV) or Path.home())
    completed = subprocess.run(
        [
            claude_bin,
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
    claude_bin = paths.require_executable(paths.CLAUDE_BIN_ENV, "claude")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["HOME"] = str(paths.optional_path(paths.CLAUDE_HOME_ENV) or Path.home())
    completed = subprocess.run(
        [
            claude_bin,
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


def _agy_binary() -> str:
    return paths.require_executable(paths.AGY_BIN_ENV, AGY_CLI, "agy-cli-2", "agy")


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
