from __future__ import annotations

import logging

import pytest

from research_engine import llm_call


def test_llm_complete_with_backend_supports_opus(monkeypatch) -> None:
    calls = []

    def fake_run_backend(
        prompt: str,
        *,
        backend: str,
        timeout: int | float,
        model: str | None = None,
    ) -> str:
        calls.append((prompt, backend, timeout, model))
        return "READY"

    monkeypatch.setattr(llm_call, "_run_backend", fake_run_backend)

    output, backend = llm_call.llm_complete_with_backend(
        "Reply with READY.",
        backend="opus",
        timeout=45,
    )

    assert output == "READY"
    assert backend == "opus"
    assert calls == [("Reply with READY.", "opus", 45, None)]


@pytest.mark.parametrize("model", [llm_call.CODEX_DEFAULT_MODEL, "gpt-5.5"])
def test_run_codex_uses_requested_model(monkeypatch, model: str) -> None:
    captured_args = {}

    monkeypatch.setattr(llm_call, "_is_executable", lambda path: True)

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured_args["args"] = args
            self.args = args
            self.returncode = 0
            self.pid = 123

        def communicate(self, prompt: str, timeout: int | float):
            assert prompt == "hello"
            assert timeout == 12
            return "READY\n", ""

    monkeypatch.setattr(llm_call.subprocess, "Popen", FakePopen)

    output = llm_call._run_codex("hello", timeout=12, model=model)

    assert output == "READY"
    assert captured_args["args"][-1] == model


def test_run_gemini_backend_uses_agy_cli_not_retired_gemini(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return llm_call.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="READY\n",
            stderr="",
        )

    monkeypatch.setattr(llm_call, "_agy_binary", lambda: "agy-cli-1")
    monkeypatch.setattr(llm_call.subprocess, "run", fake_run)

    output = llm_call._run_gemini("hello", timeout=12)

    assert output == "READY"
    assert calls
    cmd, kwargs = calls[0]
    assert cmd == [
        "agy-cli-1",
        "--dangerously-skip-permissions",
        "--print",
        "hello",
    ]
    assert "input" not in kwargs
    assert "env" not in kwargs
    assert kwargs["stdin"] == llm_call.subprocess.DEVNULL


def test_llm_complete_logs_backend_failures_and_raises_with_details(
    monkeypatch,
    caplog,
) -> None:
    def fake_run_backend(
        prompt: str,
        *,
        backend: str,
        timeout: int | float,
        model: str | None = None,
    ) -> str:
        raise RuntimeError(f"{backend} failed")

    monkeypatch.setattr(llm_call, "_run_backend", fake_run_backend)
    caplog.set_level(logging.WARNING, logger=llm_call.__name__)

    with pytest.raises(RuntimeError) as excinfo:
        llm_call.llm_complete("Reply with READY.")

    assert "codex: RuntimeError: codex failed" in str(excinfo.value)
    assert "sonnet: RuntimeError: sonnet failed" in str(excinfo.value)
    assert "llm backend failed: codex: RuntimeError: codex failed" in caplog.text
    assert "llm backend failed: sonnet: RuntimeError: sonnet failed" in caplog.text


def test_run_codex_missing_binary_names_env_var(monkeypatch) -> None:
    monkeypatch.setattr(llm_call.paths, "require_executable", lambda *args: (_ for _ in ()).throw(
        FileNotFoundError(f"Set {llm_call.paths.CODEX_BIN_ENV}")
    ))

    with pytest.raises(FileNotFoundError) as excinfo:
        llm_call._run_codex("hello", timeout=12)

    assert str(excinfo.value) == f"Set {llm_call.paths.CODEX_BIN_ENV}"


def test_run_sonnet_missing_binary_names_env_var(monkeypatch) -> None:
    monkeypatch.setattr(llm_call.paths, "require_executable", lambda *args: (_ for _ in ()).throw(
        FileNotFoundError(f"Set {llm_call.paths.CLAUDE_BIN_ENV}")
    ))

    with pytest.raises(FileNotFoundError) as excinfo:
        llm_call._run_sonnet("hello", timeout=12)

    assert str(excinfo.value) == f"Set {llm_call.paths.CLAUDE_BIN_ENV}"


def test_run_gemini_missing_binary_names_env_var(monkeypatch) -> None:
    monkeypatch.setattr(llm_call.paths, "require_executable", lambda *args: (_ for _ in ()).throw(
        FileNotFoundError(f"Set {llm_call.paths.AGY_BIN_ENV}")
    ))

    with pytest.raises(FileNotFoundError) as excinfo:
        llm_call._run_gemini("hello", timeout=12)

    assert str(excinfo.value) == f"Set {llm_call.paths.AGY_BIN_ENV}"


def test_agy_binary_uses_existing_env_var_and_fallback_names(monkeypatch) -> None:
    captured = {}

    def fake_require_executable(env_var: str, *names: str) -> str:
        captured["env_var"] = env_var
        captured["names"] = names
        return "/tmp/agy"

    monkeypatch.setattr(llm_call.paths, "require_executable", fake_require_executable)

    assert llm_call._agy_binary() == "/tmp/agy"
    assert captured == {
        "env_var": llm_call.paths.AGY_BIN_ENV,
        "names": ("agy-cli-1", "agy-cli-2", "agy"),
    }
