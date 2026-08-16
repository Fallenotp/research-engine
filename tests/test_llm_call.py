from __future__ import annotations

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
