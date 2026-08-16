from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from research_engine import extractor
from research_engine.schema import ExtractionMethod


def _completed(stdout: str, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["agent-browser"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _envelope(*, success: bool = True, data: dict | None = None, error=None) -> str:
    return json.dumps(
        {
            "success": success,
            "data": {} if data is None else data,
            "error": error,
        }
    )


def test_agent_browser_method_exists() -> None:
    assert ExtractionMethod.AGENT_BROWSER.value == "agent_browser"


def test_agent_browser_success_returns_payload() -> None:
    politeness = MagicMock()
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        command = tuple(args[3:])
        if command[:1] == ("open",):
            return _completed(
                _envelope(
                    data={"title": "Example Domain", "url": "https://example.org/"}
                )
            )
        if command[:2] == ("get", "title"):
            return _completed(_envelope(data={"title": "Exact Title"}))
        if command[:3] == ("get", "text", "body"):
            return _completed(
                _envelope(
                    data={
                        "origin": "https://example.org/",
                        "text": "Example body text from agent-browser.",
                    }
                )
            )
        if command[:1] == ("close",):
            return _completed("")
        raise AssertionError(f"unexpected command: {args}")

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=fake_run,
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is not None
    assert out["title"] == "Exact Title"
    assert "Example body text from agent-browser." in (out.get("text") or "")
    assert [tuple(call[3:]) for call in calls] == [
        ("open", "https://example.org", "--json"),
        ("get", "title", "--json"),
        ("get", "text", "body", "--json"),
        ("close",),
    ]
    politeness.wait.assert_called_once_with("example.org")


@pytest.mark.parametrize("failing_command", [("open",), ("get", "text", "body")])
def test_agent_browser_always_closes_session_on_failure(failing_command) -> None:
    politeness = MagicMock()
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        command = tuple(args[3:])
        if command[:1] == ("close",):
            return _completed("")
        if command[: len(failing_command)] == failing_command:
            raise RuntimeError("boom")
        if command[:1] == ("open",):
            return _completed(_envelope(data={"title": "Example", "url": "https://example.org/"}))
        if command[:2] == ("get", "title"):
            return _completed(_envelope(data={"title": "Example"}))
        if command[:3] == ("get", "text", "body"):
            return _completed(_envelope(data={"text": "Example body text"}))
        raise AssertionError(f"unexpected command: {args}")

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=fake_run,
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is None
    assert any(call[3] == "close" for call in calls)


def test_agent_browser_failure_envelope_returns_none() -> None:
    politeness = MagicMock()

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=[
            _completed(_envelope(success=False, data={"message": "blocked"}, error="blocked")),
            _completed(""),
        ],
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is None


def test_agent_browser_non_json_stdout_returns_none() -> None:
    politeness = MagicMock()

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=[
            _completed("not json"),
            _completed(""),
        ],
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is None


def test_agent_browser_empty_text_returns_none() -> None:
    politeness = MagicMock()

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=[
            _completed(_envelope(data={"title": "Example Domain", "url": "https://example.org/"})),
            _completed(_envelope(data={"title": "Title"})),
            _completed(_envelope(data={"origin": "https://example.org/", "text": "   "})),
            _completed(""),
        ],
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is None


def test_agent_browser_missing_binary_returns_none() -> None:
    politeness = MagicMock()

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=FileNotFoundError(),
    ):
        out = extractor._agent_browser("https://example.org")

    assert out is None


def test_agent_browser_session_name_varies_by_url(monkeypatch: pytest.MonkeyPatch) -> None:
    politeness = MagicMock()
    session_names: list[str] = []

    monkeypatch.setattr(extractor.os, "getpid", lambda: 999)
    monkeypatch.setattr(extractor.time, "time_ns", lambda: 123456789)

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        if args[3] == "open":
            session_names.append(args[2])
            return _completed(_envelope(data={"title": "Example Domain", "url": args[4]}))
        if args[3:5] == ["get", "title"]:
            return _completed(_envelope(data={"title": "Title"}))
        if args[3:6] == ["get", "text", "body"]:
            return _completed(_envelope(data={"text": "Example body text"}))
        if args[3] == "close":
            return _completed("")
        raise AssertionError(f"unexpected command: {args}")

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor.subprocess,
        "run",
        side_effect=fake_run,
    ):
        first = extractor._agent_browser("https://example.org/one")
        second = extractor._agent_browser("https://example.org/two")

    assert first is not None
    assert second is not None
    assert len(session_names) == 2
    assert session_names[0] != session_names[1]


def test_agent_browser_respects_robots_and_skips_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    politeness = MagicMock()
    politeness.allowed.return_value = False
    monkeypatch.setenv("RESEARCH_RESPECT_ROBOTS", "1")

    with patch.object(extractor, "_get_politeness", return_value=politeness), patch.object(
        extractor,
        "note_block",
    ) as note_block_mock, patch.object(extractor.subprocess, "run") as run_mock:
        out = extractor._agent_browser("https://example.org")

    assert out is None
    run_mock.assert_not_called()
    politeness.wait.assert_not_called()
    note_block_mock.assert_called_once_with(
        "https://example.org",
        method="agent_browser",
        reason="robots_disallow",
    )
