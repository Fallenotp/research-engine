from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from research_engine import paths


def test_user_agent_uses_explicit_override_verbatim(monkeypatch) -> None:
    monkeypatch.setenv(paths.USER_AGENT_ENV, "CustomAgent/9.9")
    monkeypatch.delenv(paths.CONTACT_EMAIL_ENV, raising=False)
    paths._MISSING_CONTACT_INFO_LOGGED = False

    assert paths.user_agent() == "CustomAgent/9.9"


def test_user_agent_uses_contact_email_when_set(monkeypatch) -> None:
    monkeypatch.delenv(paths.USER_AGENT_ENV, raising=False)
    monkeypatch.setenv(paths.CONTACT_EMAIL_ENV, "owner@example.com")
    paths._MISSING_CONTACT_INFO_LOGGED = False

    assert paths.user_agent() == "research-engine/1.0 (+mailto:owner@example.com)"


def test_user_agent_uses_default_and_logs_once_without_contact(monkeypatch, caplog) -> None:
    monkeypatch.delenv(paths.USER_AGENT_ENV, raising=False)
    monkeypatch.delenv(paths.CONTACT_EMAIL_ENV, raising=False)
    paths._MISSING_CONTACT_INFO_LOGGED = False

    with caplog.at_level(logging.INFO):
        first = paths.user_agent()
        second = paths.user_agent()

    assert first == "research-engine/1.0"
    assert second == "research-engine/1.0"
    assert caplog.messages == [
        "Using default User-Agent without contact info. Set RESEARCH_ENGINE_CONTACT_EMAIL for better Crossref/Wayback rate limits."
    ]


def test_redact_paths_redacts_directories_and_preserves_basename() -> None:
    assert paths.redact_paths("/tmp/alice-secret/x") == "<path>/x"
    assert paths.redact_paths("/Volumes/External/alice/data") == "<path>/data"
    assert paths.redact_paths("/private/var/folders/zz/x") == "<path>/x"
    assert paths.redact_paths("/Users/otheruser/notes") == "<path>/notes"
    assert paths.redact_paths("see https://example.com/a/b") == (
        "see https://example.com/a/b"
    )
    assert paths.redact_paths("leave / and /tmp alone") == "leave / and /tmp alone"
    assert paths.redact_paths("word/Users/alice/private/file") == (
        "word/Users/alice/private/file"
    )
    assert paths.redact_paths("'/Users/alice/Secret Project/client/file.txt'") == (
        "'<path>/file.txt'"
    )


def test_redact_paths_file_url_redacts_absolute_keeps_scheme() -> None:
    assert paths.redact_paths("file://localhost/Users/x/y") == (
        "file://localhost/<path>/y"
    )
    assert "/Users/" not in paths.redact_paths("file://localhost/Users/x/y")
    assert paths.redact_paths("file://127.0.0.1/Users/x/y") == (
        "file://127.0.0.1/<path>/y"
    )
    assert "/Users/" not in paths.redact_paths("file://127.0.0.1/Users/x/y")
    assert paths.redact_paths("file:////Users/x/y") == "file://<path>/y"
    assert "/Users/" not in paths.redact_paths("file:////Users/x/y")
    assert paths.redact_paths("file:///Users/x/y") == "file://<path>/y"
    assert "/Users/" not in paths.redact_paths("file:///Users/x/y")


def test_redact_paths_file_url_with_single_slash_redacts_absolute_path() -> None:
    assert paths.redact_paths("file:/Users/cleo/secret/x") == "file:/<path>/x"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("failed:/Users/cleo/secret/x", "failed:<path>/x"),
        ("source:/Users/cleo/a/b", "source:<path>/b"),
        ("cache_dir:/Users/cleo/.cache/x", "cache_dir:<path>/x"),
        ("C:/Users/cleo/secret/x", "C:<path>/x"),
    ],
)
def test_redact_paths_handles_colon_adjacent_paths(
    text: str, expected: str
) -> None:
    assert paths.redact_paths(text) == expected


def test_redact_paths_redacts_single_letter_colon_adjacent_paths() -> None:
    assert "/Users/" not in paths.redact_paths("y:/Users/cleo/secret/output.csv")
    assert "/Users/" not in paths.redact_paths("n:/Users/cleo/data")
    assert "/Users/" not in paths.redact_paths("x=a:/Users/cleo/secret")
    assert "/Users/" not in paths.redact_paths("C:/Users/cleo/secret/x")


@pytest.mark.parametrize(
    "text",
    [
        "file:/Users/cleo/secret/x",
        "file://localhost/Users/x/y",
        "file:///Users/cleo/secret/x",
        "file:////Users/x/y",
        "failed:/Users/cleo/secret/x",
        "/Users/cleo/secret/x",
        "y:/Users/cleo/secret/output.csv",
        "n:/Users/cleo/data",
        "x=a:/Users/cleo/secret",
        "C:/Users/cleo/secret/x",
    ],
)
def test_redact_paths_is_idempotent(text: str) -> None:
    once = paths.redact_paths(text)

    assert paths.redact_paths(once) == once


def test_redact_paths_preserves_non_file_urls_byte_identical() -> None:
    assert paths.redact_paths("https://example.com?q=/Users/alice") == (
        "https://example.com?q=/Users/alice"
    )
    assert paths.redact_paths(
        "https://example.com/search?path=/Users/alice/data"
    ) == "https://example.com/search?path=/Users/alice/data"
    assert paths.redact_paths(
        "http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions"
    ) == "http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions"
    assert paths.redact_paths("https://example.com/a/b") == "https://example.com/a/b"
    assert paths.redact_paths("http://localhost:8888/search?q=x") == (
        "http://localhost:8888/search?q=x"
    )


def test_safe_error_and_traceback_redact_absolute_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("research_engine.tests.paths")
    missing = Path.home() / ".ssh" / "missing_key"

    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise FileNotFoundError(2, "No such file or directory", missing)
        except FileNotFoundError as exc:
            safe_exception = paths.safe_error(exc)
            assert safe_exception == (
                "[Errno 2] No such file or directory: "
                "PosixPath('<path>/missing_key')"
            )
            paths.safe_log(logger, logging.ERROR, "backend=%s", "grok", exc_info=True)

    log_text = caplog.text
    assert "Traceback (most recent call last):" in log_text
    assert "<path>/missing_key" in log_text
    for forbidden_path in (
        str(Path.home()),
        "/Users/",
        "/Volumes/",
        "/private/var/",
    ):
        assert forbidden_path not in safe_exception
        assert forbidden_path not in log_text
    assert caplog.records[0].exc_info is None


def test_safe_log_redacts_paths_after_formatting(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("research_engine.tests.paths")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        paths.safe_log(
            logger,
            logging.ERROR,
            "missing=%s",
            Path("/Users/otheruser/private/file.txt"),
        )

    assert caplog.messages == ["missing=<path>/file.txt"]


def test_safe_log_preserves_mapping_interpolation(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("research_engine.tests.paths.mapping")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        paths.safe_log(
            logger,
            logging.ERROR,
            "missing=%(path)s",
            {"path": Path("/Users/otheruser/private/file.txt")},
        )

    assert caplog.messages == ["missing=<path>/file.txt"]


def test_safe_log_swallows_logging_failure() -> None:
    logger = logging.getLogger("research_engine.tests.paths.broken")

    with patch.object(logger, "log", side_effect=RuntimeError("handler failed")):
        paths.safe_log(logger, logging.ERROR, "message")
