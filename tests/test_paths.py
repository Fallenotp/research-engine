from __future__ import annotations

import logging

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
