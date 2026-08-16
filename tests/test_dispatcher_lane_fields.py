from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

import research_engine.dispatcher as dispatcher
from research_engine.schema import Protocol


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _ScoutRouterStub:
    def __init__(self, **overrides: object) -> None:
        self._config = {
            "cli_home": dispatcher.GEMINI_SCOUT_CLI_HOME,
            "health_check_timeout_seconds": 30,
            "model_candidates": list(dispatcher.GEMINI_PRO_MODEL_CANDIDATES),
            "discord_alert_channel_id": "scout-config-channel",
            "fail_loud": False,
        }
        self._config.update(overrides)

    def scout_config(self) -> dict[str, object]:
        return dict(self._config)


@pytest.fixture(autouse=True)
def _reset_dispatcher_state(monkeypatch, tmp_path) -> None:
    dispatcher._LANE_LAST_CALL_AT.clear()
    dispatcher._WARNED_IGNORED_API_FIELDS.clear()
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None
    monkeypatch.setattr(
        dispatcher,
        "GEMINI_DAILY_COUNTER_FILE",
        tmp_path / "gemini-counter.json",
    )
    monkeypatch.setenv("RESEARCH_ENGINE_GEMINI_DAILY_BUDGET", "300")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    monkeypatch.delenv("SOME_NAME", raising=False)
    monkeypatch.delenv("MISSING_NAME", raising=False)


def _completed(
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[dispatcher.AGY_CLI],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_build_api_lane_request_honors_rate_limit_qps(monkeypatch) -> None:
    clock = _FakeClock()
    lane_config = {
        "type": "api",
        "endpoint": "https://example.com/search?q={query}",
        "rate_limit_qps": 1,
    }

    monkeypatch.setattr(dispatcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dispatcher.time, "sleep", clock.sleep)

    dispatcher.build_api_lane_request("rate_limited", lane_config, "first query")
    dispatcher.build_api_lane_request("rate_limited", lane_config, "second query")

    assert clock.sleeps == [1.0]
    assert dispatcher._LANE_LAST_CALL_AT["rate_limited"] == pytest.approx(101.0)


def test_build_api_lane_request_without_rate_limit_qps_keeps_old_behavior(monkeypatch) -> None:
    clock = _FakeClock()
    lane_config = {
        "type": "api",
        "endpoint": "https://example.com/search?q={query}",
    }

    monkeypatch.setattr(dispatcher.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dispatcher.time, "sleep", clock.sleep)

    dispatcher.build_api_lane_request("unlimited", lane_config, "first query")
    dispatcher.build_api_lane_request("unlimited", lane_config, "second query")

    assert clock.sleeps == []


def test_resolve_lane_auth_handles_none_env_and_missing_env(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "endpoints.env"
    env_file.write_text("SOME_NAME=from-env-file\n", encoding="utf-8")
    monkeypatch.setattr(dispatcher, "LANE_ENV_PATH", Path(env_file))

    no_credential = dispatcher.resolve_lane_auth("none")
    env_credential = dispatcher.resolve_lane_auth("env:SOME_NAME")
    missing_credential = dispatcher.resolve_lane_auth("env:MISSING_NAME")

    assert no_credential.present is False
    assert no_credential.material is None
    assert env_credential.present is True
    assert env_credential.material == "from-env-file"
    assert env_credential.source == "SOME_NAME"
    assert missing_credential.present is False
    assert missing_credential.material is None
    assert missing_credential.source == "MISSING_NAME"


def test_resolve_lane_auth_does_not_mutate_os_environ(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "endpoints.env"
    env_file.write_text("SOME_NAME=from-env-file\n", encoding="utf-8")
    monkeypatch.setattr(dispatcher, "LANE_ENV_PATH", Path(env_file))

    before = dict(os.environ)
    resolved = dispatcher.resolve_lane_auth("env:SOME_NAME")
    after = dict(os.environ)

    assert resolved.present is True
    assert before == after


def test_api_lane_ignored_field_warning_fires_once_and_clean_lane_stays_quiet(caplog) -> None:
    noisy_lane = {
        "type": "api",
        "endpoint": "https://example.com/search?q={query}",
        "command": "/bin/echo",
    }
    clean_lane = {
        "type": "api",
        "endpoint": "https://example.com/search?q={query}",
    }

    caplog.set_level(logging.WARNING, logger="research_engine.dispatcher")

    dispatcher.build_api_lane_request("noisy_lane", noisy_lane, "first query")
    dispatcher.build_api_lane_request("noisy_lane", noisy_lane, "second query")
    dispatcher.build_api_lane_request("clean_lane", clean_lane, "third query")

    warning_records = [
        record
        for record in caplog.records
        if "ignored field 'command'" in record.getMessage()
    ]
    assert len(warning_records) == 1
    assert "noisy_lane" in warning_records[0].getMessage()
    assert not any("clean_lane" in record.getMessage() for record in caplog.records)


def test_dispatch_scout_prefers_configured_alert_channel_id(monkeypatch) -> None:
    captured_channel_ids: list[str | None] = []

    def fake_alert(reason: str, *, channel_id: str | None = None) -> None:
        del reason
        captured_channel_ids.append(channel_id)

    monkeypatch.setattr(dispatcher, "_maybe_alert_scout_failure", fake_alert)

    spec = dispatcher.dispatch_scout(
        "test question",
        _ScoutRouterStub(discord_alert_channel_id="channel-from-config"),
        protocol=Protocol.RESEARCH,
        runner=lambda *args, **kwargs: _completed(1, stderr="Opening authentication page"),
    )

    assert spec is None
    assert captured_channel_ids == ["channel-from-config"]


def test_dispatch_scout_fail_loud_raises_when_enabled() -> None:
    with pytest.raises(dispatcher.GeminiProScoutError):
        dispatcher.dispatch_scout(
            "test question",
            _ScoutRouterStub(fail_loud=True),
            protocol=Protocol.RESEARCH,
            runner=lambda *args, **kwargs: _completed(1, stderr="Opening authentication page"),
        )
