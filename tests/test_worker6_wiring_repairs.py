from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from research_engine import telemetry_observer
from research_engine.router import DEFAULT_CONFIG_PATH, EXPECTED_SCHEMA_VERSION, load_router


def _write_router_config(tmp_path: Path, *, schema_version=EXPECTED_SCHEMA_VERSION) -> Path:
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    if schema_version is None:
        config.pop("schema_version", None)
    else:
        config["schema_version"] = schema_version
    config_path = tmp_path / "router_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_router_expected_schema_version_loads_without_warning(tmp_path, caplog) -> None:
    config_path = _write_router_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="research_engine.router"):
        router = load_router(str(config_path))

    assert len(router._rules) == 18
    assert "schema_version mismatch" not in caplog.text


def test_router_wrong_schema_version_warns_and_still_loads(tmp_path, caplog) -> None:
    config_path = _write_router_config(tmp_path, schema_version="9.9.9")

    with caplog.at_level(logging.WARNING, logger="research_engine.router"):
        router = load_router(str(config_path))

    assert len(router._rules) == 18
    assert "9.9.9" in caplog.text
    assert EXPECTED_SCHEMA_VERSION in caplog.text


def test_router_missing_schema_version_warns_and_still_loads(tmp_path, caplog) -> None:
    config_path = _write_router_config(tmp_path, schema_version=None)

    with caplog.at_level(logging.WARNING, logger="research_engine.router"):
        router = load_router(str(config_path))

    assert len(router._rules) == 18
    assert "<missing>" in caplog.text
    assert EXPECTED_SCHEMA_VERSION in caplog.text


def test_summarize_calls_aggregates_synthetic_rows_and_handles_empty_input() -> None:
    summary = telemetry_observer.summarize_calls(
        [
            {"lane": "exa_direct", "ok": True, "duration_ms": 100},
            {"lane": "exa_direct", "ok": False, "duration_ms": 300},
            {"lane": "linkup", "ok": True, "duration_ms": "250"},
            {"lane": "", "ok": True, "duration_ms": 999},
            {"not_lane": "ignored"},
        ]
    )

    assert summary["exa_direct"] == {
        "calls": 2,
        "ok": 1,
        "failed": 1,
        "avg_ms": 200.0,
    }
    assert summary["linkup"] == {
        "calls": 1,
        "ok": 1,
        "failed": 0,
        "avg_ms": 250.0,
    }
    assert telemetry_observer.summarize_calls([]) == {}


def test_telemetry_observer_main_summarize_calls_prints_cli_report(tmp_path, monkeypatch, capsys) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(telemetry_observer, "CALL_LOG", call_log)
    call_log.parent.mkdir(parents=True, exist_ok=True)
    call_log.write_text(
        "\n".join(
            [
                json.dumps({"lane": "searxng_general", "ok": True, "duration_ms": 100}),
                json.dumps({"lane": "searxng_general", "ok": False, "duration_ms": 300}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = telemetry_observer.main(["--summarize-calls"])
    stdout = capsys.readouterr().out

    assert summary["searxng_general"]["calls"] == 2
    assert "searxng_general: calls=2 ok=1 failed=1 success_rate=50.0% avg_ms=200.0" in stdout


def test_telemetry_observer_main_log_buzz_exposes_cli_hook(tmp_path, monkeypatch, capsys) -> None:
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)

    result = telemetry_observer.main(
        [
            "--log-buzz",
            "Fine-tuning agents",
            "--n-signals",
            "2",
            "--platform",
            "reddit",
            "--platform",
            "x",
            "--agent",
            "agent-a",
        ]
    )
    stdout = capsys.readouterr().out
    rows = [json.loads(line) for line in master_log.read_text(encoding="utf-8").splitlines()]

    assert result["logged"] is True
    assert rows[0]["protocol"] == "buzz"
    assert rows[0]["agent"] == "agent-a"
    assert rows[0]["source_domains"] == ["reddit", "x"]
    assert "Fine-tuning agents" in stdout
