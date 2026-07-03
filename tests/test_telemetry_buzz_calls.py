from __future__ import annotations

import csv
import json
from pathlib import Path

from research_engine import telemetry_observer, telemetry_to_csv


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_buzz_writes_one_row(tmp_path, monkeypatch) -> None:
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)

    telemetry_observer.log_buzz(
        "Fine-tuning agents",
        n_signals=3,
        platforms_with_data=["reddit", "x"],
        agent="agent-a",
    )

    rows = _read_jsonl(master_log)
    assert len(rows) == 1
    assert rows[0]["protocol"] == "buzz"
    assert rows[0]["source"] == "buzz"
    assert rows[0]["agent"] == "agent-a"
    assert rows[0]["question"] == "Fine-tuning agents"
    assert rows[0]["n_sources"] == 3
    assert rows[0]["source_domains"] == ["reddit", "x"]
    assert rows[0]["lanes_used"] == ["reddit", "x"]
    assert rows[0]["final_status"] == "complete"


def test_log_buzz_weird_args_do_not_raise(tmp_path, monkeypatch) -> None:
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)

    telemetry_observer.log_buzz(
        object(),
        n_signals=object(),
        platforms_with_data=object(),
        agent=object(),
    )


def test_summarize_calls_aggregates_by_lane(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(telemetry_observer, "CALL_LOG", call_log)
    call_log.parent.mkdir(parents=True, exist_ok=True)
    call_log.write_text(
        "\n".join(
            [
                json.dumps({"lane": "tavily", "ok": True, "duration_ms": 100}),
                json.dumps({"lane": "tavily", "ok": True, "duration_ms": 200}),
                json.dumps({"lane": "tavily", "ok": False, "duration_ms": 300}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = telemetry_observer.summarize_calls()

    assert summary["tavily"]["calls"] == 3
    assert summary["tavily"]["ok"] == 2
    assert summary["tavily"]["failed"] == 1
    assert summary["tavily"]["avg_ms"] == 200.0


def test_call_log_csv_export_creates_file(tmp_path, monkeypatch) -> None:
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)
    monkeypatch.setattr(telemetry_observer, "CALL_LOG", call_log)
    monkeypatch.setattr(telemetry_to_csv.telemetry_observer, "MASTER_LOG", master_log)
    monkeypatch.setattr(telemetry_to_csv.telemetry_observer, "CALL_LOG", call_log)

    call_log.parent.mkdir(parents=True, exist_ok=True)
    call_log.write_text(
        "\n".join(
            [
                json.dumps({"lane": "tavily", "ok": True, "duration_ms": 100, "agent": "a"}),
                json.dumps({"lane": "serper", "ok": False, "duration_ms": 50, "agent": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = telemetry_to_csv.main()
    call_csv_path = master_log.with_name("research-call-log.csv")

    assert call_csv_path.exists()
    assert result["call_rows"] == 2
    assert result["call_csv_path"] == str(call_csv_path)

    with call_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(telemetry_observer.CALL_LOG_FIELDS)
    assert len(rows) == 2


def test_call_log_csv_export_skips_missing_file(tmp_path, monkeypatch) -> None:
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    missing_call_log = tmp_path / "agent_state" / "missing-call-log.jsonl"
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)
    monkeypatch.setattr(telemetry_observer, "CALL_LOG", missing_call_log)
    monkeypatch.setattr(telemetry_to_csv.telemetry_observer, "MASTER_LOG", master_log)
    monkeypatch.setattr(telemetry_to_csv.telemetry_observer, "CALL_LOG", missing_call_log)

    result = telemetry_to_csv.main()

    assert result["rows"] == 0
    assert result["csv_path"] == str(master_log.with_name("research-telemetry.csv"))
    assert "call_rows" not in result
    assert not master_log.with_name("research-call-log.csv").exists()
