from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import research_engine.grounding_telemetry_summary as summary


def write_telemetry(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_summary_reports_counts_and_tokens(tmp_path: Path, monkeypatch, capsys) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    now = datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc)
    write_telemetry(
        telemetry_path,
        [
            {
                "ts": "2026-05-27T21:00:00Z",
                "kind": "gate_judge",
                "tokens_in_approx": 100,
                "tokens_out_approx": 25,
            },
            {
                "ts": "2026-05-26T20:00:00Z",
                "kind": "grounding_synth",
                "tokens_in_approx": 200,
                "tokens_out_approx": 50,
            },
            {
                "ts": "2026-05-25T19:00:00Z",
                "kind": "grounding_escalation_grok",
                "tokens_in_approx": 80,
                "tokens_out_approx": 20,
            },
        ],
    )
    monkeypatch.setattr(summary, "TELEMETRY_PATH", telemetry_path)

    exit_code = summary.main(["--days", "7"], now=now)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Period covered: 2026-05-20 22:00 UTC through 2026-05-27 22:00 UTC." in output
    assert "Total activity: 3 calls, about 475 approximate tokens." in output
    assert "gate_judge" in output
    assert "grounding_synth" in output
    assert "grounding_escalation_grok" in output
    assert "Rough cost in calls: 3 total model calls in this window." in output


def test_summary_missing_file_is_plain(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(summary, "TELEMETRY_PATH", tmp_path / "missing.jsonl")

    exit_code = summary.main(["--days", "7"], now=datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "No grounding/judge activity in the last 7 days." in output


def test_summary_respects_day_window(tmp_path: Path, monkeypatch, capsys) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    now = datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc)
    write_telemetry(
        telemetry_path,
        [
            {
                "ts": "2026-05-27T20:00:00Z",
                "kind": "gate_judge",
                "tokens_in_approx": 40,
                "tokens_out_approx": 10,
            },
            {
                "ts": "2026-05-10T20:00:00Z",
                "kind": "grounding_synth",
                "tokens_in_approx": 300,
                "tokens_out_approx": 100,
            },
        ],
    )
    monkeypatch.setattr(summary, "TELEMETRY_PATH", telemetry_path)

    exit_code = summary.main(["--days", "7"], now=now)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Total activity: 1 calls, about 50 approximate tokens." in output
    assert "grounding_synth" not in output
