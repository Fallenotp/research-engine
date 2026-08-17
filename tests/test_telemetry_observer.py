from __future__ import annotations

import csv
import json
from pathlib import Path

from research_engine import telemetry_observer, telemetry_to_csv


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_observer_and_csv_flow(tmp_path, monkeypatch) -> None:
    sessions_dir = tmp_path / "research-sessions"
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    monkeypatch.setattr(telemetry_observer, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)
    monkeypatch.setattr(telemetry_to_csv.telemetry_observer, "MASTER_LOG", master_log)

    clean_source = tmp_path / "sources" / "clean.txt"
    clean_source.parent.mkdir(parents=True, exist_ok=True)
    clean_source.write_text("Version 9.0 released.", encoding="utf-8")
    clean_session = {
        "session_id": "clean-1",
        "protocol": "/search",
        "question": "What version shipped?",
        "final_status": "complete",
        "answer": "Version 9.0 released.",
        "sources": [{"domain": "example.com", "raw_text_path": str(clean_source)}],
        "queries_run": [{"lane": "web", "worker_model": "haiku"}],
    }
    _write_json(sessions_dir / "2026-05-23" / "clean.json", clean_session)

    clean_summary = telemetry_observer.run()
    assert clean_summary["added"] == 1

    rows = [json.loads(line) for line in master_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "clean-1"
    assert rows[0]["flagged"] is False

    hallucinated_source = tmp_path / "sources" / "hallucinated.txt"
    hallucinated_source.write_text("Version 9 released on May 1, 2026.", encoding="utf-8")
    hallucinated_session = {
        "session_id": "hallucinated-1",
        "protocol": "/research",
        "question": "What version shipped?",
        "final_status": "complete",
        "answer": "Version 42 released on June 2, 2026.",
        "sources": [{"domain": "example.org", "raw_text_path": str(hallucinated_source)}],
        "territories": [{"assigned_worker_model": "haiku"}],
    }
    _write_json(sessions_dir / "2026-05-24" / "hallucinated.json", hallucinated_session)

    hallucinated_summary = telemetry_observer.run()
    assert hallucinated_summary["added"] == 1

    rows = [json.loads(line) for line in master_log.read_text(encoding="utf-8").splitlines()]
    hallucinated_row = next(row for row in rows if row["session_id"] == "hallucinated-1")
    assert hallucinated_row["flagged"] is True
    assert hallucinated_row["flagged_claims"]
    assert {claim["model"] for claim in hallucinated_row["flagged_claims"]} == {"haiku"}

    idempotent_summary = telemetry_observer.run()
    assert idempotent_summary["added"] == 0

    (sessions_dir / "2026-05-25").mkdir(parents=True, exist_ok=True)
    (sessions_dir / "2026-05-25" / "broken.json").write_text("{not-json", encoding="utf-8")
    _write_json(sessions_dir / "2026-05-25" / "missing-session-id.json", {"protocol": "/search"})

    malformed_summary = telemetry_observer.run()
    assert malformed_summary["added"] == 0
    assert malformed_summary["errors"] >= 1
    assert malformed_summary["skipped"] >= 3

    csv_result = telemetry_to_csv.main()
    csv_path = master_log.with_name("research-telemetry.csv")
    assert csv_result["rows"] == 2
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)

    assert reader.fieldnames == list(telemetry_observer.ROW_FIELDS)
    assert len(csv_rows) == 2


def test_run_reports_missing_sessions_root_as_not_configured(
    tmp_path, monkeypatch, caplog
) -> None:
    sessions_dir = tmp_path / "missing-research-sessions"
    master_log = tmp_path / "agent_state" / "research-telemetry.jsonl"
    master_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(telemetry_observer, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(telemetry_observer, "MASTER_LOG", master_log)

    with caplog.at_level("WARNING", logger="research_engine.telemetry_observer"):
        summary = telemetry_observer.run()

    assert summary == {
        "scanned": 0,
        "added": 0,
        "skipped": 0,
        "errors": 1,
        "master_log": str(master_log),
        "error": (
            "Research sessions root is not configured: missing "
            f"{sessions_dir}. Set RESEARCH_ENGINE_RESEARCH_SESSIONS_DIR."
        ),
    }
    assert summary["error"] in caplog.text
