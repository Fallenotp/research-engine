from __future__ import annotations

import json
import time

from research_engine import extractor


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_appends_reader_telemetry_success_and_failure(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "reader-telemetry.jsonl"
    monkeypatch.setattr(extractor, "READER_TELEMETRY_LOG", log_path)

    extractor._log("jina", time.perf_counter(), True, 500)
    extractor._log("scrapling", time.perf_counter(), False, 0, "timeout")

    rows = _rows(log_path)
    assert len(rows) == 2
    assert rows[0]["method"] == "jina"
    assert rows[0]["status"] == "success"
    assert "error" not in rows[0]
    assert rows[1]["status"] == "fail"
    assert rows[1]["error"] == "timeout"


def test_log_omits_empty_error_from_reader_telemetry(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "reader-telemetry.jsonl"
    monkeypatch.setattr(extractor, "READER_TELEMETRY_LOG", log_path)

    extractor._log("crawlee", time.perf_counter(), False, 0, "")

    rows = _rows(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "fail"
    assert "error" not in rows[0]
