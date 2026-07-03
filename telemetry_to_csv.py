from __future__ import annotations

import csv
import json

try:
    from research_engine import telemetry_observer
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import telemetry_observer


def _read_jsonl_rows(path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _stringify_cell(field: str, value: object) -> object:
    if field not in {
        "source_domains",
        "lanes_used",
        "lanes_failed",
        "models_used",
        "flagged_claims",
    }:
        return value
    if not isinstance(value, list):
        return ""
    if field == "flagged_claims":
        return ";".join(json.dumps(item, sort_keys=True) for item in value)
    return ";".join(str(item) for item in value)


def _write_csv(path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _stringify_cell(field, row.get(field)) for field in fieldnames}
            )


def main() -> dict[str, object]:
    master_log = telemetry_observer.MASTER_LOG
    csv_path = master_log.with_name("research-telemetry.csv")
    rows = _read_jsonl_rows(master_log)
    _write_csv(csv_path, rows, list(telemetry_observer.ROW_FIELDS))

    result = {"rows": len(rows), "csv_path": str(csv_path)}
    call_log = telemetry_observer.CALL_LOG
    if call_log.exists():
        call_rows = _read_jsonl_rows(call_log)
        call_csv_path = master_log.with_name("research-call-log.csv")
        _write_csv(call_csv_path, call_rows, list(telemetry_observer.CALL_LOG_FIELDS))
        result["call_rows"] = len(call_rows)
        result["call_csv_path"] = str(call_csv_path)

    print(result)
    return result


if __name__ == "__main__":
    main()
