from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


TELEMETRY_PATH = Path(
    os.environ.get("NO_BLUFF_TELEMETRY_PATH", "/Users/cleo/.claude/no-bluff-telemetry.jsonl")
)


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def record_tokens(record: dict) -> int:
    try:
        return int(record.get("tokens_in_approx", 0)) + int(record.get("tokens_out_approx", 0))
    except (TypeError, ValueError):
        return 0


def load_window_records(*, days: int, now: datetime | None = None) -> tuple[list[dict], datetime, datetime]:
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = end - timedelta(days=days)
    if not TELEMETRY_PATH.exists():
        return [], start, end

    records: list[dict] = []
    with TELEMETRY_PATH.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            timestamp = parse_timestamp(str(payload.get("ts") or ""))
            if timestamp is None or timestamp < start or timestamp > end:
                continue
            records.append(payload)
    return records, start, end


def build_summary(*, days: int, now: datetime | None = None) -> str:
    records, start, end = load_window_records(days=days, now=now)
    lines = [f"Period covered: {format_timestamp(start)} through {format_timestamp(end)}."]
    if not records:
        lines.append(f"No grounding/judge activity in the last {days} days.")
        return "\n".join(lines)

    total_calls = len(records)
    total_tokens = sum(record_tokens(record) for record in records)
    lines.append(f"Total activity: {total_calls} calls, about {total_tokens} approximate tokens.")
    lines.append("")
    lines.append("Breakdown by kind:")
    lines.append(f"{'kind':32} {'calls':>5} {'approx tokens':>14}")

    counts_by_kind: dict[str, int] = defaultdict(int)
    tokens_by_kind: dict[str, int] = defaultdict(int)
    for record in records:
        kind = str(record.get("kind") or "unknown")
        counts_by_kind[kind] += 1
        tokens_by_kind[kind] += record_tokens(record)

    for kind in sorted(counts_by_kind):
        lines.append(f"{kind:32} {counts_by_kind[kind]:>5} {tokens_by_kind[kind]:>14}")

    lines.append("")
    lines.append(f"Rough cost in calls: {total_calls} total model calls in this window.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize no-bluff grounding telemetry.")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        print(build_summary(days=max(0, args.days), now=now))
    except Exception:
        print(f"No grounding/judge activity in the last {max(0, args.days)} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
