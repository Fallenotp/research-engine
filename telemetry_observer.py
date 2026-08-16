from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

try:
    from research_engine.verbatim_check import check_verbatim
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from verbatim_check import check_verbatim


SESSIONS_DIR = paths.optional_path(paths.RESEARCH_SESSIONS_DIR_ENV) or paths.data_path(
    "research-sessions"
)
MASTER_LOG = paths.telemetry_path("research-telemetry.jsonl")
CALL_LOG = paths.telemetry_path("research-call-log.jsonl")
ROW_FIELDS = (
    "ts",
    "run_ts",
    "protocol",
    "agent",
    "source",
    "question",
    "final_status",
    "answer_kind",
    "confidence",
    "n_sources",
    "source_domains",
    "n_queries",
    "lanes_used",
    "lanes_failed",
    "models_used",
    "flagged",
    "flagged_claims",
    "cost_usd",
    "duration_ms",
    "iteration_count",
    "session_id",
    "session_file",
)
CALL_LOG_FIELDS = (
    "ts",
    "protocol",
    "topic",
    "lane",
    "ok",
    "duration_ms",
    "result_count",
    "error",
    "agent",
)
logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sorted_unique(values: list[Any]) -> list[Any]:
    return sorted({str(value) for value in values if value})


def _sorted_strings(values: Any) -> list[str]:
    if isinstance(values, str):
        return sorted([values]) if values else []
    if isinstance(values, (list, tuple, set)):
        return sorted(str(value) for value in values if value)
    return []


def audit_answer(session_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        answer = session_dict.get("answer")
        if not answer:
            return {"flagged": False, "claims": []}

        source_texts: list[str] = []
        for source in _as_list(session_dict.get("sources")):
            if not isinstance(source, dict):
                continue
            raw_path = source.get("raw_text_path")
            if raw_path and os.path.exists(raw_path):
                with open(raw_path, "r", encoding="utf-8", errors="ignore") as handle:
                    source_texts.append(handle.read())

        models_used = _sorted_unique(
            [
                territory.get("assigned_worker_model")
                for territory in _as_list(session_dict.get("territories"))
                if isinstance(territory, dict)
            ]
            + [
                query.get("worker_model")
                for query in _as_list(session_dict.get("queries_run"))
                if isinstance(query, dict)
            ]
        )
        result = check_verbatim(str(answer), source_texts)
        unverified = [item.token for item in result.unsupported]
        model = models_used[0] if len(models_used) == 1 else "unknown"
        return {
            "flagged": bool(unverified),
            "claims": [
                {"claim": token, "model": model, "check": "verbatim"}
                for token in unverified
            ],
        }
    except Exception as exc:
        return {"flagged": False, "claims": [], "audit_error": str(exc)}


def session_to_row(session_dict: dict[str, Any], session_file: Path) -> dict[str, Any]:
    sources = [
        source for source in _as_list(session_dict.get("sources")) if isinstance(source, dict)
    ]
    queries_run = [
        query for query in _as_list(session_dict.get("queries_run")) if isinstance(query, dict)
    ]
    territories = [
        territory
        for territory in _as_list(session_dict.get("territories"))
        if isinstance(territory, dict)
    ]
    gemini_pro_runs = [
        run for run in _as_list(session_dict.get("gemini_pro_runs")) if isinstance(run, dict)
    ]
    audit = audit_answer(session_dict)

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_ts": session_dict.get("updated_at") or session_dict.get("created_at"),
        "protocol": session_dict.get("protocol"),
        "agent": session_dict.get("triggered_by") or "unknown",
        "source": "observer",
        "question": str(session_dict.get("question") or "")[:200],
        "final_status": session_dict.get("final_status"),
        "answer_kind": session_dict.get("answer_kind"),
        "confidence": session_dict.get("confidence"),
        "n_sources": len(sources),
        "source_domains": _sorted_unique([source.get("domain") for source in sources]),
        "n_queries": len(queries_run),
        "lanes_used": _sorted_unique([query.get("lane") for query in queries_run]),
        "lanes_failed": _sorted_unique(
            [
                query.get("lane")
                for query in queries_run
                if query.get("lane") and query.get("error")
            ]
        ),
        "models_used": _sorted_unique(
            [territory.get("assigned_worker_model") for territory in territories]
            + [query.get("worker_model") for query in queries_run]
            + [run.get("run_type") for run in gemini_pro_runs]
        ),
        "flagged": audit.get("flagged", False),
        "flagged_claims": audit.get("claims", []),
        "cost_usd": session_dict.get("total_cost_usd_estimate"),
        "duration_ms": session_dict.get("total_duration_ms"),
        "iteration_count": session_dict.get("iteration_count"),
        "session_id": session_dict.get("session_id"),
        "session_file": str(session_file),
    }
    return {field: row.get(field) for field in ROW_FIELDS}


def _append_row(row: dict[str, Any]) -> Exception | None:
    try:
        MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with MASTER_LOG.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return None
    except Exception as exc:
        logger.warning("telemetry append failed: %s", exc)
        return exc


def log_buzz(
    topic, *, n_signals=0, platforms_with_data=None, agent=None
) -> None:
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        platforms = _sorted_strings(platforms_with_data or [])
        row = {
            "ts": now_iso,
            "run_ts": now_iso,
            "protocol": "buzz",
            "agent": (
                agent
                or os.environ.get("RESEARCH_AGENT")
                or os.environ.get("CLAUDE_AGENT")
                or "unknown"
            ),
            "source": "buzz",
            "question": str(topic)[:200],
            "final_status": "complete",
            "answer_kind": None,
            "confidence": None,
            "n_sources": n_signals,
            "source_domains": platforms,
            "n_queries": 0,
            "lanes_used": platforms,
            "lanes_failed": [],
            "models_used": [],
            "flagged": False,
            "flagged_claims": [],
            "cost_usd": 0.0,
            "duration_ms": 0,
            "iteration_count": 0,
            "session_id": None,
            "session_file": None,
        }
        error = _append_row({field: row.get(field) for field in ROW_FIELDS})
        if error is not None:
            logger.warning("buzz telemetry append failed: %s", error)
    except Exception as exc:
        logger.warning("buzz telemetry append failed: %s", exc)


def existing_session_ids() -> set[str]:
    if not MASTER_LOG.exists():
        return set()

    seen: set[str] = set()
    try:
        with MASTER_LOG.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    session_id = record.get("session_id")
                    if session_id:
                        seen.add(str(session_id))
    except Exception:
        return set()
    return seen


def _read_jsonl_dict_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def summarize_calls(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None
) -> dict[str, dict[str, float | int]]:
    try:
        buckets: dict[str, dict[str, float | int]] = {}
        for row in rows if rows is not None else _read_jsonl_dict_rows(CALL_LOG):
            if not isinstance(row, dict):
                continue
            lane = str(row.get("lane") or "").strip()
            if not lane:
                continue
            duration_ms = row.get("duration_ms")
            try:
                duration_value = float(duration_ms or 0)
            except (TypeError, ValueError):
                duration_value = 0.0
            bucket = buckets.setdefault(
                lane,
                {"calls": 0, "ok": 0, "failed": 0, "avg_ms": 0.0, "_total_ms": 0.0},
            )
            bucket["calls"] += 1
            bucket["_total_ms"] += duration_value
            if row.get("ok") is True:
                bucket["ok"] += 1
            else:
                bucket["failed"] += 1

        summary: dict[str, dict[str, float | int]] = {}
        for lane, bucket in buckets.items():
            calls = int(bucket["calls"])
            total_ms = float(bucket["_total_ms"])
            summary[lane] = {
                "calls": calls,
                "ok": int(bucket["ok"]),
                "failed": int(bucket["failed"]),
                "avg_ms": (total_ms / calls) if calls else 0.0,
            }
        return summary
    except Exception:
        return {}


def format_call_summary(summary: dict[str, dict[str, float | int]]) -> str:
    if not summary:
        return "No call rows found."

    lines: list[str] = []
    for lane, bucket in sorted(summary.items()):
        calls = int(bucket["calls"])
        ok = int(bucket["ok"])
        failed = int(bucket["failed"])
        avg_ms = float(bucket["avg_ms"])
        success_rate = (ok / calls * 100.0) if calls else 0.0
        lines.append(
            f"{lane}: calls={calls} ok={ok} failed={failed} "
            f"success_rate={success_rate:.1f}% avg_ms={avg_ms:.1f}"
        )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_session_ids()
    scanned = 0
    added = 0
    skipped = 0
    errors = 0

    for session_file in sorted(SESSIONS_DIR.rglob("*.json")):
        if session_file.name.endswith("evidence-gate-decision.json"):
            continue
        try:
            with session_file.open("r", encoding="utf-8") as handle:
                session_dict = json.load(handle)
            if not isinstance(session_dict, dict):
                skipped += 1
                scanned += 1
                continue
            session_id = session_dict.get("session_id")
            if not session_id:
                skipped += 1
                scanned += 1
                continue
            session_id_str = str(session_id)
            if session_id_str in seen:
                skipped += 1
                scanned += 1
                continue
            _append_row(session_to_row(session_dict, session_file))
            seen.add(session_id_str)
            added += 1
        except Exception:
            errors += 1
        scanned += 1

    summary = {
        "scanned": scanned,
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "master_log": str(MASTER_LOG),
    }
    print(summary)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--summarize-calls", action="store_true")
    actions.add_argument("--log-buzz", metavar="TOPIC")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    parser.add_argument("--n-signals", type=int, default=0)
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--agent")
    args = parser.parse_args(argv)

    if args.log_buzz is not None:
        log_buzz(
            args.log_buzz,
            n_signals=args.n_signals,
            platforms_with_data=args.platform,
            agent=args.agent,
        )
        result = {
            "logged": True,
            "topic": args.log_buzz,
            "n_signals": args.n_signals,
            "platforms": _sorted_strings(args.platform),
            "master_log": str(MASTER_LOG),
        }
        print(json.dumps(result, sort_keys=True) if args.emit_json else result)
        return result

    if args.summarize_calls:
        summary = summarize_calls()
        if args.emit_json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(format_call_summary(summary))
        return summary

    return run()


if __name__ == "__main__":
    main()
