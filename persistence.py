from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from pydantic import ValidationError

from research_engine.evidence_gate import enforce_evidence_gate, write_evidence_gate_sidecar

from . import paths
from research_engine.schema import (
    EvidenceChunk,
    ExtractionMethod,
    FinalStatus,
    GeminiProRunKind,
    GeminiProRunRecord,
    Protocol,
    ResearchSession,
    SourceRecord,
    SourceTier,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_ROOT = paths.optional_path(paths.RESEARCH_SESSIONS_DIR_ENV) or paths.data_path(
    "research-sessions"
)
INDEX_KEYS = {"session_id", "created_at", "protocol", "question", "final_status"}
CANONICAL_GEMINI_PRO_MODEL_ID = "Gemini 3.7 Flash (Medium)"
_AGY_GEMINI_FLASH_MODEL_PREFIX = "Gemini 3.7 Flash"


def save_session(
    session: "ResearchSession",
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Persist session to root/YYYY-MM-DD/{session_id}.json."""
    root = root.expanduser().resolve()
    session.updated_at = datetime.now(timezone.utc)
    session = enforce_evidence_gate(session)
    session = enforce_gemini_pro_interlock(session)
    try:
        ResearchSession.model_validate(session.model_dump(mode="python"))
    except ValidationError as exc:
        raise ValidationError.from_exception_data(
            f"ResearchSession(session_id={session.session_id})",
            exc.errors(),
        ) from exc

    path = session.to_jsonl_path(root).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp")
    payload = session.model_dump_json(indent=2)
    if not payload.endswith("\n"):
        payload += "\n"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)
    write_evidence_gate_sidecar(session, path)
    return path


def load_session(
    session_id: Union[str, UUID],
    root: Path = DEFAULT_ROOT,
) -> "ResearchSession":
    """Find and load session by id, searching all date subdirs."""
    root = root.expanduser().resolve()
    session_str = str(session_id)
    matches = sorted(root.glob(f"*/{session_str}.json"))
    if not matches:
        searched = root / "*" / f"{session_str}.json"
        raise FileNotFoundError(f"Session {session_str} not found; searched {searched}")
    return ResearchSession.model_validate_json(matches[0].read_text(encoding="utf-8"))


def list_sessions(
    root: Path = DEFAULT_ROOT,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    protocol: Optional[str] = None,
    final_status: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Return a lightweight newest-first index of saved sessions."""
    root = root.expanduser().resolve()
    if limit <= 0 or not root.exists():
        return []

    since = _normalize_filter_datetime(since)
    until = _normalize_filter_datetime(until)
    entries: list[dict] = []
    for path in sorted(root.glob("*/*.json"), reverse=True):
        try:
            entry = _read_index_entry(path.resolve())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable session file %s: %s", path, exc)
            continue
        created_at = entry["created_at"]
        if since is not None and created_at < since:
            continue
        if until is not None and created_at > until:
            continue
        if protocol is not None and entry["protocol"] != protocol:
            continue
        if final_status is not None and entry["final_status"] != final_status:
            continue
        entries.append(entry)

    entries.sort(key=lambda item: item["created_at"], reverse=True)
    return entries[:limit]


def delete_session(
    session_id: Union[str, UUID],
    root: Path = DEFAULT_ROOT,
) -> bool:
    """Soft-delete a session by renaming it with a .deleted suffix."""
    root = root.expanduser().resolve()
    session_str = str(session_id)
    matches = sorted(root.glob(f"*/{session_str}.json"))
    if not matches:
        return False
    path = matches[0].resolve()
    os.replace(path, Path(f"{path}.deleted"))
    return True


def enforce_gemini_pro_interlock(session: ResearchSession) -> ResearchSession:
    if not should_enforce_gemini_pro_interlock(session):
        return session
    if _has_successful_gemini_flash_scout(session):
        return session
    if _has_successful_final_synthesis_run(session):
        return session

    from research_engine.dispatcher import GeminiProScoutError

    protocol = getattr(session.protocol, "value", str(session.protocol))
    final_status = getattr(session.final_status, "value", str(session.final_status))
    detail = _describe_gemini_pro_runs(session)
    raise GeminiProScoutError(
        "Gemini 3.7 Flash interlock failed closed: "
        f"{protocol} session {session.session_id} with final_status={final_status!r} "
        "cannot be saved without a successful Gemini 3.7 Flash scout record or a "
        "successful final-synthesis fallback record. "
        "Scout records must use a live agy Gemini 3.7 Flash model id, such as "
        f"{CANONICAL_GEMINI_PRO_MODEL_ID!r}. "
        f"Recorded runs: {detail}."
    )


def should_enforce_gemini_pro_interlock(session: ResearchSession) -> bool:
    return (
        session.protocol in (Protocol.RESEARCH, Protocol.DEEP_RESEARCH)
        and session.final_status != FinalStatus.IN_PROGRESS
    )


def _normalize_filter_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_index_entry(path: Path) -> dict:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith('"territories"'):
                break
            if not stripped.startswith('"'):
                continue
            key_raw, separator, value_raw = stripped.partition(":")
            if not separator:
                continue
            key = json.loads(key_raw)
            if key not in INDEX_KEYS:
                continue
            values[key] = json.loads(value_raw.rstrip(",\n"))
            if values.keys() >= INDEX_KEYS:
                break
    missing = INDEX_KEYS - values.keys()
    if missing:
        raise ValueError(f"missing index keys {sorted(missing)}")
    return {
        "session_id": values["session_id"],
        "created_at": _parse_datetime(values["created_at"]),
        "protocol": values["protocol"],
        "question": values["question"],
        "final_status": values["final_status"],
        "path": str(path),
    }


def _has_successful_gemini_flash_scout(session: ResearchSession) -> bool:
    for run in session.gemini_pro_runs:
        if (
            run.run_type == GeminiProRunKind.SCOUT
            and run.success
            and _is_agy_gemini_flash_model(run.model_id)
        ):
            return True
    return False


def _is_agy_gemini_flash_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    return model_id == _AGY_GEMINI_FLASH_MODEL_PREFIX or model_id.startswith(
        f"{_AGY_GEMINI_FLASH_MODEL_PREFIX} ("
    )


def _has_successful_final_synthesis_run(session: ResearchSession) -> bool:
    return any(
        run.run_type in {
            GeminiProRunKind.FINAL_SYNTHESIS,
            GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
        }
        and run.success
        and run.model_id
        for run in session.gemini_pro_runs
    )


def _describe_gemini_pro_runs(session: ResearchSession) -> str:
    if not session.gemini_pro_runs:
        return "none"
    parts: list[str] = []
    for run in session.gemini_pro_runs:
        model_id = run.model_id or "unknown-model"
        state = "success" if run.success else "failed"
        reason = f" reason={run.failure_reason}" if run.failure_reason else ""
        parts.append(f"{run.run_type.value}:{state}:{model_id}{reason}")
    return "; ".join(parts)


def _build_self_test_session() -> ResearchSession:
    now = datetime.now(timezone.utc)
    source = SourceRecord(
        url="https://example.com/research",
        domain="example.com",
        title="Example Source",
        fetched_at=now,
        content_hash=SourceRecord.hash_text("source text"),
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=Path("/tmp/research-engine-draft/source.txt"),
        char_count=11,
        tier=SourceTier.T1,
        topic_authority_score=1.0,
    )
    chunk = EvidenceChunk(
        source_id=source.source_id,
        paragraph_text="source text",
        char_offset=0,
        char_length=11,
        rerank_score=0.95,
        supports_claim="A minimal claim",
        crystal_check_passed=True,
        crystal_check_score=1.0,
    )
    return ResearchSession(
        protocol=Protocol.RESEARCH,
        question="Minimal self-test question",
        final_status=FinalStatus.COMPLETE,
        sources=[source],
        evidence_chunks=[chunk],
        rerank_passed_count=1,
        answer="x",
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.SCOUT,
                success=True,
                model_id=CANONICAL_GEMINI_PRO_MODEL_ID,
            )
        ],
    )


if __name__ == "__main__":
    ok = False
    detail = ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            session = _build_self_test_session()
            path = save_session(session, root=root)
            saved_exists = path.exists()
            loaded = load_session(session.session_id, root=root)
            listed = list_sessions(root=root)
            deleted = delete_session(session.session_id, root=root)
            remaining = list_sessions(root=root)
            ok = all(
                [
                    path.is_absolute(),
                    saved_exists,
                    loaded.model_dump() == session.model_dump(),
                    len(listed) == 1,
                    listed[0]["session_id"] == str(session.session_id),
                    deleted is True,
                    remaining == [],
                ]
            )
            detail = f"path={path}"
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
    print(f"{'PASS' if ok else 'FAIL'} persistence self-test {detail}")
