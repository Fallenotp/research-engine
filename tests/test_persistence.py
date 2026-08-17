from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from research_engine import persistence
from research_engine.dispatcher import GeminiProScoutError
from research_engine.persistence import CANONICAL_GEMINI_PRO_MODEL_ID, save_session
from research_engine.schema import (
    AnswerKind,
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


def _source(index: int, *, raw_text: str | None = None) -> SourceRecord:
    text = raw_text or f"source text {index}"
    return SourceRecord(
        url=f"https://source-{index}.example/article",
        domain=f"source-{index}.example",
        title=f"Source {index}",
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(text),
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=Path(f"/tmp/research-engine-persistence/source-{index}.txt"),
        char_count=len(text),
        tier=SourceTier.T1,
        topic_authority_score=1.0,
    )


def _chunk(source: SourceRecord, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        source_id=source.source_id,
        paragraph_text=text,
        char_offset=0,
        char_length=len(text),
        rerank_score=0.95,
        supports_claim=text[:120],
        crystal_check_passed=True,
        crystal_check_score=1.0,
    )


def _session(
    protocol: Protocol,
    *,
    gemini_pro_runs: list[GeminiProRunRecord] | None = None,
) -> ResearchSession:
    source = _source(1, raw_text="Canonical source text")
    chunk = _chunk(source, "Canonical source text")
    return ResearchSession(
        protocol=protocol,
        question=f"{protocol.value} test question",
        final_status=FinalStatus.COMPLETE,
        sources=[source],
        evidence_chunks=[chunk],
        rerank_passed_count=1,
        answer="A grounded answer.",
        answer_kind=AnswerKind.FULL,
        confidence=0.9,
        answer_confidence=0.9,
        gemini_pro_runs=gemini_pro_runs or [],
    )


def _identity(session: ResearchSession) -> ResearchSession:
    return session


def test_save_session_fail_closed_without_gemini_pro_record(tmp_path) -> None:
    session = _session(Protocol.RESEARCH)

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        with pytest.raises(GeminiProScoutError) as excinfo:
            save_session(session, root=tmp_path)

    assert "Gemini 3.7 Flash interlock failed closed" in str(excinfo.value)
    assert CANONICAL_GEMINI_PRO_MODEL_ID in str(excinfo.value)


def test_save_session_rejects_successful_non_pro_scout_record(tmp_path) -> None:
    session = _session(
        Protocol.RESEARCH,
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.SCOUT,
                success=True,
                model_id="gemini-pro",
            )
        ],
    )

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        with pytest.raises(GeminiProScoutError):
            save_session(session, root=tmp_path)


def test_save_session_accepts_live_agy_gemini_scout_record(tmp_path) -> None:
    assert CANONICAL_GEMINI_PRO_MODEL_ID == "Gemini 3.7 Flash (Medium)"
    session = _session(
        Protocol.RESEARCH,
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.SCOUT,
                success=True,
                model_id="Gemini 3.7 Flash (Medium)",
            )
        ],
    )

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=tmp_path)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gemini_pro_runs"][0]["run_type"] == GeminiProRunKind.SCOUT.value
    assert payload["gemini_pro_runs"][0]["model_id"] == "Gemini 3.7 Flash (Medium)"


def test_save_session_accepts_canonical_pro_synthesis_fallback_record(tmp_path) -> None:
    session = _session(
        Protocol.DEEP_RESEARCH,
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
                success=True,
                model_id="sonnet",
            )
        ],
    )

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=tmp_path)

    assert path.exists()


def test_save_session_accepts_final_synthesis_record_without_scout(tmp_path) -> None:
    session = _session(
        Protocol.RESEARCH,
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.FINAL_SYNTHESIS,
                success=True,
                model_id="sonnet",
            )
        ],
    )

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=tmp_path)

    assert path.exists()


def test_search_session_still_saves_without_gemini_pro_record(tmp_path) -> None:
    session = _session(Protocol.SEARCH)

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=tmp_path)

    assert path.exists()


def test_save_session_rejects_missing_default_root(monkeypatch, tmp_path) -> None:
    session = _session(Protocol.SEARCH)
    missing_root = tmp_path / "missing-research-sessions"
    monkeypatch.setattr(persistence, "DEFAULT_ROOT", missing_root)

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        with pytest.raises(FileNotFoundError) as excinfo:
            save_session(session, root=persistence.DEFAULT_ROOT)

    assert str(excinfo.value) == (
        "Research sessions root is not configured: missing "
        f"{missing_root}. Set RESEARCH_ENGINE_RESEARCH_SESSIONS_DIR."
    )
