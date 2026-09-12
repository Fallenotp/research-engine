from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from typing import Any

import pytest

from research_engine import evidence_gate, persistence, research_cli
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
        total_duration_ms=1350,
        total_cost_usd_estimate=0.0042,
    )


def _identity(session: ResearchSession) -> ResearchSession:
    return session


class _SufficientClient:
    provider_label = "fake"
    model_id = "fake"

    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = [
            {"verdict": "sufficient", "reason": "direct answer", "missing": []},
        ]

    def generate_json(self, _prompt: str) -> dict[str, Any]:
        return self.responses.pop(0)


def _patch_sufficient_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evidence_gate.sufficiency, "SUFFICIENCY_PREFILTER_ENABLED", False)
    monkeypatch.setattr(
        evidence_gate, "get_sufficiency_clients", lambda: (_SufficientClient(), None)
    )


def test_finalize_runs_verbatim_and_demotes_full_on_unsupported_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source(1, raw_text="The port is 5432.")
    path = tmp_path / "source.txt"
    path.write_text("The port is 5432.", encoding="utf-8")
    source = source.model_copy(update={"raw_text_path": path})
    session = _session(Protocol.SEARCH)
    session.sources = [source]
    session.evidence_chunks = [_chunk(source, "The port is 5432.")]
    session.question = "What is the default port?"
    session.answer = "The default port is 5433."
    _patch_sufficient_client(monkeypatch)

    result = persistence.finalize_session(session)

    assert result.answer_kind == AnswerKind.PARTIAL
    assert result.verbatim_check["unsupported_count"] == 1
    assert result.verbatim_check["unsupported"][0]["token"] == "5433"
    assert any("5433" in question for question in result.open_questions)


def test_finalize_keeps_full_when_all_tokens_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source(1, raw_text="The port is 5432.")
    path = tmp_path / "source.txt"
    path.write_text("The port is 5432.", encoding="utf-8")
    source = source.model_copy(update={"raw_text_path": path})
    session = _session(Protocol.SEARCH)
    session.sources = [source]
    session.evidence_chunks = [_chunk(source, "The port is 5432.")]
    session.question = "What is the default port?"
    session.answer = "The default port is 5432."
    _patch_sufficient_client(monkeypatch)

    result = persistence.finalize_session(session)

    assert result.answer_kind == AnswerKind.FULL
    assert result.verbatim_check["unsupported_count"] == 0


def test_finalize_does_not_demote_when_no_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(Protocol.SEARCH)
    session.evidence_chunks = []
    session.sources = [
        session.sources[0].model_copy(
            update={"raw_text_path": Path("/tmp/absent-verbatim-source.txt")}
        )
    ]
    _patch_sufficient_client(monkeypatch)

    result = persistence.finalize_session(session)

    assert result.verbatim_check["applicable"] is False
    assert not any("verbatim:" in question for question in result.open_questions)


def test_abstain_session_skips_verbatim() -> None:
    session = _session(Protocol.SEARCH)
    session.answer = None
    session.answer_kind = AnswerKind.ABSTAIN
    session.final_status = FinalStatus.INSUFFICIENT_EVIDENCE
    session.confidence = None
    session.answer_confidence = None
    session.open_questions = ["Which sources would answer this?"]

    result = persistence.finalize_session(session)

    assert result.verbatim_check is None or result.verbatim_check["applicable"] is False


def test_storage_failure_writes_gated_session_not_ungated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(Protocol.SEARCH)
    monkeypatch.setattr(persistence, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(
        evidence_gate,
        "score_session_sufficiency",
        lambda _session, *, source_texts: {
            "terminal_state": "unchecked",
            "proceed": False,
            "stop_reason": "checker_unavailable",
            "final_judge": {"checker_route": "unavailable", "fail_closed": True},
        },
    )
    monkeypatch.setattr(persistence, "write_session", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    saved, path = research_cli.save_session_with_fallback(session)

    assert path is not None
    persisted = ResearchSession.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.answer_kind == AnswerKind.ABSTAIN
    assert saved.evidence_gate_decision["gate_reason"] == "checker_unavailable"
    assert persisted.answer_kind == AnswerKind.ABSTAIN
    assert persisted.evidence_gate_decision["gate_reason"] == "checker_unavailable"


def test_storage_non_oserror_still_writes_gated_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(Protocol.SEARCH)
    monkeypatch.setattr(persistence, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(
        persistence,
        "write_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad serializer")),
    )

    saved, path = research_cli.save_session_with_fallback(session)

    assert saved is session
    assert path is not None
    assert ResearchSession.model_validate_json(path.read_text(encoding="utf-8")) == session


def test_answer_without_measured_confidence_requires_explicit_abstention() -> None:
    with pytest.raises(
        ValueError,
        match="an answer without a measured confidence has no answer_kind; abstain or measure",
    ):
        ResearchSession(
            protocol=Protocol.SEARCH,
            question="What happened?",
            final_status=FinalStatus.COMPLETE,
            answer="An unmeasured answer.",
            total_duration_ms=0,
            total_cost_usd_estimate=0.0,
        )


def test_gate_validation_error_never_direct_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(Protocol.SEARCH)
    monkeypatch.setattr(persistence, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(
        persistence, "finalize_session", lambda _session: (_ for _ in ()).throw(ValueError("bad session"))
    )

    saved, path = research_cli.save_session_with_fallback(session)

    assert saved is session
    assert path is None
    assert not list(tmp_path.rglob("*"))


def test_save_session_fail_closed_without_gemini_pro_record(tmp_path) -> None:
    session = _session(Protocol.RESEARCH)

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        with pytest.raises(GeminiProScoutError) as excinfo:
            save_session(session, root=tmp_path)

    assert "Gemini 3.8 Flash interlock failed closed" in str(excinfo.value)
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
    assert CANONICAL_GEMINI_PRO_MODEL_ID == "Gemini 3.8 Flash (Medium)"
    session = _session(
        Protocol.RESEARCH,
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.SCOUT,
                success=True,
                model_id="Gemini 3.8 Flash (Medium)",
            )
        ],
    )

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=tmp_path)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gemini_pro_runs"][0]["run_type"] == GeminiProRunKind.SCOUT.value
    assert payload["gemini_pro_runs"][0]["model_id"] == "Gemini 3.8 Flash (Medium)"


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


def test_save_session_creates_missing_default_root(monkeypatch, tmp_path) -> None:
    session = _session(Protocol.SEARCH)
    missing_root = tmp_path / "missing-research-sessions"
    monkeypatch.setattr(persistence, "DEFAULT_ROOT", missing_root)

    with patch("research_engine.persistence.enforce_evidence_gate", side_effect=_identity):
        path = save_session(session, root=persistence.DEFAULT_ROOT)

    assert missing_root.exists()
    assert path.exists()
