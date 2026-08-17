"""Fail-closed evidence gate backed by shared query sufficiency checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from research_engine import sufficiency
from research_engine.schema import AnswerKind, FinalStatus, ResearchSession


ABSTAIN_OPEN_QUESTION = (
    "Which independent, on-topic sources directly answer this question?"
)

get_sufficiency_clients = sufficiency.get_sufficiency_clients


def enforce_evidence_gate(session: ResearchSession) -> ResearchSession:
    """Enforce query-aware source sufficiency for committed answers."""
    if not should_enforce_evidence_gate(session):
        return session

    source_texts = collect_session_source_texts(session)
    if not source_texts:
        decision = _empty_evidence_decision(session)
        session.evidence_gate_decision = decision
        _clear_evidence_as_failed(session, total_chunks=len(session.evidence_chunks))
        return force_abstain(session, reason="empty_evidence_pool")

    result = score_session_sufficiency(session, source_texts=source_texts)
    session.evidence_gate_decision = result
    terminal_state = str(result.get("terminal_state") or "exhausted")

    if terminal_state == "sufficient":
        result["overridden"] = False
        result["reason"] = result.get("reason") or "passed"
        session.evidence_gate_decision = result
        session.rerank_passed_count = len(session.evidence_chunks)
        session.rerank_failed_count = 0
        _prune_invalid_chunk_references(session)
        return session

    if terminal_state == "partial":
        return mark_partial(session, result)

    _clear_evidence_as_failed(session, total_chunks=len(session.evidence_chunks))
    return force_abstain(session, reason="sufficiency_exhausted")


def should_enforce_evidence_gate(session: ResearchSession) -> bool:
    answer_kind = _enum_value(session.answer_kind)
    if answer_kind is None and session.answer:
        answer_kind = AnswerKind.FULL.value
    terminal_commit_statuses = {
        FinalStatus.COMPLETE.value,
        FinalStatus.WEAK_SOURCES.value,
        FinalStatus.INSUFFICIENT_EVIDENCE.value,
    }
    return (
        answer_kind in {AnswerKind.FULL.value, AnswerKind.PARTIAL.value}
        and _enum_value(session.final_status) in terminal_commit_statuses
    )


def score_session_sufficiency(
    session: ResearchSession,
    *,
    source_texts: list[sufficiency.SourceText] | None = None,
) -> dict[str, Any]:
    return sufficiency.run_sufficiency_loop(
        query=session.question,
        source_texts=source_texts if source_texts is not None else collect_session_source_texts(session),
        clients=get_sufficiency_clients(),
    )


def collect_session_source_texts(session: ResearchSession) -> list[sufficiency.SourceText]:
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in session.evidence_chunks:
        chunks_by_source.setdefault(str(chunk.source_id), []).append(chunk.paragraph_text)

    collected: list[sufficiency.SourceText] = []
    remaining_chars = sufficiency.MAX_SOURCE_CHARS
    for source in session.sources:
        source_id = str(source.source_id)
        chunk_text = "\n\n".join(chunks_by_source.get(source_id, []))
        text = _read_raw_source_text(source.raw_text_path) or chunk_text
        text = " ".join(text.split())
        if not text:
            continue
        capped = text[: min(sufficiency.MAX_SOURCE_CHARS_PER_SOURCE, remaining_chars)]
        if not capped:
            break
        collected.append(
            sufficiency.SourceText(
                source_id=source_id,
                title=source.title,
                domain=source.domain,
                text=capped,
            )
        )
        remaining_chars -= len(capped)
        if remaining_chars <= 0:
            break
    return collected


def mark_partial(session: ResearchSession, decision: dict[str, Any]) -> ResearchSession:
    decision.update(
        {
            "overridden": False,
            "low_confidence": True,
            "answer_kind_before_override": _enum_value(session.answer_kind),
            "final_status_before_override": _enum_value(session.final_status),
        }
    )
    session.evidence_gate_decision = decision
    session.final_status = FinalStatus.WEAK_SOURCES
    session.answer_kind = AnswerKind.PARTIAL
    measured_confidence = session.confidence
    session.confidence = 0.0 if measured_confidence is None else min(float(measured_confidence), 0.5)
    measured_answer_confidence = session.answer_confidence
    if measured_answer_confidence is None:
        session.answer_confidence = session.confidence
    else:
        session.answer_confidence = min(float(measured_answer_confidence), 0.5)
    _append_open_question(session, sufficiency.low_confidence_reason_for_result(decision))
    _prune_invalid_chunk_references(session)
    return session


def force_abstain(session: ResearchSession, *, reason: str) -> ResearchSession:
    decision = session.evidence_gate_decision or _empty_evidence_decision(session)
    decision.update(
        {
            "overridden": True,
            "gate_reason": reason,
            "final_status_before_override": _enum_value(session.final_status),
            "answer_kind_before_override": _enum_value(session.answer_kind),
        }
    )
    session.evidence_gate_decision = decision
    session.final_status = FinalStatus.INSUFFICIENT_EVIDENCE
    session.answer_kind = AnswerKind.ABSTAIN
    session.answer = None
    session.confidence = 0.0
    session.answer_confidence = 0.0
    if not session.open_questions:
        session.open_questions.append(ABSTAIN_OPEN_QUESTION)
    _prune_invalid_chunk_references(session)
    return session


def evidence_gate_sidecar_path(session_path: Path) -> Path:
    return session_path.parent / "evidence-gate-decision.json"


def evidence_gate_session_sidecar_path(session_path: Path) -> Path:
    return session_path.with_name(f"{session_path.stem}.evidence-gate-decision.json")


def write_evidence_gate_sidecar(session: ResearchSession, session_path: Path) -> Path | None:
    if session.evidence_gate_decision is None:
        return None
    payload = json.dumps(session.evidence_gate_decision, indent=2, sort_keys=True)
    if not payload.endswith("\n"):
        payload += "\n"
    primary_path = evidence_gate_sidecar_path(session_path)
    _write_sidecar_payload(primary_path, payload)
    mirror_path = evidence_gate_session_sidecar_path(session_path)
    _write_sidecar_payload(mirror_path, payload)
    return primary_path


def _empty_evidence_decision(session: ResearchSession) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mode": "query_sufficiency_loop",
        "original_query": session.question,
        "terminal_state": "exhausted",
        "verdict": "insufficient",
        "reason": "empty_evidence_pool",
        "missing": ["No raw source text was available."],
        "proceed": False,
        "low_confidence": True,
        "stop_reason": "empty_evidence_pool",
        "reformulations": [],
        "attempts": [],
        "self_synthesized_answer_used": False,
        "input_guardrail": "original_query_and_raw_sources_only",
    }


def _read_raw_source_text(path: Path) -> str:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def _append_open_question(session: ResearchSession, text: str) -> None:
    clean = " ".join((text or "").split())
    if not clean:
        clean = ABSTAIN_OPEN_QUESTION
    if clean not in session.open_questions:
        session.open_questions.append(clean)


def _clear_evidence_as_failed(session: ResearchSession, *, total_chunks: int) -> None:
    session.evidence_chunks = []
    session.rerank_passed_count = 0
    session.rerank_failed_count = total_chunks


def _write_sidecar_payload(path: Path, payload: str) -> None:
    tmp_path = Path(f"{path}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _prune_invalid_chunk_references(session: ResearchSession) -> None:
    chunk_ids = {chunk.chunk_id for chunk in session.evidence_chunks}
    session.agent_disagreements = [
        disagreement
        for disagreement in session.agent_disagreements
        if all(
            chunk_id in chunk_ids
            for chunk_id in disagreement.agent_a_evidence + disagreement.agent_b_evidence
        )
    ]
    session.cross_model_verifications = [
        verification
        for verification in session.cross_model_verifications
        if all(chunk_id in chunk_ids for chunk_id in verification.grounding_chunks)
    ]
