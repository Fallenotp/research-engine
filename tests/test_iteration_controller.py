from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import Mock, patch

from research_engine import evidence_gate
from research_engine.iteration_controller import (
    RecursionTierConfig,
    RecursiveNode,
    RecursivePlan,
    expand_recursive_node,
    judge_findings,
)
from research_engine.schema import (
    AnswerKind,
    EvidenceChunk,
    ExtractionMethod,
    FinalStatus,
    Protocol,
    ResearchSession,
    SourceRecord,
    SourceTier,
)


class FakeCheckerClient:
    def __init__(
        self,
        label: str,
        model_id: str,
        responses: Iterable[dict[str, Any] | Exception],
    ) -> None:
        self.provider_label = label
        self.model_id = model_id
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError(f"{self.provider_label} had no scripted response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _source(index: int, *, raw_text: str | None = None) -> SourceRecord:
    text = raw_text or f"source text {index}"
    return SourceRecord(
        url=f"https://source-{index}.example/article",
        domain=f"source-{index}.example",
        title=f"Source {index}",
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(text),
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=Path(f"/tmp/research-engine-iteration-controller/source-{index}.txt"),
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
    question: str,
    answer: str,
    chunk_texts: Iterable[str],
    *,
    final_status: FinalStatus = FinalStatus.COMPLETE,
    answer_kind: AnswerKind = AnswerKind.FULL,
) -> ResearchSession:
    sources: list[SourceRecord] = []
    chunks: list[EvidenceChunk] = []
    for index, text in enumerate(chunk_texts, start=1):
        source = _source(index, raw_text=text)
        sources.append(source)
        chunks.append(_chunk(source, text))
    return ResearchSession(
        protocol=Protocol.RESEARCH,
        question=question,
        final_status=final_status,
        sources=sources,
        evidence_chunks=chunks,
        rerank_passed_count=len(chunks),
        rerank_failed_count=0,
        answer=answer,
        answer_kind=answer_kind,
        confidence=0.9 if answer_kind != AnswerKind.ABSTAIN else None,
        answer_confidence=0.9 if answer_kind != AnswerKind.ABSTAIN else None,
        open_questions=["What exact evidence would resolve this?"]
        if answer_kind in {AnswerKind.PARTIAL, AnswerKind.ABSTAIN}
        else [],
    )


class IterationControllerTests(unittest.TestCase):
    def test_expand_recursive_node_returns_no_children_when_tier_budget_is_spent(self) -> None:
        for total_cost_usd_estimate in (0.5, 0.75):
            with self.subTest(total_cost_usd_estimate=total_cost_usd_estimate):
                session = SimpleNamespace(total_cost_usd_estimate=total_cost_usd_estimate)
                plan = RecursivePlan(
                    session=session,
                    protocol=Protocol.RESEARCH,
                    tier=RecursionTierConfig(
                        protocol=Protocol.RESEARCH,
                        tier_name="one_llm",
                        max_depth_below_scout=2,
                        subquestions_per_node=2,
                        leaf_search_cap=5,
                        budget_ceiling_usd=0.5,
                    ),
                    distinct_llm_count=1,
                    roots=[],
                    asked_questions=[],
                    dropped_repeated_questions=[],
                    leaf_count=1,
                    budget_remaining_usd=0.0,
                )
                parent = RecursiveNode(
                    question="Parent question",
                    depth=1,
                    parent_summary="Scout summary",
                    budget_slice_usd=0.2,
                    recommended_lanes=["searxng_general"],
                    recommended_worker_model="haiku",
                )
                router = Mock()

                children = expand_recursive_node(
                    plan,
                    parent,
                    ["Child question"],
                    parent_summary="Parent summary",
                    router=router,
                )

                self.assertEqual(children, [])

    def test_judge_findings_keeps_supported_session(self) -> None:
        session = _session(
            "What were the settlement terms?",
            "Minimum compensation increased. Streaming residuals include a viewership bonus.",
            [
                "The agreement increased minimum compensation for writers.",
                "The agreement added a viewership-based streaming residual bonus.",
            ],
        )
        primary = FakeCheckerClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "The sources are about the settlement terms.",
                },
                {
                    "verdict": "relevant",
                    "reason": "The second source is also about the settlement terms.",
                },
                {
                    "verdict": "sufficient",
                    "reason": "The raw source text answers the settlement question.",
                    "missing": [],
                },
            ],
        )
        fallback = FakeCheckerClient("haiku-fallback", "haiku", [])
        router = Mock()
        router.judge_config.return_value = {"worker_model": "haiku"}

        with patch.object(
            evidence_gate,
            "get_sufficiency_clients",
            return_value=(primary, fallback),
            create=True,
        ):
            result = judge_findings(session, router)

        self.assertEqual(
            result.kept_chunk_ids,
            [chunk.chunk_id for chunk in session.evidence_chunks],
        )
        self.assertEqual(result.dropped_chunk_ids, [])
        self.assertEqual(result.drop_reasons, {})
        self.assertEqual(result.worker_model, "haiku")
        self.assertEqual(session.rerank_passed_count, 2)
        self.assertEqual(session.rerank_failed_count, 0)
        self.assertEqual(session.final_status, FinalStatus.COMPLETE)
        self.assertEqual(session.answer_kind, AnswerKind.FULL)
        self.assertEqual(session.evidence_gate_decision["terminal_state"], "sufficient")

    def test_judge_findings_abstains_content_free_session_without_crashing(self) -> None:
        session = _session(
            "What does the report prove?",
            "It depends. More research is needed.",
            [
                "The source discusses the topic but gives no concrete answer.",
                "The source says the available evidence is inconclusive.",
            ],
        )
        original_chunk_ids = [chunk.chunk_id for chunk in session.evidence_chunks]
        primary = FakeCheckerClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "The sources are about the report question.",
                },
                {
                    "verdict": "relevant",
                    "reason": "The second source is also about the report question.",
                },
                {
                    "verdict": "insufficient",
                    "reason": "The raw source text does not answer the query.",
                    "missing": ["concrete answer"],
                }
            ],
        )
        fallback = FakeCheckerClient("haiku-fallback", "haiku", [])
        router = Mock()
        router.judge_config.return_value = {"worker_model": "haiku"}

        with patch.object(
            evidence_gate,
            "get_sufficiency_clients",
            return_value=(primary, fallback),
            create=True,
        ):
            result = judge_findings(session, router)

        self.assertEqual(result.kept_chunk_ids, [])
        self.assertEqual(result.dropped_chunk_ids, original_chunk_ids)
        self.assertEqual(
            result.drop_reasons,
            {chunk_id: "sufficiency_exhausted" for chunk_id in original_chunk_ids},
        )
        self.assertEqual(result.worker_model, "haiku")
        self.assertEqual(session.rerank_passed_count, 0)
        self.assertEqual(session.rerank_failed_count, 2)
        self.assertEqual(session.evidence_chunks, [])
        self.assertEqual(session.final_status, FinalStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(session.answer_kind, AnswerKind.ABSTAIN)
        self.assertEqual(session.evidence_gate_decision["gate_reason"], "sufficiency_exhausted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
