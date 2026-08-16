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
    decide_next_iteration,
    detect_gaps,
    expand_recursive_node,
    judge_findings,
    plan_recursive_research,
)
from research_engine.router import load_router
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


def _source(
    index: int,
    *,
    raw_text: str | None = None,
    published_date: str | None = None,
    counter_evidence_flagged: bool = False,
) -> SourceRecord:
    text = raw_text or f"source text {index}"
    return SourceRecord(
        url=f"https://source-{index}.example/article",
        domain=f"source-{index}.example",
        title=f"Source {index}",
        published_date=published_date,
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(text),
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=Path(f"/tmp/research-engine-iteration-controller/source-{index}.txt"),
        char_count=len(text),
        tier=SourceTier.T1,
        topic_authority_score=1.0,
        counter_evidence_flagged=counter_evidence_flagged,
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

    def test_decide_next_iteration_returns_iterate_for_weak_session_and_not_for_strong(self) -> None:
        router = load_router()

        strong_source_a = _source(201, raw_text="strong a", counter_evidence_flagged=True)
        strong_source_b = _source(202, raw_text="strong b")
        strong_chunk_a = _chunk(strong_source_a, "Claim A")
        strong_chunk_b = _chunk(strong_source_b, "Claim A")
        strong_session = ResearchSession(
            protocol=Protocol.DEEP_RESEARCH,
            question="What changed this morning in the merger talks?",
            final_status=FinalStatus.COMPLETE,
            answer="A grounded answer.",
            sources=[strong_source_a, strong_source_b],
            evidence_chunks=[strong_chunk_a, strong_chunk_b],
            rerank_passed_count=2,
            rerank_failed_count=0,
            cross_model_verifications=[
                {
                    "claim": "Claim A",
                    "grounding_chunks": [strong_chunk_a.chunk_id],
                    "analytical_lens_passed": True,
                    "analytical_lens_notes": "ok",
                    "creative_lens_passed": True,
                    "creative_lens_notes": "ok",
                    "skeptical_lens_passed": True,
                    "skeptical_lens_notes": "ok",
                }
            ],
        )

        stale_date = "2026-08-10"
        weak_source_a = _source(203, raw_text="weak a", published_date=stale_date)
        weak_source_b = _source(204, raw_text="weak b", published_date=stale_date)
        weak_chunk_a = _chunk(weak_source_a, "Claim B")
        weak_chunk_b = _chunk(weak_source_b, "Claim B")
        weak_session = ResearchSession(
            protocol=Protocol.DEEP_RESEARCH,
            question="What was announced today about the merger?",
            final_status=FinalStatus.COMPLETE,
            answer="Thin answer.",
            sources=[weak_source_a, weak_source_b],
            evidence_chunks=[weak_chunk_a, weak_chunk_b],
            rerank_passed_count=1,
            rerank_failed_count=3,
            open_questions=["What did regulators say?"],
            cross_model_verifications=[],
        )

        strong_decision = decide_next_iteration(strong_session.question, strong_session)
        weak_decision = decide_next_iteration(weak_session.question, weak_session)

        self.assertEqual(router.iteration_policy.max_loops, 3)
        self.assertFalse(strong_decision.should_iterate)
        self.assertEqual(strong_decision.reason, "no triggering gaps")
        self.assertTrue(weak_decision.should_iterate)
        self.assertIn("stale data", weak_decision.reason)
        self.assertTrue(any(gap.triggered_iteration for gap in weak_session.gaps_detected))

    def test_detect_gaps_flags_stale_source_for_freshness_bounded_route(self) -> None:
        stale_source = _source(301, raw_text="stale", published_date="2026-08-10")
        stale_chunk = _chunk(stale_source, "Merger claim")
        session = ResearchSession(
            protocol=Protocol.DEEP_RESEARCH,
            question="What was announced today about the merger?",
            final_status=FinalStatus.COMPLETE,
            answer="Answer.",
            sources=[stale_source],
            evidence_chunks=[stale_chunk],
            rerank_passed_count=1,
            rerank_failed_count=0,
            cross_model_verifications=[
                {
                    "claim": "Merger claim",
                    "grounding_chunks": [stale_chunk.chunk_id],
                    "analytical_lens_passed": True,
                    "analytical_lens_notes": "ok",
                    "creative_lens_passed": True,
                    "creative_lens_notes": "ok",
                    "skeptical_lens_passed": True,
                    "skeptical_lens_notes": "ok",
                }
            ],
        )

        gaps = detect_gaps(session)

        self.assertIn(
            "stale data",
            [gap.detection_reason for gap in gaps],
        )

    def test_detect_gaps_treats_offset_timestamp_as_utc_instead_of_relabeling_it(self) -> None:
        fresh_offset_source = _source(
            302,
            raw_text="fresh offset",
            published_date="2026-08-14T23:30:00-06:00",
        )
        fresh_offset_chunk = _chunk(fresh_offset_source, "Fresh claim")
        session = ResearchSession(
            protocol=Protocol.DEEP_RESEARCH,
            question="What was announced today about the merger?",
            final_status=FinalStatus.COMPLETE,
            answer="Answer.",
            sources=[fresh_offset_source],
            evidence_chunks=[fresh_offset_chunk],
            rerank_passed_count=1,
            rerank_failed_count=0,
            cross_model_verifications=[
                {
                    "claim": "Fresh claim",
                    "grounding_chunks": [fresh_offset_chunk.chunk_id],
                    "analytical_lens_passed": True,
                    "analytical_lens_notes": "ok",
                    "creative_lens_passed": True,
                    "creative_lens_notes": "ok",
                    "skeptical_lens_passed": True,
                    "skeptical_lens_notes": "ok",
                }
            ],
        )

        with patch("research_engine.iteration_controller.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = datetime(2026, 8, 15, 4, 0, tzinfo=timezone.utc)
            mocked_datetime.fromisoformat.side_effect = datetime.fromisoformat
            gaps = detect_gaps(session)

        self.assertNotIn(
            "stale data",
            [gap.detection_reason for gap in gaps],
        )

    def test_plan_recursive_research_returns_no_roots_when_tier_depth_is_zero(self) -> None:
        session = ResearchSession(
            protocol=Protocol.RESEARCH,
            question="How should a launch work?",
            final_status=FinalStatus.COMPLETE,
            answer="Draft answer.",
        )

        plan = plan_recursive_research(
            session,
            load_router(),
            ["Question A", "Question B", "Question C"],
        )

        self.assertEqual(plan.tier.max_depth_below_scout, 0)
        self.assertEqual(plan.roots, [])
        self.assertEqual(plan.leaf_count, 0)
        self.assertEqual(plan.asked_questions, [])

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
