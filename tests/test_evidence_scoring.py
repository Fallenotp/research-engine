from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_engine import research_cli
from research_engine.grounding import _source_record_from_extract
from research_engine.schema import (
    AnswerKind,
    EvidenceChunk,
    ExtractionMethod,
    FinalStatus,
    SourceRecord,
    SourceTier,
)


def _write_source(tmp_path: Path, name: str, text: str, *, domain: str = "example.com") -> dict[str, object]:
    raw_text_path = tmp_path / f"{name}.txt"
    raw_text_path.write_text(text, encoding="utf-8")
    return {
        "url": f"https://{domain}/{name}",
        "domain": domain,
        "title": name,
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": len(text),
    }


def _make_source(
    tmp_path: Path,
    name: str,
    text: str,
    *,
    authority_score: float = 0.5,
    tier: SourceTier = SourceTier.T2,
    domain: str = "example.com",
) -> SourceRecord:
    raw_text_path = tmp_path / f"{name}.txt"
    raw_text_path.write_text(text, encoding="utf-8")
    return SourceRecord(
        url=f"https://{domain}/{name}",
        domain=domain,
        title=name,
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(text),
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=raw_text_path,
        char_count=len(text),
        tier=tier,
        topic_authority_score=authority_score,
    )


def _make_chunk(source: SourceRecord, *, rerank_score: float, passed: bool) -> EvidenceChunk:
    return EvidenceChunk(
        source_id=source.source_id,
        paragraph_text="synthetic paragraph",
        char_offset=0,
        char_length=19,
        rerank_score=rerank_score,
        supports_claim="synthetic claim",
        crystal_check_passed=passed,
        crystal_check_score=rerank_score,
    )


@pytest.mark.parametrize("builder", [research_cli.source_record, _source_record_from_extract])
def test_supporting_paragraph_scores_higher_than_unrelated(builder, tmp_path: Path) -> None:
    claim = "The device includes a two year warranty."
    supporting_text = "The manufacturer confirms the device includes a two year warranty for all retail purchases."
    unrelated_text = "This pasta recipe uses tomatoes, basil, and olive oil for dinner."

    supporting_source = builder(_write_source(tmp_path, "supporting", supporting_text), supporting_text, topic="legal")
    unrelated_source = builder(_write_source(tmp_path, "unrelated", unrelated_text), unrelated_text, topic="legal")

    supporting_chunk = research_cli.evidence_chunk(supporting_source, supporting_text, claim)
    unrelated_chunk = research_cli.evidence_chunk(unrelated_source, unrelated_text, claim)

    assert supporting_chunk.rerank_score > unrelated_chunk.rerank_score
    assert supporting_chunk.crystal_check_passed is True
    assert supporting_chunk.rerank_score > research_cli.EVIDENCE_OVERLAP_PASS_THRESHOLD
    assert supporting_chunk.supports_claim == claim
    assert unrelated_chunk.crystal_check_passed is False
    assert unrelated_chunk.rerank_score < research_cli.EVIDENCE_OVERLAP_PASS_THRESHOLD


@pytest.mark.parametrize(
    ("paragraph_text", "claim_text"),
    [
        ("", "The product ships with a battery."),
        ("The product ships with a battery.", ""),
        ("", ""),
    ],
)
def test_lexical_overlap_score_returns_zero_for_empty_inputs(
    paragraph_text: str,
    claim_text: str,
) -> None:
    assert research_cli.lexical_overlap_score(paragraph_text, claim_text) == 0.0


def test_near_verbatim_claim_scores_at_least_ninety_percent() -> None:
    claim = "The product ships with a battery and charger."
    paragraph = "The product ships with a battery and charger in every retail box."

    score = research_cli.lexical_overlap_score(paragraph, claim)

    assert score >= 0.9
    assert 0.0 <= score <= 1.0


def test_long_multi_claim_answer_matches_single_claim_score(tmp_path: Path) -> None:
    paragraph = (
        "Hawthorn offers the lowest minimum order quantity in the UK for fully custom "
        "clothing, from 50 pieces per design."
    )
    target_claim = (
        "Hawthorn [2] Small runs Yes. From 50 pieces per design, the lowest MOQ in "
        "the UK for fully custom clothing [2]"
    )
    long_answer = """
## Short answer

The sources do not name a single UK company proven to do all three: sew organic cotton t-shirts, in the UK, in small runs.

## The two real leads

| Company | What it does | Small runs? | Organic cotton? |
|---|---|---|---|
| Hawthorn [2] | Full UK clothing manufacture | Yes. From 50 pieces per design, the lowest MOQ in the UK for fully custom clothing [2] | Not stated in the source [2] |
| Teemill [3] | UK print-on-demand on GOTS-certified organic cotton t-shirts [3] | Yes. Print-on-demand, so no minimum [3] | Yes, GOTS-certified [3] |

## The catch

Print is not manufacture. Teemill [3], source [1] and Garment Printing [4] all print in the UK on blanks.
""".strip()
    source = _make_source(tmp_path, "hawthorn", paragraph)

    chunk_from_answer = research_cli.evidence_chunk(source, paragraph, long_answer)
    chunk_from_claim = research_cli.evidence_chunk(
        source,
        paragraph,
        long_answer,
        claim=target_claim,
    )

    assert chunk_from_answer.rerank_score == pytest.approx(chunk_from_claim.rerank_score)
    assert chunk_from_answer.crystal_check_passed is True
    assert chunk_from_answer.crystal_check_passed == chunk_from_claim.crystal_check_passed
    assert chunk_from_answer.supports_claim == target_claim
    assert chunk_from_answer.supports_claim == chunk_from_claim.supports_claim
    assert chunk_from_answer.rerank_score > research_cli.EVIDENCE_OVERLAP_PASS_THRESHOLD


def test_confidence_selection_uses_graduated_answer_config(tmp_path: Path) -> None:
    sources = [
        _make_source(tmp_path, "one", "support text one", authority_score=0.5),
        _make_source(tmp_path, "two", "support text two", authority_score=0.5),
    ]
    evidence_chunks = [
        _make_chunk(sources[0], rerank_score=0.4, passed=True),
        _make_chunk(sources[1], rerank_score=0.4, passed=True),
    ]
    router = SimpleNamespace(
        graduated_answer_config=lambda: {
            "full_confidence_min": 0.7,
            "partial_confidence_min": 0.4,
            "abstain_confidence_below": 0.3,
        }
    )

    partial = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert partial.answer_kind == AnswerKind.PARTIAL

    router.graduated_answer_config = lambda: {
        "full_confidence_min": 0.55,
        "partial_confidence_min": 0.35,
        "abstain_confidence_below": 0.2,
    }
    full = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert full.answer_kind == AnswerKind.FULL

    router.graduated_answer_config = lambda: {
        "full_confidence_min": 0.9,
        "partial_confidence_min": 0.8,
        "abstain_confidence_below": 0.7,
    }
    abstain = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert abstain.answer_kind == AnswerKind.ABSTAIN


def _mid_strength_sources_and_chunks(tmp_path: Path) -> tuple[list[SourceRecord], list[EvidenceChunk]]:
    """~0.60 measured confidence under the session formula (below default full=0.70)."""
    sources = [
        _make_source(tmp_path, "one", "support text one", authority_score=0.5),
        _make_source(tmp_path, "two", "support text two", authority_score=0.5),
    ]
    evidence_chunks = [
        _make_chunk(sources[0], rerank_score=0.4, passed=True),
        _make_chunk(sources[1], rerank_score=0.4, passed=True),
    ]
    return sources, evidence_chunks


def test_raising_router_config_does_not_fail_open_to_full(tmp_path: Path) -> None:
    """B-001 regression: graduated_answer_config raise must not invent FULL/COMPLETE."""
    sources, evidence_chunks = _mid_strength_sources_and_chunks(tmp_path)

    def _raise() -> dict:
        raise RuntimeError("config unavailable")

    router = SimpleNamespace(graduated_answer_config=_raise)
    decision = research_cli.session_answer_decision(
        sources, evidence_chunks, router=router
    )

    assert decision.answer_kind != AnswerKind.FULL
    assert decision.final_status != FinalStatus.COMPLETE
    assert decision.confidence == pytest.approx(
        research_cli.compute_session_confidence(sources, evidence_chunks)
    )


def test_non_dict_router_config_does_not_fail_open_to_full(tmp_path: Path) -> None:
    sources, evidence_chunks = _mid_strength_sources_and_chunks(tmp_path)
    router = SimpleNamespace(graduated_answer_config=lambda: "not-a-dict")
    decision = research_cli.session_answer_decision(
        sources, evidence_chunks, router=router
    )

    assert decision.answer_kind != AnswerKind.FULL
    assert decision.final_status != FinalStatus.COMPLETE


def test_missing_router_config_method_does_not_fail_open_to_full(tmp_path: Path) -> None:
    sources, evidence_chunks = _mid_strength_sources_and_chunks(tmp_path)
    router = SimpleNamespace()  # no graduated_answer_config
    decision = research_cli.session_answer_decision(
        sources, evidence_chunks, router=router
    )

    assert decision.answer_kind != AnswerKind.FULL
    assert decision.final_status != FinalStatus.COMPLETE


def test_fallback_thresholds_grade_on_measured_confidence(tmp_path: Path) -> None:
    """Broken router still grades FULL for strong evidence and ABSTAIN for weak."""
    broken_router = SimpleNamespace(
        graduated_answer_config=lambda: (_ for _ in ()).throw(RuntimeError("no config"))
    )

    strong_sources = [
        _make_source(tmp_path, f"s{i}", f"strong support {i}", authority_score=0.95)
        for i in range(3)
    ]
    strong_chunks = [
        _make_chunk(source, rerank_score=0.95, passed=True) for source in strong_sources
    ]
    strong = research_cli.session_answer_decision(
        strong_sources, strong_chunks, router=broken_router
    )
    assert strong.answer_kind == AnswerKind.FULL
    assert strong.final_status == FinalStatus.COMPLETE
    assert strong.confidence == pytest.approx(
        research_cli.compute_session_confidence(strong_sources, strong_chunks)
    )
    assert strong.confidence >= research_cli.DEFAULT_GRADUATED_ANSWER_THRESHOLDS.full_confidence_min

    weak_sources = [
        _make_source(tmp_path, "weak", "weak support", authority_score=0.1),
    ]
    weak_chunks = [
        _make_chunk(weak_sources[0], rerank_score=0.1, passed=False),
    ]
    weak = research_cli.session_answer_decision(
        weak_sources, weak_chunks, router=broken_router
    )
    assert weak.answer_kind == AnswerKind.ABSTAIN
    assert weak.final_status != FinalStatus.COMPLETE
    assert weak.confidence == research_cli.ABSTAIN_CONFIDENCE_FLOOR


def test_working_router_config_happy_path_unchanged(tmp_path: Path) -> None:
    """Working graduated_answer_config still drives FULL / PARTIAL / ABSTAIN as before."""
    sources, evidence_chunks = _mid_strength_sources_and_chunks(tmp_path)
    measured = research_cli.compute_session_confidence(sources, evidence_chunks)

    router = SimpleNamespace(
        graduated_answer_config=lambda: {
            "full_confidence_min": 0.55,
            "partial_confidence_min": 0.35,
            "abstain_confidence_below": 0.2,
        }
    )
    full = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert full.answer_kind == AnswerKind.FULL
    assert full.final_status == FinalStatus.COMPLETE
    assert full.confidence == pytest.approx(measured)

    router.graduated_answer_config = lambda: {
        "full_confidence_min": 0.7,
        "partial_confidence_min": 0.4,
        "abstain_confidence_below": 0.3,
    }
    partial = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert partial.answer_kind == AnswerKind.PARTIAL
    assert partial.final_status == FinalStatus.WEAK_SOURCES
    assert partial.confidence == pytest.approx(measured)

    router.graduated_answer_config = lambda: {
        "full_confidence_min": 0.9,
        "partial_confidence_min": 0.8,
        "abstain_confidence_below": 0.7,
    }
    abstain = research_cli.session_answer_decision(sources, evidence_chunks, router=router)
    assert abstain.answer_kind == AnswerKind.ABSTAIN
    assert abstain.final_status == FinalStatus.WEAK_SOURCES
    assert abstain.confidence == research_cli.ABSTAIN_CONFIDENCE_FLOOR


@pytest.mark.parametrize("builder", [research_cli.source_record, _source_record_from_extract])
def test_authority_score_maps_to_real_source_tiers(builder, tmp_path: Path) -> None:
    authoritative = builder(
        _write_source(tmp_path, "authoritative", "verified legal source", domain="courtlistener.com"),
        "verified legal source",
        topic="legal",
    )
    unknown = builder(
        _write_source(tmp_path, "unknown", "unknown legal source", domain="unknown-domain.example"),
        "unknown legal source",
        topic="legal",
    )
    no_topic = builder(
        _write_source(tmp_path, "no-topic", "fallback legal source", domain="courtlistener.com"),
        "fallback legal source",
        topic=None,
    )

    assert authoritative.tier == SourceTier.T1
    assert unknown.tier == SourceTier.T3
    assert no_topic.tier == SourceTier.T2


def test_skip_web_avoids_searxng_and_runs_local_lanes(monkeypatch, tmp_path: Path) -> None:
    local_calls = 0
    raw_text_path = tmp_path / "local-memory.txt"
    full_text = "We discussed battery replacements and noted the device carries a two year warranty."
    raw_text_path.write_text(full_text, encoding="utf-8")

    class FakeRouter:
        def route(self, question: str):
            return SimpleNamespace(
                topic="personal",
                lanes=["mentor_memory"],
                require_tier_1=False,
                skip_web=True,
            )

        def graduated_answer_config(self) -> dict[str, float]:
            # Explicit thresholds so this skip_web path does not depend on
            # the old fail-open FULL behavior when config is missing (B-001).
            return {
                "full_confidence_min": 0.20,
                "partial_confidence_min": 0.10,
                "abstain_confidence_below": 0.05,
            }

    def fake_local_lanes(*args, **kwargs):
        nonlocal local_calls
        local_calls += 1
        return [("mentor_memory", {"results": [{"url": raw_text_path.resolve().as_uri()}]})]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("web search should be skipped when skip_web=True")

    def fake_extract(url, *args, **kwargs):
        return {
            "url": url,
            "domain": "local",
            "title": "Local memory note",
            "extraction_method": "curl",
            "raw_text_path": str(raw_text_path),
            "char_count": len(full_text),
        }

    def fake_save_session(session):
        session_path.write_text(session.model_dump_json(), encoding="utf-8")
        return session_path

    session_path = tmp_path / "session.json"
    monkeypatch.setattr(research_cli, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(research_cli, "run_local_search_lanes", fake_local_lanes)
    monkeypatch.setattr(research_cli.logged_search, "searxng", fail_if_called)
    monkeypatch.setattr(research_cli.logged_search, "proxy", fail_if_called)
    monkeypatch.setattr(research_cli, "run_grok_x_search", fail_if_called)
    monkeypatch.setattr(research_cli, "extract_clean_text", fake_extract)
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *args, **kwargs: ("local memory answer [1]", "codex"),
    )
    monkeypatch.setattr(research_cli, "save_session_safely", fake_save_session)
    monkeypatch.setattr(research_cli, "telemetry_safely", lambda: None)

    result = research_cli.run_search("did we discuss the warranty?", topic="memory", agent="pytest")

    assert local_calls == 1
    assert result.session.answer == "local memory answer [1]"
