from __future__ import annotations

import argparse
import contextlib
import glob
import io
import itertools
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ in {None, ""}:  # pragma: no cover - direct file execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_engine import llm_call, logged_search, telemetry_observer
from research_engine.anti_hallucination_gate import validate as validate_model_output

from . import paths
from research_engine.dispatcher import (
    AGY_SKIP_PERMISSIONS_FLAG,
    AGY_INTERACTIVE_GEMINI_MODEL,
    GEMINI_CLI_HOME,
    GEMINI_DAILY_COUNTER_FILE,
    GEMINI_TIMEOUT_SECONDS,
    GeminiProScoutError,
    GROK_TIMEOUT_SECONDS,
    WorkerSpec,
    build_api_lane_request,
    dispatch,
    dispatch_scout,
    finalize_gemini_daily_budget,
    gemini_daily_budget_available,
    is_unattended_research_run as dispatcher_is_unattended_research_run,
    reserve_gemini_daily_budget,
    resolve_agy_model,
)
from research_engine.extractor import extract_clean_text
from research_engine.router import load_router, source_authority_score
from research_engine.schema import (
    AgentRole,
    AnswerKind,
    Disagreement,
    EvidenceChunk,
    ExtractionMethod,
    FinalStatus,
    GeminiProRunKind,
    GeminiProRunRecord,
    Protocol,
    QueryCall,
    ResearchSession,
    SourceRecord,
    SourceTier,
    Territory,
    WorkerModel,
)
import research_engine.persistence as persistence


SPECIALIZED_PROVIDERS = ("tavily", "linkup", "exa", "youcom")
ABSTAIN_MESSAGE = "Insufficient evidence to answer from the retrieved sources."
CODEX_TIMEOUT_SECONDS = 180
CODEX_MODEL_ID = "gpt-5.4-mini"
MISTRAL_FREE_KEYS_PATH = paths.optional_path(paths.MISTRAL_KEYS_FILE_ENV) or paths.data_path(
    "missing-mistral-free-keys.env"
)
MISTRAL_FREE_MODEL = os.environ.get("MISTRAL_FREE_MODEL", "mistral-small-latest")
MISTRAL_TIMEOUT_SECONDS = 120
_MISTRAL_KEY_COUNTER = itertools.count()
FINAL_SYNTHESIS_BACKEND = WorkerModel.OPUS.value
FINAL_SYNTHESIS_CHAIN: tuple[tuple[str, str | None, str], ...] = (
    (WorkerModel.OPUS.value, None, WorkerModel.OPUS.value),
    ("codex", "gpt-5.5", WorkerModel.CODEX_5_5.value),
    (WorkerModel.SONNET.value, None, WorkerModel.SONNET.value),
)
QUESTION_TOPIC_BUSINESS = "business"
QUESTION_TOPIC_ACADEMIC = "academic"
QUESTION_TOPIC_CODE = "code"
QUESTION_TOPIC_MIXED = "mixed"
QUESTION_TOPIC_GENERAL = "general"
BUSINESS_DEEP_LANE_SEQUENCE = [
    "linkup_direct",
    "tavily_direct",
    "youcom_direct",
    "exa_direct",
    "reddit_rss",
    "hn_algolia",
    "x_pulse",
    "reddit_failures",
    "searxng_forums",
    "linkup_direct",
    "reddit_rss",
    "hn_algolia",
    "x_pulse",
    "tavily_direct",
    "youcom_direct",
    "grok_x_search",
]
BUSINESS_RESEARCH_LANE_SEQUENCE = ["linkup_direct", "reddit_rss", "grok_x_search"]
MIXED_DEEP_LANE_SEQUENCE = [
    "searxng_general",
    "linkup_direct",
    "semantic_scholar",
    "tavily_direct",
    "exa_direct",
    "reddit_rss",
    "github_code",
    "hn_algolia",
    "youcom_direct",
    "arxiv",
    "x_pulse",
    "searxng_forums",
    "core",
    "reddit_failures",
    "linkup_direct",
    "grok_x_search",
]
MIN_FREE_SEARCH_RESULTS = 5
QUERY_PHRASING_PLAYBOOK_LINES = (
    "Query phrasing rules:",
    "- Use keyword-style queries for lexical engines; use semantic/full-question phrasing only for semantic lanes.",
    "- For X connects to Y questions, split into parallel X and Y searches or pivot X -> Z, then search Z + Y.",
    "- Cap reformulation at 2 retries before escalating or abstaining.",
    "- Pair premise-confirming queries with premise-challenging counter-case queries.",
)
TIER_1_AUTHORITY_MIN = 0.85
TIER_2_AUTHORITY_MIN = 0.5
NON_T1_SOURCE_MARKER = "[NON-T1] "
# Calibrated 2026-08-16 on 12 source/answer pairs from sessions
# 199ef0e9-7899-46a3-b012-34d08cb3c1a5 and cb5df857-29a2-4a79-850a-e2ac37113534
# using claim-token containment (|paragraph ∩ claim| / |claim|).
EVIDENCE_OVERLAP_PASS_THRESHOLD = 0.45
LEGACY_COMPLETE_CONFIDENCE = 0.7
ABSTAIN_CONFIDENCE_FLOOR = 0.0
SESSION_CONFIDENCE_SOURCE_TARGET = 3
SESSION_CONFIDENCE_SOURCE_WEIGHT = 0.2
SESSION_CONFIDENCE_AUTHORITY_WEIGHT = 0.3
SESSION_CONFIDENCE_RERANK_WEIGHT = 0.3
SESSION_CONFIDENCE_SUPPORT_WEIGHT = 0.2
LOCAL_LANE_RESULT_LIMIT = 5
LOCAL_MEMORY_SCAN_LIMIT = 200
LOCAL_SOURCE_TEXT_LIMIT = 4000
LOCAL_MEMORY_CACHE_DIR = Path(tempfile.gettempdir()) / "research-engine-local-memory"
EVIDENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
EVIDENCE_META_SECTION_HEADINGS = frozenset({"next action", "what i would do"})
EVIDENCE_META_LINE_PREFIXES = ("caveats/disagreements:",)
EVIDENCE_META_LINE_STARTS = ("**moq**", "**gots**")
EVIDENCE_CLAIM_SKIP_PHRASES = (
    "not stated in the source",
    "not stated in source",
    "not written down",
    "no source proves",
)
EVIDENCE_TABLE_HEADER_TITLES = frozenset(
    {
        "company",
        "what they actually do",
        "what it does",
        "small runs?",
        "organic cotton?",
        "organic cotton",
    }
)

logger = logging.getLogger(__name__)


def is_gemini_quota_model(model_id: str | None) -> bool:
    normalized = str(model_id or "").strip().lower()
    return normalized.startswith("gemini ") or normalized.startswith("gemini-")


def is_unattended_research_run() -> bool:
    return dispatcher_is_unattended_research_run()


def model_for_agy_worker(spec: WorkerSpec) -> str:
    requested_model = spec.model_id or AGY_INTERACTIVE_GEMINI_MODEL
    if not is_gemini_quota_model(requested_model):
        return requested_model
    budget_available = gemini_daily_budget_available(path=GEMINI_DAILY_COUNTER_FILE)
    model_id = resolve_agy_model(
        requested_model,
        gemini_budget_available=budget_available,
    )
    if model_id != requested_model and not is_unattended_research_run():
        logger.warning(
            "Gemini daily budget exhausted; routing agy worker to %s",
            model_id,
        )
    return model_id


@dataclass(frozen=True)
class SearchRunResult:
    session: ResearchSession
    path: Path | None
    backend: str
    n_sources: int
    fleet_warning: str | None = None


@dataclass(frozen=True)
class TerritoryRun:
    territory: Territory
    sources: list[SourceRecord]
    full_texts: list[str]
    queries_run: list[QueryCall]
    summary: str
    counter: bool = False
    output_path: str | None = None


@dataclass(frozen=True)
class GeminiInterlockAttempt:
    run_type: GeminiProRunKind
    record: GeminiProRunRecord | None = None
    output_text: str = ""
    failure_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.record is not None


@dataclass(frozen=True)
class GraduatedAnswerThresholds:
    full_confidence_min: float
    partial_confidence_min: float
    abstain_confidence_below: float


# Fail-closed default when router graduated_answer_config is missing or unreadable.
# Mirrors research_engine/router_config.yaml graduated_answer so the engine still
# grades FULL/PARTIAL/ABSTAIN from measured confidence instead of inventing
# FULL + COMPLETE without measurement (B-001).
DEFAULT_GRADUATED_ANSWER_THRESHOLDS = GraduatedAnswerThresholds(
    full_confidence_min=0.70,
    partial_confidence_min=0.35,
    abstain_confidence_below=0.35,
)


@dataclass(frozen=True)
class AnswerDecision:
    answer_kind: AnswerKind
    final_status: FinalStatus
    confidence: float
    open_questions: tuple[str, ...] = ()


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _overlap_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in EVIDENCE_STOPWORDS
    }


def lexical_overlap_score(paragraph_text: str, claim_text: str) -> float:
    """Return claim-token containment for paragraph-vs-claim support.

    Both strings are lowercased, reduced to alphanumeric tokens, stripped of a small
    stopword list, then scored as |paragraph_tokens ∩ claim_tokens| / |claim_tokens|.
    Empty claim token sets score 0.0, and the result is clamped to [0.0, 1.0].
    """

    paragraph_tokens = _overlap_tokens(paragraph_text)
    claim_tokens = _overlap_tokens(claim_text)
    if not paragraph_tokens or not claim_tokens:
        return 0.0
    return clamp_unit_interval(len(paragraph_tokens & claim_tokens) / len(claim_tokens))


def _is_markdown_table_separator(line: str) -> bool:
    return bool(
        re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", line)
    )


def _register_claim_candidate(
    claim_text: str,
    *,
    cited_claims: list[str],
    fallback_claims: list[str],
    seen: set[str],
) -> None:
    normalized = " ".join(claim_text.split()).strip()
    if not normalized:
        return
    lowered = normalized.lower()
    if any(phrase in lowered for phrase in EVIDENCE_CLAIM_SKIP_PHRASES):
        return
    if len(_overlap_tokens(normalized)) < 4:
        return
    if normalized in seen:
        return
    seen.add(normalized)
    if re.search(r"\[\d+\]", normalized):
        cited_claims.append(normalized)
    else:
        fallback_claims.append(normalized)


def _claim_candidates(answer_text: str, explicit_claim: str | None = None) -> list[str]:
    if explicit_claim is not None:
        normalized = " ".join(explicit_claim.split()).strip()
        return [normalized] if normalized else []

    cited_claims: list[str] = []
    fallback_claims: list[str] = []
    seen: set[str] = set()
    current_heading = ""
    table_headers: list[str] | None = None

    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            table_headers = None
            continue
        if line.startswith("#"):
            current_heading = line.lstrip("#").strip().lower()
            table_headers = None
            continue

        lowered = line.lower()
        if current_heading in EVIDENCE_META_SECTION_HEADINGS:
            continue
        if any(lowered.startswith(prefix) for prefix in EVIDENCE_META_LINE_PREFIXES):
            continue
        if any(lowered.startswith(prefix) for prefix in EVIDENCE_META_LINE_STARTS):
            continue
        if _is_markdown_table_separator(line):
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells:
                continue
            lowered_cells = [cell.lower() for cell in cells]
            if all(
                cell in EVIDENCE_TABLE_HEADER_TITLES or cell.endswith("?")
                for cell in lowered_cells
            ):
                table_headers = cells
                continue

            label = cells[0]
            headers = table_headers or [""] * len(cells)
            for index, cell in enumerate(cells[1:], start=1):
                if not cell:
                    continue
                header = headers[index].rstrip("?").strip() if index < len(headers) else ""
                _register_claim_candidate(
                    " ".join(part for part in (label, header, cell) if part),
                    cited_claims=cited_claims,
                    fallback_claims=fallback_claims,
                    seen=seen,
                )
            continue

        _register_claim_candidate(
            line,
            cited_claims=cited_claims,
            fallback_claims=fallback_claims,
            seen=seen,
        )

    claims = cited_claims or fallback_claims
    if claims:
        return claims
    fallback = " ".join(answer_text.split()).strip()
    return [fallback] if fallback else []


def _paragraph_candidates(full_text: str) -> list[str]:
    candidates: list[str] = []
    for block in re.split(r"\n\s*\n+", full_text):
        normalized = " ".join(block.split()).strip()
        if not normalized:
            continue
        if len(normalized) <= 500:
            candidates.append(normalized)
            continue
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + 500)
            if end < len(normalized):
                split = normalized.rfind(" ", start, end)
                if split > start + 250:
                    end = split
            window = normalized[start:end].strip()
            if window:
                candidates.append(window)
            start = end
    fallback = " ".join(full_text.split()).strip()
    if not candidates and fallback:
        candidates.append(fallback)
    return candidates or ["n/a"]


def best_supporting_paragraph(
    full_text: str,
    answer_text: str,
    *,
    claim: str | None = None,
) -> tuple[str, str, float]:
    best_paragraph = "n/a"
    claim_candidates = _claim_candidates(answer_text, explicit_claim=claim)
    best_claim = claim_candidates[0] if claim_candidates else "n/a"
    best_score = 0.0
    for paragraph in _paragraph_candidates(full_text):
        for claim_text in claim_candidates:
            score = lexical_overlap_score(paragraph, claim_text)
            if score > best_score:
                best_paragraph = paragraph
                best_claim = claim_text
                best_score = score
    return best_paragraph[:300] or "n/a", best_claim or "n/a", best_score


def authority_tier(
    domain: str,
    *,
    topic: str | None,
    authority_score: float | None = None,
) -> SourceTier:
    if topic is None:
        return SourceTier.T2
    score = source_authority_score(domain, topic) if authority_score is None else authority_score
    if score >= TIER_1_AUTHORITY_MIN:
        return SourceTier.T1
    if score >= TIER_2_AUTHORITY_MIN:
        return SourceTier.T2
    return SourceTier.T3


def compute_session_confidence(
    sources: list[SourceRecord],
    evidence_chunks: list[EvidenceChunk],
) -> float:
    """Blend measured source count, authority, overlap, and support into one score.

    Formula:
    - source_count_factor = min(number_of_sources / 3, 1.0)
    - authority_mean = average topic_authority_score across sources
    - rerank_mean = average rerank_score across evidence chunks
    - support_ratio = fraction of chunks that passed the overlap threshold

    Confidence = 0.2*source_count_factor + 0.3*authority_mean
               + 0.3*rerank_mean + 0.2*support_ratio
    """

    if not sources or not evidence_chunks:
        return ABSTAIN_CONFIDENCE_FLOOR
    source_count_factor = clamp_unit_interval(
        len(sources) / SESSION_CONFIDENCE_SOURCE_TARGET
    )
    authority_mean = sum(source.topic_authority_score for source in sources) / len(sources)
    rerank_mean = sum(chunk.rerank_score for chunk in evidence_chunks) / len(evidence_chunks)
    support_ratio = sum(
        1.0 for chunk in evidence_chunks if chunk.crystal_check_passed
    ) / len(evidence_chunks)
    return clamp_unit_interval(
        (SESSION_CONFIDENCE_SOURCE_WEIGHT * source_count_factor)
        + (SESSION_CONFIDENCE_AUTHORITY_WEIGHT * authority_mean)
        + (SESSION_CONFIDENCE_RERANK_WEIGHT * rerank_mean)
        + (SESSION_CONFIDENCE_SUPPORT_WEIGHT * support_ratio)
    )


def graduated_answer_thresholds(router) -> GraduatedAnswerThresholds | None:
    getter = getattr(router, "graduated_answer_config", None)
    if not callable(getter):
        logger.warning(
            "graduated_answer_config missing or not callable on router type %s; "
            "caller will use fail-closed DEFAULT_GRADUATED_ANSWER_THRESHOLDS",
            type(router).__name__,
        )
        return None
    try:
        config = getter()
    except Exception as exc:
        logger.warning(
            "graduated_answer_config raised %s: %s; "
            "caller will use fail-closed DEFAULT_GRADUATED_ANSWER_THRESHOLDS",
            type(exc).__name__,
            exc,
        )
        return None
    if not isinstance(config, dict):
        logger.warning(
            "graduated_answer_config returned %s, not dict; "
            "caller will use fail-closed DEFAULT_GRADUATED_ANSWER_THRESHOLDS",
            type(config).__name__,
        )
        return None
    try:
        return GraduatedAnswerThresholds(
            full_confidence_min=clamp_unit_interval(config["full_confidence_min"]),
            partial_confidence_min=clamp_unit_interval(config["partial_confidence_min"]),
            abstain_confidence_below=clamp_unit_interval(
                config["abstain_confidence_below"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "graduated_answer_config keys/values invalid (%s); "
            "caller will use fail-closed DEFAULT_GRADUATED_ANSWER_THRESHOLDS",
            exc,
        )
        return None


def session_answer_decision(
    sources: list[SourceRecord],
    evidence_chunks: list[EvidenceChunk],
    *,
    router,
) -> AnswerDecision:
    thresholds = graduated_answer_thresholds(router)
    if thresholds is None:
        # Fail closed: grade on measured confidence with defaults, never invent FULL.
        thresholds = DEFAULT_GRADUATED_ANSWER_THRESHOLDS

    try:
        measured_confidence = compute_session_confidence(sources, evidence_chunks)
    except Exception as exc:
        logger.warning(
            "compute_session_confidence failed (%s); abstaining at confidence 0.0",
            exc,
        )
        return AnswerDecision(
            answer_kind=AnswerKind.ABSTAIN,
            final_status=FinalStatus.WEAK_SOURCES,
            confidence=ABSTAIN_CONFIDENCE_FLOOR,
            open_questions=(
                "Session confidence could not be measured from sources and evidence.",
            ),
        )

    if measured_confidence >= thresholds.full_confidence_min:
        return AnswerDecision(
            answer_kind=AnswerKind.FULL,
            final_status=FinalStatus.COMPLETE,
            confidence=measured_confidence,
        )

    open_question = (
        f"Measured confidence {measured_confidence:.2f} stayed below the full-answer "
        "threshold; gather more high-authority sources or stronger paragraph overlap."
    )
    if measured_confidence >= thresholds.partial_confidence_min:
        return AnswerDecision(
            answer_kind=AnswerKind.PARTIAL,
            final_status=FinalStatus.WEAK_SOURCES,
            confidence=measured_confidence,
            open_questions=(open_question,),
        )
    if measured_confidence < thresholds.abstain_confidence_below:
        return AnswerDecision(
            answer_kind=AnswerKind.ABSTAIN,
            final_status=FinalStatus.WEAK_SOURCES,
            confidence=ABSTAIN_CONFIDENCE_FLOOR,
            open_questions=(open_question,),
        )
    return AnswerDecision(
        answer_kind=AnswerKind.PARTIAL,
        final_status=FinalStatus.WEAK_SOURCES,
        confidence=measured_confidence,
        open_questions=(open_question,),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", default="search")
    parser.add_argument("--topic")
    parser.add_argument(
        "--agent",
        default=os.environ.get("RESEARCH_AGENT") or "harness",
    )
    parser.add_argument("--llm")
    args = parser.parse_args(argv)

    topic = args.topic or slugify_question(args.question)
    try:
        if args.mode == "search":
            result = run_search(
                args.question,
                topic=topic,
                agent=args.agent,
                llm_prefer=args.llm,
            )
        elif args.mode == "research":
            result = run_research(
                args.question,
                topic=topic,
                agent=args.agent,
                llm_prefer=args.llm,
            )
        elif args.mode == "deep-research":
            result = run_deep_research(
                args.question,
                topic=topic,
                agent=args.agent,
                llm_prefer=args.llm,
            )
        else:
            print("supported modes: search, research, deep-research")
            return 0
    except Exception as exc:  # noqa: BLE001 - last-resort no-traceback boundary
        session = build_abstain_session(
            args.question,
            protocol=protocol_for_mode(args.mode),
            agent=args.agent,
            queries_run=[],
            open_question=f"unexpected CLI failure: {type(exc).__name__}: {exc}",
        )
        path = save_session_safely(session)
        telemetry_safely()
        result = SearchRunResult(session=session, path=path, backend="none", n_sources=0)

    print_result(result)
    return 0


def run_search(
    question: str,
    *,
    topic: str | None = None,
    agent: str | None = None,
    llm_prefer: str | None = None,
) -> SearchRunResult:
    topic = topic or slugify_question(question)
    agent = agent or os.environ.get("RESEARCH_AGENT") or "harness"
    errors: list[str] = []

    chosen_provider = "tavily"
    router = None
    route_lanes: list[str] = []
    authority_topic: str | None = None
    require_tier_1 = False
    skip_web = False
    try:
        router = load_router()
        route = router.route(question)
        route_lanes = list(getattr(route, "lanes", []))
        authority_topic = getattr(route, "topic", None)
        require_tier_1 = bool(getattr(route, "require_tier_1", False))
        skip_web = bool(getattr(route, "skip_web", False))
        chosen_provider = choose_provider(route_lanes)
    except Exception as exc:  # noqa: BLE001 - router failure degrades later
        errors.append(f"router failed: {type(exc).__name__}: {exc}")

    local_payloads: list[tuple[str, dict[str, Any]]] = []
    if skip_web:
        sx = {"results": [], "skipped": "skip_web"}
        px = {"results": [], "skipped": "skip_web"}
        specialty_payloads: list[tuple[str, dict[str, Any]]] = []
        local_payloads = run_local_search_lanes(
            route_lanes,
            router=router,
            question=question,
            errors=errors,
        )
        queries_run = build_local_query_calls(question, local_payloads=local_payloads)
        grok_x_summary = ""
        source_urls = candidate_urls(*(payload for _lane, payload in local_payloads))
    else:
        sx = run_logged_search(
            logged_search.searxng,
            question,
            protocol=Protocol.SEARCH.value,
            topic=topic,
            agent=agent,
            errors=errors,
        )
        specialty_payloads = []
        if not has_enough_free_results(sx):
            specialty_payloads = run_free_api_lanes(
                route_lanes,
                router=router,
                question=question,
                protocol=Protocol.SEARCH.value,
                topic=topic,
                agent=agent,
                errors=errors,
            )

        free_payloads = [sx, *(payload for _lane, payload in specialty_payloads)]
        if any(has_enough_free_results(payload) for payload in free_payloads) or (
            sum(len(_results(payload)) for payload in free_payloads) >= MIN_FREE_SEARCH_RESULTS
        ):
            px = {"results": [], "skipped": "free_first_sufficient"}
        else:
            px = run_logged_search(
                logged_search.proxy,
                question,
                provider=chosen_provider,
                protocol=Protocol.SEARCH.value,
                topic=topic,
                agent=agent,
                errors=errors,
            )
        queries_run = build_query_calls(
            question,
            sx=sx,
            px=px,
            chosen_provider=chosen_provider,
            api_payloads=specialty_payloads,
        )
        grok_x_summary, grok_x_query = run_grok_x_search(
            question,
            protocol=Protocol.SEARCH,
            topic=topic,
            router=router,
        )
        queries_run.append(grok_x_query)
        source_urls = candidate_urls(sx, *free_payloads[1:], px)

    source_pairs = extract_sources(
        source_urls,
        errors=errors,
        topic=authority_topic,
        require_tier_1=require_tier_1,
    )
    if not source_pairs:
        session = build_abstain_session(
            question,
            agent=agent,
            queries_run=queries_run,
            open_question=first_error_or("no usable sources retrieved", errors),
        )
        path = save_session_safely(session)
        telemetry_safely()
        return SearchRunResult(session=session, path=path, backend="none", n_sources=0)

    try:
        answer, backend = llm_call.llm_complete(
            synthesis_prompt(
                question,
                source_pairs,
                grok_x_summary=grok_x_summary,
            ),
            prefer=llm_prefer,
        )
        if backend == "gemini":
            answer = apply_anti_hallucination_gate(answer, label="search_synthesis_gemini")
    except Exception as exc:  # noqa: BLE001 - synthesis failure becomes abstain
        session = build_abstain_session(
            question,
            agent=agent,
            queries_run=queries_run,
            sources=[source for source, _ in source_pairs],
            open_question=f"synthesis failed: {type(exc).__name__}: {exc}",
        )
        path = save_session_safely(session)
        telemetry_safely()
        return SearchRunResult(
            session=session,
            path=path,
            backend="none",
            n_sources=len(source_pairs),
        )

    session = build_complete_session(
        question,
        answer=answer,
        agent=agent,
        queries_run=queries_run,
        source_pairs=source_pairs,
        router=router,
    )
    path = save_session_safely(session)
    telemetry_safely()
    return SearchRunResult(
        session=session,
        path=path,
        backend=backend,
        n_sources=len(session.sources),
    )


def run_research(
    question: str,
    *,
    topic: str | None = None,
    agent: str | None = None,
    llm_prefer: str | None = None,
) -> SearchRunResult:
    plan = research_fleet_plan()
    return run_multi_territory_research(
        question,
        protocol=Protocol.RESEARCH,
        territory_specs=plan.specs,
        topic=topic,
        agent=agent,
        llm_prefer=llm_prefer,
        allow_iteration=False,
        fleet_warning=plan.warning,
    )


def run_deep_research(
    question: str,
    *,
    topic: str | None = None,
    agent: str | None = None,
    llm_prefer: str | None = None,
) -> SearchRunResult:
    plan = deep_research_fleet_plan()
    return run_multi_territory_research(
        question,
        protocol=Protocol.DEEP_RESEARCH,
        territory_specs=plan.specs,
        topic=topic,
        agent=agent,
        llm_prefer=llm_prefer,
        allow_iteration=True,
        fleet_warning=plan.warning,
    )


def run_multi_territory_research(
    question: str,
    *,
    protocol: Protocol,
    territory_specs: list[tuple[AgentRole, str, bool, WorkerModel]],
    topic: str | None = None,
    agent: str | None = None,
    llm_prefer: str | None = None,
    allow_iteration: bool = False,
    fleet_warning: str | None = None,
) -> SearchRunResult:
    topic = topic or slugify_question(question)
    agent = agent or os.environ.get("RESEARCH_AGENT") or "harness"
    router = load_router_or_none()
    authority_topic: str | None = None
    require_tier_1 = False
    if router is not None:
        try:
            route = router.route(question)
            authority_topic = route.topic
            require_tier_1 = bool(getattr(route, "require_tier_1", False))
        except Exception as exc:
            logger.warning(
                "router route failed; disabling authority topic and Tier-1 enforcement: %s",
                exc,
            )
            authority_topic = None
            require_tier_1 = False
    scout_attempt = run_gemini_scout(
        question,
        router=router,
        protocol=protocol,
        topic=topic,
    )
    provider = provider_for_question(question)
    territory_count = len(territory_specs)
    sub_questions = decompose_question(
        question,
        territory_count,
        llm_prefer=llm_prefer,
        scout_context=scout_attempt.output_text if scout_attempt.success else None,
    )
    territories = build_territories(
        sub_questions,
        territory_specs,
        provider=provider,
        protocol=protocol,
        original_question=question,
    )
    seen_urls: set[str] = set()
    runs: list[TerritoryRun] = []
    workers_run = 0

    for territory, (_role, _lane, counter, _worker_model) in zip(
        territories,
        territory_specs,
        strict=True,
    ):
        run = execute_territory(
            question,
            territory,
            protocol=protocol,
            topic=topic,
            agent=agent,
            provider=provider,
            router=router,
            counter=counter,
            seen_urls=seen_urls,
            llm_prefer=llm_prefer,
            authority_topic=authority_topic,
            require_tier_1=require_tier_1,
        )
        runs.append(run)
        workers_run += 1

    saved_session, path, backend, gemini_unavailable = build_and_save_research_session(
        question,
        protocol=protocol,
        agent=agent,
        territories=territories,
        runs=runs,
        llm_prefer=llm_prefer,
        scout_attempt=scout_attempt,
        router=router,
        topic=topic,
    )
    if gemini_unavailable:
        telemetry_safely()
        return SearchRunResult(
            session=saved_session,
            path=path,
            backend=backend,
            n_sources=len(saved_session.sources),
            fleet_warning=fleet_warning,
        )

    if (
        allow_iteration
        and workers_run < max(6, len(territories) + 1)
        and needs_deep_iteration(saved_session)
    ):
        weakest = weakest_territory_run(runs)
        focused_query = f"{weakest.territory.description} focused follow-up"
        extra_sources, extra_texts, extra_queries = _run_worker(
            focused_query,
            protocol=protocol.value,
            topic=topic,
            agent=agent,
            provider=provider,
            worker_model=weakest.territory.assigned_worker_model,
            counter=weakest.counter,
            seen_urls=seen_urls,
            authority_topic=authority_topic,
            require_tier_1=require_tier_1,
        )
        workers_run += 1
        extra_summary = ""
        if extra_sources:
            extra_summary = summarize_territory_safely(
                question,
                weakest.territory,
                list(zip(extra_sources, extra_texts, strict=True)),
                llm_prefer=llm_prefer,
            )
        runs.append(
            TerritoryRun(
                territory=weakest.territory,
                sources=extra_sources,
                full_texts=extra_texts,
                queries_run=extra_queries,
                summary=extra_summary,
                counter=weakest.counter,
            )
        )
        saved_session, path, backend, gemini_unavailable = build_and_save_research_session(
            question,
            protocol=protocol,
            agent=agent,
            territories=territories,
            runs=runs,
            llm_prefer=llm_prefer,
            iteration_count=1,
            scout_attempt=scout_attempt,
            router=router,
            topic=topic,
        )
        if gemini_unavailable:
            telemetry_safely()
            return SearchRunResult(
                session=saved_session,
                path=path,
                backend=backend,
                n_sources=len(saved_session.sources),
                fleet_warning=fleet_warning,
            )

    telemetry_safely()
    return SearchRunResult(
        session=saved_session,
        path=path,
        backend=backend,
        n_sources=len(saved_session.sources),
        fleet_warning=fleet_warning,
    )


def slugify_question(question: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    return slug[:80].strip("-") or "search"


def choose_provider(lanes: list[str] | tuple[str, ...]) -> str:
    for lane in lanes:
        lane_text = str(lane).lower()
        for provider in SPECIALIZED_PROVIDERS:
            if provider in lane_text:
                return provider
    return "tavily"


def provider_lane(provider: str) -> str:
    mapping = {
        "tavily": "tavily_direct",
        "linkup": "linkup_direct",
        "exa": "exa_direct",
        "youcom": "youcom_direct",
    }
    return mapping.get(provider, provider)


def provider_for_question(question: str) -> str:
    try:
        route = load_router().route(question)
        return choose_provider(getattr(route, "lanes", []))
    except Exception as exc:  # noqa: BLE001 - router failure degrades to default provider
        logger.warning(
            "router provider selection failed; falling back to tavily: %s",
            exc,
        )
        return "tavily"


def load_router_or_none():
    try:
        return load_router()
    except Exception as exc:
        logger.warning("router load failed; continuing without router: %s", exc)
        return None


def protocol_for_mode(mode: str) -> Protocol:
    if mode == "research":
        return Protocol.RESEARCH
    if mode == "deep-research":
        return Protocol.DEEP_RESEARCH
    return Protocol.SEARCH


RESEARCH_FLEET_NAME = "research"
DEEP_RESEARCH_FLEET_NAME = "deep_research"

RESEARCH_LANE_PLAN: tuple[tuple[AgentRole, str, bool], ...] = (
    (AgentRole.KEYWORD, "github_code", False),
    (AgentRole.DOMAIN_SPECIALIST, "reddit_rss", False),
    (AgentRole.COUNTER_EVIDENCE, "counter_evidence", True),
)

DEEP_RESEARCH_LANE_PLAN: tuple[tuple[AgentRole, str, bool], ...] = (
    *((AgentRole.SEMANTIC, lane, False) for lane in ("arxiv", "semantic_scholar", "core", "papers_with_code", "pubmed")),
    *((AgentRole.KEYWORD, lane, False) for lane in ("github_code", "sourcegraph", "stack_exchange", "github_code", "sourcegraph")),
    *((AgentRole.DOMAIN_SPECIALIST, lane, False) for lane in ("reddit_rss", "reddit_failures", "x_pulse", "bluesky_jetstream", "hn_algolia")),
    (AgentRole.COUNTER_EVIDENCE, "counter_evidence", True),
)


@dataclass(frozen=True)
class FleetPlan:
    specs: list[tuple[AgentRole, str, bool, WorkerModel]]
    warning: str | None


def fleet_plan(
    fleet_name: str,
    lane_plan: tuple[tuple[AgentRole, str, bool], ...],
    *,
    router=None,
) -> FleetPlan:
    router = router if router is not None else load_router()
    raw_models = router.fleet_worker_models(fleet_name)
    if len(raw_models) != len(lane_plan):
        raise ValueError(
            f"fleet '{fleet_name}' declares {len(raw_models)} workers but "
            f"the lane plan has {len(lane_plan)} slots"
        )

    models = [WorkerModel(value) for value in raw_models]
    specs = [
        (role, lane, counter, model)
        for (role, lane, counter), model in zip(lane_plan, models, strict=True)
    ]
    return FleetPlan(specs=specs, warning=None)


def _fleet_warning(fleet_name: str, exc: Exception) -> str:
    return f"fleet '{fleet_name}' router fleets config unusable: {type(exc).__name__}: {exc}"


def research_fleet_plan(*, router=None) -> FleetPlan:
    try:
        return fleet_plan(RESEARCH_FLEET_NAME, RESEARCH_LANE_PLAN, router=router)
    except Exception as exc:
        warning = _fleet_warning(RESEARCH_FLEET_NAME, exc)
        logger.error("ROUTER FLEETS ERROR - %s", warning)
        return FleetPlan(specs=[], warning=warning)


def deep_research_fleet_plan(*, router=None) -> FleetPlan:
    try:
        return fleet_plan(DEEP_RESEARCH_FLEET_NAME, DEEP_RESEARCH_LANE_PLAN, router=router)
    except Exception as exc:
        warning = _fleet_warning(DEEP_RESEARCH_FLEET_NAME, exc)
        logger.error("ROUTER FLEETS ERROR - %s", warning)
        return FleetPlan(specs=[], warning=warning)


def research_territory_specs(
    *, router=None
) -> list[tuple[AgentRole, str, bool, WorkerModel]]:
    return research_fleet_plan(router=router).specs


def deep_research_territory_specs(
    *, router=None
) -> list[tuple[AgentRole, str, bool, WorkerModel]]:
    return deep_research_fleet_plan(router=router).specs


def run_gemini_scout(
    question: str,
    *,
    router,
    protocol: Protocol,
    topic: str,
) -> GeminiInterlockAttempt:
    try:
        spec = dispatch_scout(question, router, protocol=protocol, topic_slug=topic)
        if spec is None:
            return GeminiInterlockAttempt(
                run_type=GeminiProRunKind.SCOUT,
                failure_reason="Gemini scout skipped after failed health check",
            )
        output_text, actual_model_id = _execute_agy_worker_spec(spec, router=router)
    except Exception as exc:  # noqa: BLE001 - scout failure must degrade to fallback
        return failed_gemini_attempt(GeminiProRunKind.SCOUT, exc)
    return GeminiInterlockAttempt(
        run_type=GeminiProRunKind.SCOUT,
        record=successful_gemini_record(
            GeminiProRunKind.SCOUT,
            spec=spec,
            model_id=actual_model_id,
        ),
        output_text=output_text,
    )


def run_gemini_pro_synthesis_fallback(
    question: str,
    *,
    router,
    protocol: Protocol,
    topic: str,
    runs: list[TerritoryRun],
) -> GeminiInterlockAttempt:
    del router, topic
    return run_final_synthesis(
        question,
        protocol=protocol,
        runs=runs,
        run_type=GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
    )


def run_final_synthesis(
    question: str,
    *,
    protocol: Protocol,
    runs: list[TerritoryRun],
    run_type: GeminiProRunKind,
) -> GeminiInterlockAttempt:
    prompt = final_research_prompt(
        question,
        territory_summaries=[run.summary for run in runs],
        worker_disagreements=detect_worker_disagreements(question, runs),
        source_pairs=source_pairs_from_runs(runs),
        worker_output_paths=worker_output_paths_from_runs(runs),
    )
    failures: list[str] = []
    for backend, model, model_id in FINAL_SYNTHESIS_CHAIN:
        try:
            output_text, _backend = llm_call.llm_complete_with_backend(
                prompt,
                backend=backend,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001 - final synthesis failure must degrade through the chain
            label = backend if model is None else f"{backend}({model})"
            failures.append(f"{label}: {type(exc).__name__}: {trim_for_log(str(exc))}")
            continue
        return GeminiInterlockAttempt(
            run_type=run_type,
            record=successful_final_synthesis_record(run_type, model_id=model_id),
            output_text=output_text,
        )
    return GeminiInterlockAttempt(
        run_type=run_type,
        failure_reason=" | ".join(failures) or "no final synthesis backends configured",
    )


def run_gemini_final_synthesis(
    question: str,
    *,
    router,
    protocol: Protocol,
    topic: str,
    runs: list[TerritoryRun],
) -> GeminiInterlockAttempt:
    del router, topic
    return run_final_synthesis(
        question,
        protocol=protocol,
        runs=runs,
        run_type=GeminiProRunKind.FINAL_SYNTHESIS,
    )


def run_grok_x_search(
    question: str,
    *,
    protocol: Protocol,
    topic: str,
    router,
) -> tuple[str, QueryCall]:
    territory = grok_x_search_territory(question)
    spec = dispatch(territory, router, topic_slug=topic, protocol=protocol)
    Path(spec.brief_path).write_text(
        grok_x_search_brief(question, protocol=protocol),
        encoding="utf-8",
    )
    output_text = ""
    error = None
    try:
        output_text = execute_grok_worker_spec(spec)
    except Exception as exc:  # noqa: BLE001 - X-search worker failure should not kill search
        error = f"{type(exc).__name__}: {trim_for_log(str(exc))}"
    return output_text.strip(), grok_x_query_call(
        question,
        spec=spec,
        output_text=output_text,
        error=error,
    )


def grok_x_search_territory(question: str) -> Territory:
    return Territory(
        territory_id="grok-x",
        description=f"Live X/Twitter search for {question}",
        queries=[question],
        assigned_agent_role=AgentRole.DOMAIN_SPECIALIST,
        assigned_lanes=["grok_x_search", "x_pulse"],
        assigned_worker_model=WorkerModel.GROK,
    )


def grok_x_search_brief(question: str, *, protocol: Protocol) -> str:
    return "\n".join(
        [
            f"You are the Grok X-search worker for {protocol.value}.",
            "Use live X/Twitter search for the topic. Focus on posts, handles, dates, and cited evidence.",
            "Return compact bullets. Each factual bullet must include a URL citation or a verbatim quote.",
            "For each useful post capture: handle, date, one-line summary, and citation.",
            "Do not speculate. If X has no useful signal, say that plainly with the search terms used.",
            "",
            f"Topic: {question}",
        ]
    ).strip()


def grok_x_query_call(
    query: str,
    *,
    spec: WorkerSpec,
    output_text: str,
    error: str | None,
) -> QueryCall:
    return QueryCall(
        query_text=query,
        lane="grok_x_search",
        worker_model=WorkerModel.GROK,
        started_at=datetime.now(timezone.utc),
        duration_ms=0,
        result_count=1 if output_text.strip() else 0,
        error=error,
    )


def execute_gemini_worker_spec(spec: WorkerSpec, *, router) -> str:
    output_text, _model_id = _execute_agy_worker_spec(spec, router=router)
    return output_text


def _execute_agy_worker_spec(spec: WorkerSpec, *, router) -> tuple[str, str]:
    prompt = Path(spec.brief_path).read_text(encoding="utf-8")
    if prompt.lstrip().startswith("-"):
        prompt = f"Brief:\n{prompt}"
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > 200_000:
        raise GeminiProScoutError(
            f"brief too large for agy argv: {prompt_bytes} bytes; trim the brief"
        )

    output_path = Path(spec.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requested_model = spec.model_id or AGY_INTERACTIVE_GEMINI_MODEL
    model_id = model_for_agy_worker(spec)
    budget_reserved = False
    if is_gemini_quota_model(model_id):
        budget_reserved = reserve_gemini_daily_budget(path=GEMINI_DAILY_COUNTER_FILE)
        if not budget_reserved:
            model_id = resolve_agy_model(
                requested_model,
                gemini_budget_available=False,
            )
    successful_exit = False
    try:
        completed = subprocess.run(
            [
                GEMINI_CLI_HOME,  # dispatcher.AGY_CLI is the locked canonical agy command.
                AGY_SKIP_PERMISSIONS_FLAG,
                "-p",
                prompt,
                "--model",
                model_id,
            ],
            text=True,
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=gemini_timeout_for_spec(spec, router=router),
        )
        successful_exit = completed.returncode == 0
    finally:
        if budget_reserved:
            finalize_gemini_daily_budget(
                success=successful_exit,
                path=GEMINI_DAILY_COUNTER_FILE,
            )
    stdout_text = completed.stdout.strip()
    stderr_text = completed.stderr.strip()
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ).strip()
    if completed.returncode != 0:
        raise GeminiProScoutError(
            f"agy Gemini failed with exit={completed.returncode}: "
            f"{trim_for_log(combined_output) or 'no output'}"
        )
    if stderr_text:
        logger.warning("agy Gemini worker stderr: %s", trim_for_log(stderr_text))
    if not stdout_text:
        raise GeminiProScoutError("agy Gemini returned empty output")
    if _looks_like_worker_failure(stdout_text):
        raise GeminiProScoutError(
            f"agy Gemini returned failure stub: {trim_for_log(stdout_text)}"
        )
    stdout_text = apply_anti_hallucination_gate(
        stdout_text,
        label="gemini_worker",
    )
    output_path.write_text(stdout_text + "\n", encoding="utf-8")
    return stdout_text, model_id


def gemini_timeout_for_spec(spec: WorkerSpec, *, router) -> int:
    if "gemini_pro_scout" in spec.lanes:
        config = router_scout_config(router)
        return max(
            int(config.get("health_check_timeout_seconds", GEMINI_TIMEOUT_SECONDS)),
            GEMINI_TIMEOUT_SECONDS,
        )
    return GEMINI_TIMEOUT_SECONDS


def execute_codex_worker_spec(spec: WorkerSpec) -> str:
    prompt = Path(spec.brief_path).read_text(encoding="utf-8")
    output_path = Path(spec.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codex_bin = paths.require_executable(paths.CODEX_BIN_ENV, "codex")
    completed = subprocess.run(
        [
            codex_bin,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-m",
            CODEX_MODEL_ID,
        ],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=CODEX_TIMEOUT_SECONDS,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex worker failed with exit={completed.returncode}: "
            f"{trim_for_log(combined_output) or 'no output'}"
        )
    if not combined_output:
        raise RuntimeError("Codex worker returned empty output")
    output_path.write_text(combined_output + "\n", encoding="utf-8")
    return combined_output


def execute_mistral_worker_spec(spec: WorkerSpec) -> str:
    prompt = Path(spec.brief_path).read_text(encoding="utf-8")
    output_path = Path(spec.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = next_mistral_free_key()
    payload = {
        "model": MISTRAL_FREE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "tools": mistral_worker_tools(),
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 900,
    }
    request = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=MISTRAL_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Mistral free API failed with http={exc.code}: {trim_for_log(body)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Mistral free API unavailable: {exc.reason}") from exc

    try:
        data = json.loads(raw)
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Mistral free API returned an unexpected response shape") from exc

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        summary = summarize_mistral_tool_calls(tool_calls)
    else:
        summary = str(message.get("content") or "").strip()
    if not summary:
        raise RuntimeError("Mistral free API returned empty output")
    summary = apply_anti_hallucination_gate(summary, label="mistral_worker")
    output_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def next_mistral_free_key() -> str:
    if not MISTRAL_FREE_KEYS_PATH.is_file():
        raise RuntimeError(f"Mistral free key file not found: {MISTRAL_FREE_KEYS_PATH}")
    keys: list[str] = []
    for raw_line in MISTRAL_FREE_KEYS_PATH.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().startswith("MISTRAL_FREE_KEY_") and value.strip():
            keys.append(value.strip().strip('"').strip("'"))
    if not keys:
        raise RuntimeError(f"No MISTRAL_FREE_KEY_* entries found in {MISTRAL_FREE_KEYS_PATH}")
    return keys[(os.getpid() + next(_MISTRAL_KEY_COUNTER)) % len(keys)]


def mistral_worker_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "emit_grounded_summary",
                "description": "Return the grounded territory summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Concise answer grounded in the provided sources.",
                        },
                        "weak_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Evidence gaps, disagreements, or caveats.",
                        },
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def summarize_mistral_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    first_call = tool_calls[0]
    function = first_call.get("function") or {}
    raw_arguments = str(function.get("arguments") or "{}")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments.strip()
    summary = str(arguments.get("summary") or "").strip()
    weak_evidence = arguments.get("weak_evidence") or []
    if isinstance(weak_evidence, list) and weak_evidence:
        caveats = "\n".join(
            f"- {str(item).strip()}"
            for item in weak_evidence
            if str(item).strip()
        )
        if caveats:
            return f"{summary}\n\nCaveats/disagreements:\n{caveats}".strip()
    return summary


def execute_grok_worker_spec(spec: WorkerSpec) -> str:
    if spec.model_id is None:
        raise ValueError("Grok worker spec is missing model_id")
    prompt = Path(spec.brief_path).read_text(encoding="utf-8")
    output_path = Path(spec.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            paths.executable(paths.GROK_BIN_ENV, "grok") or "grok",
            "--single",
            prompt,
        ],
        text=True,
        capture_output=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=GROK_TIMEOUT_SECONDS,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Hermes Grok failed with exit={completed.returncode}: "
            f"{trim_for_log(combined_output) or 'no output'}"
        )
    if not combined_output:
        raise RuntimeError("Hermes Grok returned empty output")
    combined_output = apply_anti_hallucination_gate(
        combined_output,
        label="grok_worker",
    )
    output_path.write_text(combined_output + "\n", encoding="utf-8")
    return combined_output


def router_scout_config(router) -> dict[str, Any]:
    method = getattr(router, "scout_config", None)
    if callable(method):
        config = method()
        if isinstance(config, dict):
            return config
    config = getattr(router, "config", None)
    if isinstance(config, dict):
        scout = config.get("scout")
        if isinstance(scout, dict):
            return scout
    return {}


def successful_gemini_record(
    run_type: GeminiProRunKind,
    *,
    spec: WorkerSpec,
    model_id: str,
) -> GeminiProRunRecord:
    return GeminiProRunRecord(
        run_type=run_type,
        success=True,
        model_id=model_id,
        brief_path=Path(spec.brief_path),
        output_path=Path(spec.output_path),
    )


def successful_final_synthesis_record(
    run_type: GeminiProRunKind,
    *,
    model_id: str,
) -> GeminiProRunRecord:
    return GeminiProRunRecord(
        run_type=run_type,
        success=True,
        model_id=model_id,
    )


def failed_gemini_attempt(
    run_type: GeminiProRunKind,
    exc: BaseException,
) -> GeminiInterlockAttempt:
    return GeminiInterlockAttempt(
        run_type=run_type,
        failure_reason=f"{type(exc).__name__}: {trim_for_log(str(exc))}",
    )


def trim_for_log(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def apply_anti_hallucination_gate(text: str, *, label: str) -> str:
    validated, dropped, flagged = validate_model_output(text)
    if not flagged:
        return validated
    gate_lines = [
        "[anti-hallucination-gate]",
        f"label={label}",
        f"dropped_claims={dropped}",
    ]
    gate_lines.extend(f"flagged={item}" for item in flagged[:5])
    if len(flagged) > 5:
        gate_lines.append(f"flagged_more={len(flagged) - 5}")
    if validated.strip():
        return f"{validated.strip()}\n\n" + "\n".join(gate_lines)
    return "\n".join(gate_lines)


def run_logged_search(search_fn, question: str, *, errors: list[str], **kwargs) -> dict[str, Any]:
    try:
        payload = search_fn(question, **kwargs)
    except Exception as exc:  # noqa: BLE001 - search failure becomes empty payload
        errors.append(f"search failed: {type(exc).__name__}: {exc}")
        return {"results": [], "error": str(exc)}
    if not isinstance(payload, dict):
        errors.append("search returned a non-object payload")
        return {"results": [], "error": "non-object payload"}
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        errors.append(f"search degraded: {error.strip()}")
    elif not _results(payload):
        errors.append("search degraded: 0 results")
    return payload


def run_free_api_lanes(
    lanes: list[str],
    *,
    router,
    question: str,
    protocol: str,
    topic: str,
    agent: str,
    errors: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    if router is None:
        return []
    lane_endpoint = getattr(router, "lane_endpoint", None)
    if not callable(lane_endpoint):
        return []

    payloads: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for lane in lanes:
        if lane in seen:
            continue
        seen.add(lane)
        try:
            lane_config = lane_endpoint(lane)
        except Exception as exc:  # noqa: BLE001 - bad lane config should not abort search
            errors.append(f"{lane} api lane config failed: {type(exc).__name__}: {exc}")
            continue
        if not is_free_api_lane(lane_config):
            continue
        try:
            request = build_api_lane_request(lane, lane_config, question)
            payload = logged_search.api_lane(
                lane,
                request,
                protocol=protocol,
                topic=topic,
                agent=agent,
            )
        except Exception as exc:  # noqa: BLE001 - one lane should not abort search
            errors.append(f"{lane} api lane failed: {type(exc).__name__}: {exc}")
            continue
        payloads.append((lane, payload))
    return payloads


def run_local_search_lanes(
    lanes: list[str],
    *,
    router,
    question: str,
    errors: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    if router is None:
        return []
    lane_endpoint = getattr(router, "lane_endpoint", None)
    if not callable(lane_endpoint):
        return []

    payloads: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for lane in lanes:
        if lane in seen:
            continue
        seen.add(lane)
        try:
            lane_config = lane_endpoint(lane)
        except Exception as exc:  # noqa: BLE001 - bad lane config should not abort search
            errors.append(f"{lane} local lane config failed: {type(exc).__name__}: {exc}")
            continue
        if str(lane_config.get("type")) != "local":
            continue
        try:
            payload = local_lane_payload(lane_config, question)
        except Exception as exc:  # noqa: BLE001 - one lane should not abort search
            errors.append(f"{lane} local lane failed: {type(exc).__name__}: {exc}")
            payload = {"results": [], "error": str(exc)}
        payloads.append((lane, payload))
    return payloads


def local_lane_payload(lane_config: dict[str, Any], question: str) -> dict[str, Any]:
    env_var = str(lane_config.get("env_var") or "").strip()
    db_path = str(lane_config.get("db_path") or "").strip()
    if db_path:
        db_file = Path(db_path).expanduser()
        if not db_file.is_file():
            error = (
                f"local lane not configured: missing {db_file}; set {env_var}"
                if env_var
                else f"local lane not configured: missing {db_file}"
            )
            return {"results": [], "error": error, "not_configured": True}
        return {"results": _local_db_results(db_file, question)}
    glob_pattern = str(lane_config.get("glob") or "").strip()
    if glob_pattern:
        root = paths.glob_root(glob_pattern)
        if not root.exists():
            error = (
                f"local lane not configured: missing {root}; set {env_var}"
                if env_var
                else f"local lane not configured: missing {root}"
            )
            return {"results": [], "error": error, "not_configured": True}
        return {"results": _local_file_results(glob_pattern, question)}
    return {"results": [], "error": "unsupported local lane config"}


def _local_db_results(db_path: Path, question: str) -> list[dict[str, Any]]:
    if not db_path.is_file():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, project, topic, content
                FROM memory_records
                WHERE stale = 0
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (LOCAL_MEMORY_SCAN_LIMIT,),
            ).fetchall()
    except sqlite3.Error:
        return []

    scored: list[tuple[float, int, str, str]] = []
    for record_id, project, topic, content in rows:
        text = "\n".join(
            [
                f"Project: {project}",
                f"Topic: {topic}",
                str(content or "").strip(),
            ]
        ).strip()
        if not text:
            continue
        paragraph, score = best_supporting_paragraph(text, question)
        if score <= 0.0:
            continue
        scored.append((score, int(record_id), text[:LOCAL_SOURCE_TEXT_LIMIT], paragraph))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results: list[dict[str, Any]] = []
    for score, record_id, text, paragraph in scored[:LOCAL_LANE_RESULT_LIMIT]:
        results.append(
            {
                "url": _materialize_local_memory_record(record_id, text).as_uri(),
                "snippet": paragraph,
                "score": score,
            }
        )
    return results


def _materialize_local_memory_record(record_id: int, text: str) -> Path:
    LOCAL_MEMORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_MEMORY_CACHE_DIR / f"memory-record-{record_id}.md"
    path.write_text(text, encoding="utf-8")
    return path.resolve()


def _local_file_results(glob_pattern: str, question: str) -> list[dict[str, Any]]:
    scored: list[tuple[float, Path, str]] = []
    for raw_path in glob.glob(os.path.expanduser(glob_pattern), recursive=True):
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        paragraph, score = best_supporting_paragraph(text[:LOCAL_SOURCE_TEXT_LIMIT], question)
        if score <= 0.0:
            continue
        scored.append((score, path.resolve(), paragraph))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [
        {"url": path.as_uri(), "snippet": paragraph, "score": score}
        for score, path, paragraph in scored[:LOCAL_LANE_RESULT_LIMIT]
    ]


def is_free_api_lane(lane_config: dict[str, Any]) -> bool:
    if str(lane_config.get("type")) != "api":
        return False
    try:
        cost = float(lane_config.get("cost_per_call_usd"))
    except (TypeError, ValueError):
        return False
    return cost <= 0.0


def _run_worker(
    query: str,
    *,
    protocol: str,
    topic: str,
    agent: str,
    provider: str,
    worker_model: WorkerModel,
    counter: bool = False,
    seen_urls: set[str],
    authority_topic: str | None = None,
    require_tier_1: bool = False,
) -> tuple[list[SourceRecord], list[str], list[QueryCall]]:
    errors: list[str] = []
    payloads: list[tuple[str, str, dict[str, Any]]] = []
    worker_queries = [query]
    if counter:
        worker_queries.append(f"{query} failure OR problem OR criticism OR limitation OR bug")

    for worker_query in worker_queries:
        sx = run_logged_search(
            logged_search.searxng,
            worker_query,
            protocol=protocol,
            topic=topic,
            agent=agent,
            errors=errors,
        )
        px = run_logged_search(
            logged_search.proxy,
            worker_query,
            provider=provider,
            protocol=protocol,
            topic=topic,
            agent=agent,
            errors=errors,
        )
        payloads.append((worker_query, "searxng_general", sx))
        payloads.append((worker_query, provider, px))

    queries_run = [
        query_call(worker_query, lane=lane, payload=payload, worker_model=worker_model)
        for worker_query, lane, payload in payloads
    ]
    urls = candidate_urls(*(payload for _worker_query, _lane, payload in payloads))
    source_pairs = extract_sources(
        urls,
        errors=errors,
        max_sources=2,
        seen_urls=seen_urls,
        counter_evidence=counter,
        topic=authority_topic,
        require_tier_1=require_tier_1,
    )
    return (
        [source for source, _full_text in source_pairs],
        [full_text for _source, full_text in source_pairs],
        queries_run,
    )


def query_call(
    query: str,
    *,
    lane: str,
    payload: dict[str, Any],
    worker_model: WorkerModel,
) -> QueryCall:
    return QueryCall(
        query_text=query,
        lane=lane,
        worker_model=worker_model,
        started_at=datetime.now(timezone.utc),
        duration_ms=0,
        result_count=len(_results(payload)),
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
    )


def candidate_urls(*payloads: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        for result in _results(payload):
            url = str(result.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def has_enough_free_results(
    payload: dict[str, Any],
    *,
    min_results: int = MIN_FREE_SEARCH_RESULTS,
) -> bool:
    return len(_results(payload)) >= min_results


def extract_sources(
    urls: list[str],
    *,
    errors: list[str],
    max_sources: int = 3,
    seen_urls: set[str] | None = None,
    counter_evidence: bool = False,
    topic: str | None = None,
    require_tier_1: bool = False,
) -> list[tuple[SourceRecord, str]]:
    source_pairs: list[tuple[SourceRecord, str]] = []
    for url in urls:
        if len(source_pairs) >= max_sources:
            break
        if seen_urls is not None and url in seen_urls:
            continue
        try:
            requested_tier = authority_tier(
                urlparse(url).netloc or "unknown",
                topic=topic,
            )
            extracted = extract_clean_text(url, tier=requested_tier)
            if not extracted or int(extracted.get("char_count") or 0) <= 0:
                continue
            if not extracted.get("url"):
                extracted = {**extracted, "url": url}
            raw_text_path = Path(str(extracted["raw_text_path"]))
            full_text = raw_text_path.read_text(encoding="utf-8", errors="ignore")
            if not full_text.strip():
                continue
            record = source_record(
                extracted,
                full_text,
                counter_evidence=counter_evidence,
                topic=topic,
                require_tier_1=require_tier_1,
            )
            if require_tier_1 and record.tier != SourceTier.T1:
                errors.append(
                    f"tier-1 required but {record.domain} scored {record.tier.name} "
                    f"(authority={record.topic_authority_score:.2f})"
                )
            source_pairs.append(
                (
                    record,
                    full_text,
                )
            )
            if seen_urls is not None:
                seen_urls.add(url)
        except Exception as exc:  # noqa: BLE001 - source-level failure should not abort run
            errors.append(f"extract failed for {url}: {type(exc).__name__}: {exc}")
    return source_pairs


def source_record(
    extracted: dict[str, Any],
    full_text: str,
    *,
    counter_evidence: bool = False,
    topic: str | None = None,
    require_tier_1: bool = False,
) -> SourceRecord:
    try:
        method = ExtractionMethod(str(extracted.get("extraction_method") or ""))
    except ValueError:
        method = ExtractionMethod.CURL
    domain = str(extracted.get("domain") or "unknown")
    authority_score = source_authority_score(domain, topic)
    tier = authority_tier(domain, topic=topic, authority_score=authority_score)
    title = str(extracted.get("title") or "Untitled")
    if require_tier_1 and tier != SourceTier.T1 and not title.startswith(NON_T1_SOURCE_MARKER):
        title = f"{NON_T1_SOURCE_MARKER}{title}"
    return SourceRecord(
        url=str(extracted.get("url") or ""),
        domain=domain,
        title=title,
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(full_text),
        extraction_method=method,
        raw_text_path=Path(str(extracted["raw_text_path"])),
        char_count=int(extracted.get("char_count") or len(full_text)),
        tier=tier,
        topic_authority_score=authority_score,
        counter_evidence_flagged=counter_evidence,
    )


def decompose_question(
    question: str,
    count: int,
    *,
    llm_prefer: str | None = None,
    scout_context: str | None = None,
) -> list[str]:
    example_seeking = is_example_seeking_question(question)
    prompt = (
        f"Break this question into exactly {count} non-overlapping sub-questions; "
        f"return ONLY a JSON array of {count} strings."
    )
    prompt = f"{prompt}\n\n{query_phrasing_playbook()}"
    if example_seeking:
        prompt = (
            f"{prompt}\nFor FIND/NAME/LIST/WHICH example questions, produce concrete "
            "example-seeking search queries. Hunt named examples, lists, named cases, "
            "journalist/founder/VC quotes, and source-backed industries. Do not spend "
            "more than one item on definitions, criteria, formatting, or what the "
            "question means."
        )
    prompt = f"{prompt}\n\nQuestion: {question}"
    if scout_context:
        prompt = (
            f"{prompt}\n\nGemini scout context for decomposition only:\n"
            f"{scout_context[:4000]}"
        )
    try:
        raw, _backend = llm_call.llm_complete(prompt, prefer=llm_prefer)
        parsed = parse_subquestion_array(raw, count)
        if parsed:
            if example_seeking:
                return example_seeking_subquestions(question, count, parsed)
            return parsed
    except Exception:  # noqa: BLE001 - decomposition failure uses deterministic fallback
        pass
    if example_seeking:
        return example_seeking_subquestions(question, count, [])
    return fallback_subquestions(question, count)


def parse_subquestion_array(raw: str, count: int) -> list[str] | None:
    candidates = [raw.strip()]
    bracketed = re.search(r"\[[\s\S]*\]", raw)
    if bracketed:
        candidates.append(bracketed.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            candidate = re.sub(r"```$", "", candidate).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list) or len(parsed) != count:
            continue
        values = [" ".join(str(item).split()) for item in parsed if str(item).strip()]
        if len(values) == count:
            return values
    return None


def fallback_subquestions(question: str, count: int) -> list[str]:
    base = [
        question,
        f"{question} alternatives/options",
        f"{question} problems/criticism",
        f"{question} academic evidence",
        f"{question} social evidence",
    ]
    while len(base) < count:
        base.append(f"{question} evidence angle {len(base) + 1}")
    return base[:count]


def is_example_seeking_question(question: str) -> bool:
    normalized = _normalize_for_topic(question)
    action_markers = (
        "name ",
        "list ",
        "find ",
        "which ",
        "what are ",
        "identify ",
        "give me ",
        "examples",
        "instances",
        "cases",
    )
    target_markers = (
        "companies",
        "company",
        "startups",
        "startup",
        "products",
        "product",
        "examples",
        "instances",
        "cases",
        "often called",
        "called",
    )
    return any(marker in normalized for marker in action_markers) and any(
        marker in normalized for marker in target_markers
    )


def example_seeking_subquestions(
    question: str,
    count: int,
    candidates: list[str],
) -> list[str]:
    direct: list[str] = []
    for candidate in candidates:
        if is_meta_subquestion(candidate):
            continue
        if candidate not in direct:
            direct.append(candidate)

    generated = generated_example_queries(question)
    for query in generated:
        if query not in direct:
            direct.append(query)
        if len(direct) >= count:
            return direct[:count]
    while len(direct) < count:
        direct.append(f"{question} named examples evidence source {len(direct) + 1}")
    return direct[:count]


def is_meta_subquestion(question: str) -> bool:
    normalized = _normalize_for_topic(question)
    meta_markers = (
        "what does ",
        "what counts ",
        "what qualifies ",
        "what is meant ",
        "in this context",
        "definition",
        "meaning",
        "criteria",
        "how should the answer",
        "should the answer",
        "how should this be formatted",
        "what wording best",
        "broad industries or narrower",
    )
    return any(marker in normalized for marker in meta_markers)


def generated_example_queries(question: str) -> list[str]:
    normalized_question = (
        question.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    quoted_phrases = re.findall(r'"([^"]+)"|' "'([^']+)'", normalized_question)
    phrases = [next(part for part in match if part).strip() for match in quoted_phrases]
    phrase = phrases[0] if phrases else ""
    queries = [
        f"{question} named examples",
        f"{question} list of examples",
        f"{question} real named cases",
        f"{question} real-world instances",
    ]
    if phrase:
        queries.extend(
            [
                f'"{phrase}" examples',
                f'"{phrase}" list',
                f'"{phrase}" named cases',
                f'"{phrase}" real-world instances',
            ]
        )
    else:
        queries.extend(
            [
                f"{question} documented examples",
                f"{question} specific cases with names",
            ]
        )
    return queries


def classify_question_topic(question: str) -> str:
    normalized = _normalize_for_topic(question)
    code_hits = _keyword_hits(
        normalized,
        (
            "github",
            "source code",
            "stack overflow",
            "traceback",
            "exception",
            "sdk",
            "library",
            "function",
            "implementation",
            "code example",
            "api error",
        ),
    )
    academic_hits = _keyword_hits(
        normalized,
        (
            "arxiv",
            "pubmed",
            "doi",
            "peer reviewed",
            "peer-reviewed",
            "journal",
            "paper",
            "clinical trial",
            "randomized",
            "cohort",
            "meta analysis",
            "study",
        ),
    )
    business_hits = _keyword_hits(
        normalized,
        (
            "company",
            "companies",
            "startup",
            "startups",
            "founder",
            "vc",
            "venture",
            "investor",
            "industry",
            "market",
            "business model",
            "product",
            "well-known",
            "uber of",
            "uber for",
        ),
    )
    if business_hits and not code_hits and not academic_hits:
        return QUESTION_TOPIC_BUSINESS
    if code_hits and not academic_hits and not business_hits:
        return QUESTION_TOPIC_CODE
    if academic_hits and not code_hits and not business_hits:
        return QUESTION_TOPIC_ACADEMIC
    if code_hits or academic_hits:
        return QUESTION_TOPIC_MIXED
    if business_hits:
        return QUESTION_TOPIC_BUSINESS
    return QUESTION_TOPIC_GENERAL


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _normalize_for_topic(text: str) -> str:
    return " ".join(text.lower().replace("'", "").split())


def build_territories(
    sub_questions: list[str],
    specs: list[tuple[AgentRole, str, bool, WorkerModel]],
    *,
    provider: str,
    protocol: Protocol = Protocol.RESEARCH,
    original_question: str | None = None,
) -> list[Territory]:
    territories: list[Territory] = []
    topic = classify_question_topic(original_question or " ".join(sub_questions))
    for index, (sub_question, (role, lane, _counter, worker_model)) in enumerate(
        zip(sub_questions, specs, strict=True),
        start=1,
    ):
        territory_id = chr(ord("A") + index - 1)
        routed_lane = lane_for_topic(
            topic,
            protocol=protocol,
            index=index,
            fallback_lane=lane,
        )
        territories.append(
            Territory(
                territory_id=territory_id,
                description=sub_question,
                queries=[sub_question],
                assigned_agent_role=role,
                assigned_lanes=territory_lanes(routed_lane, provider),
                assigned_worker_model=worker_model,
                do_not_overlap_with=[
                    chr(ord("A") + other_index - 1)
                    for other_index in range(1, len(specs) + 1)
                    if other_index != index
                ],
            )
        )
    return territories


def lane_for_topic(
    topic: str,
    *,
    protocol: Protocol,
    index: int,
    fallback_lane: str,
) -> str:
    if topic in {QUESTION_TOPIC_BUSINESS, QUESTION_TOPIC_GENERAL}:
        sequence = (
            BUSINESS_DEEP_LANE_SEQUENCE
            if protocol == Protocol.DEEP_RESEARCH
            else BUSINESS_RESEARCH_LANE_SEQUENCE
        )
        return sequence[(index - 1) % len(sequence)]
    if topic == QUESTION_TOPIC_MIXED:
        return MIXED_DEEP_LANE_SEQUENCE[(index - 1) % len(MIXED_DEEP_LANE_SEQUENCE)]
    return fallback_lane


def territory_lanes(primary_lane: str, provider: str) -> list[str]:
    return unique_lanes([primary_lane, "searxng_general", provider_lane(provider)])


def unique_lanes(lanes: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for lane in lanes:
        if lane in seen:
            continue
        seen.add(lane)
        unique.append(lane)
    return unique


def execute_territory(
    question: str,
    territory: Territory,
    *,
    protocol: Protocol,
    topic: str,
    agent: str,
    provider: str,
    router,
    counter: bool,
    seen_urls: set[str],
    llm_prefer: str | None,
    authority_topic: str | None = None,
    require_tier_1: bool = False,
) -> TerritoryRun:
    spec = dispatch(territory, router, topic_slug=topic, protocol=protocol)
    sources, full_texts, queries = _run_worker(
        territory.description,
        protocol=protocol.value,
        topic=topic,
        agent=agent,
        provider=provider,
        worker_model=territory.assigned_worker_model,
        counter=counter,
        seen_urls=seen_urls,
        authority_topic=authority_topic,
        require_tier_1=require_tier_1,
    )
    summary = ""
    source_pairs = list(zip(sources, full_texts, strict=True))
    if spec.provider in {
        "agy_cli",
        "codex_cli",
        "grok_cli",
        "mistral_free_api",
    }:
        Path(spec.brief_path).write_text(
            worker_territory_brief(
                question,
                territory,
                protocol=protocol,
                source_pairs=source_pairs,
                worker_model=territory.assigned_worker_model,
            ),
            encoding="utf-8",
        )
    if spec.provider == "grok_cli":
        worker_output = ""
        worker_error = None
        try:
            worker_output = execute_grok_worker_spec(spec)
        except Exception as exc:  # noqa: BLE001 - one worker failure should not kill the run
            worker_error = f"{type(exc).__name__}: {trim_for_log(str(exc))}"
        queries.append(
            cli_worker_query_call(
                territory.description,
                spec=spec,
                lane="grok_cli",
                worker_model=WorkerModel.GROK,
                output_text=worker_output,
                error=worker_error,
            )
        )
        summary = worker_output.strip()
    elif spec.provider == "agy_cli":
        worker_output = ""
        worker_error = None
        try:
            worker_output = execute_gemini_worker_spec(spec, router=router)
        except Exception as exc:  # noqa: BLE001 - one worker failure should not kill the run
            worker_error = f"{type(exc).__name__}: {trim_for_log(str(exc))}"
        queries.append(
            cli_worker_query_call(
                territory.description,
                spec=spec,
                lane="agy_cli",
                worker_model=WorkerModel.GEMINI_FLASH,
                output_text=worker_output,
                error=worker_error,
            )
        )
        summary = worker_output.strip()
    elif spec.provider == "mistral_free_api":
        worker_output = ""
        worker_error = None
        try:
            worker_output = execute_mistral_worker_spec(spec)
        except Exception as exc:  # noqa: BLE001 - one worker failure should not kill the run
            worker_error = f"{type(exc).__name__}: {trim_for_log(str(exc))}"
        queries.append(
            cli_worker_query_call(
                territory.description,
                spec=spec,
                lane="mistral_free_api",
                worker_model=WorkerModel.MISTRAL,
                output_text=worker_output,
                error=worker_error,
            )
        )
        summary = worker_output.strip()
    elif spec.provider == "codex_cli":
        worker_output = ""
        worker_error = None
        try:
            worker_output = execute_codex_worker_spec(spec)
        except Exception as exc:  # noqa: BLE001 - one worker failure should not kill the run
            worker_error = f"{type(exc).__name__}: {trim_for_log(str(exc))}"
        queries.append(
            cli_worker_query_call(
                territory.description,
                spec=spec,
                lane="codex_cli",
                worker_model=WorkerModel.CODEX_5_4,
                output_text=worker_output,
                error=worker_error,
            )
        )
        summary = worker_output.strip()
    if not summary and sources:
        summary = summarize_territory_safely(
            question,
            territory,
            source_pairs,
            llm_prefer=llm_prefer,
        )
    write_worker_output_note(spec.output_path, territory, summary, sources)
    return TerritoryRun(
        territory=territory,
        sources=sources,
        full_texts=full_texts,
        queries_run=queries,
        summary=summary,
        counter=counter,
        output_path=spec.output_path,
    )


def worker_territory_brief(
    question: str,
    territory: Territory,
    *,
    protocol: Protocol,
    source_pairs: list[tuple[SourceRecord, str]],
    worker_model: WorkerModel,
) -> str:
    if "grok_x_search" in territory.assigned_lanes:
        return grok_x_search_brief(territory.description, protocol=protocol)

    code_forge_lanes = {
        "github_code",
        "gitlab_code",
        "codeberg_code",
        "sourcehut_code",
        "sourcegraph",
        "stack_exchange",
    }
    sections = [
        f"You are the {worker_model.value} worker for {protocol.value}.",
        "Return a concise grounded territory summary.",
        "Name URLs or source titles for concrete claims. Flag disagreements and weak evidence.",
        "Use only the extracted sources below unless your assigned worker has live-search access.",
        query_phrasing_playbook(),
        "",
    ]
    if code_forge_lanes.intersection(territory.assigned_lanes):
        sections.append(
            "When the question touches code, libraries, or engineering, search MULTIPLE forges — GitHub, GitLab, Codeberg, and SourceHut — not GitHub alone."
        )
        sections.append("")
    sections.extend(
        [
            f"Question: {question}",
            f"Territory: {territory.description}",
            "",
            "Existing extracted sources:",
        ]
    )
    if source_pairs:
        sections.extend(numbered_sources(source_pairs, max_chars=1200))
    else:
        sections.append("No extracted source text was available before the Grok pass.")
    return "\n".join(sections).strip()


def cli_worker_query_call(
    query: str,
    *,
    spec: WorkerSpec,
    lane: str,
    worker_model: WorkerModel,
    output_text: str,
    error: str | None,
) -> QueryCall:
    return QueryCall(
        query_text=query,
        lane="grok_x_search" if lane == "grok_cli" and "grok_x_search" in spec.lanes else lane,
        worker_model=worker_model,
        started_at=datetime.now(timezone.utc),
        duration_ms=0,
        result_count=1 if output_text.strip() else 0,
        error=error,
    )


def summarize_territory_safely(
    question: str,
    territory: Territory,
    source_pairs: list[tuple[SourceRecord, str]],
    *,
    llm_prefer: str | None,
) -> str:
    try:
        summary, backend = llm_call.llm_complete(
            territory_summary_prompt(question, territory, source_pairs),
            prefer=llm_prefer,
        )
        if backend == "gemini":
            summary = apply_anti_hallucination_gate(
                summary,
                label="territory_summary_gemini",
            )
        return summary.strip()
    except Exception:  # noqa: BLE001 - a failed territory summary is skipped
        return ""


def territory_summary_prompt(
    question: str,
    territory: Territory,
    source_pairs: list[tuple[SourceRecord, str]],
) -> str:
    sections = [
        "Summarize this territory grounded ONLY in its sources. Use [n] citations.",
        "",
        f"Question: {question}",
        f"Territory: {territory.description}",
        "",
        "Sources:",
    ]
    sections.extend(numbered_sources(source_pairs, max_chars=1200))
    return "\n".join(sections).strip()


def synthesis_prompt(
    question: str,
    source_pairs: list[tuple[SourceRecord, str]],
    *,
    grok_x_summary: str = "",
) -> str:
    sections = [
        "Answer the question grounded ONLY in these sources, cite as [n]; "
        "if the sources don't answer it, say so plainly.",
        query_phrasing_playbook(),
        "",
        f"Question: {question}",
    ]
    if grok_x_summary.strip():
        sections.extend(
            [
                "",
                "Grok X-search worker notes. Use only if the note contains a URL or verbatim quote:",
                grok_x_summary.strip(),
            ]
        )
    sections.extend(["", "Sources:"])
    for index, (source, full_text) in enumerate(source_pairs, start=1):
        sections.append(f"[{index}] {source.title}")
        sections.append(full_text[:1500])
        sections.append("")
    return "\n".join(sections).strip()


def query_phrasing_playbook() -> str:
    return "\n".join(QUERY_PHRASING_PLAYBOOK_LINES)


def final_research_prompt(
    question: str,
    *,
    territory_summaries: list[str],
    worker_disagreements: list[Disagreement] | None = None,
    source_pairs: list[tuple[SourceRecord, str]],
    worker_output_paths: list[str] | None = None,
    scout_summary: str = "",
    parent_summaries: list[str] | None = None,
) -> str:
    sections = [
        "Write the final answer grounded ONLY in the numbered sources.",
        "Use [n] citations after factual claims.",
        "No-tool input contract: use the original question, worker output file "
        "paths, scout summary, parent summaries, territory summaries, and numbered "
        "sources below. Do not start fresh research.",
        "End with one short line that begins: Caveats/disagreements:",
        "",
        f"Question: {question}",
        "",
        "Worker output file paths:",
    ]
    sections.extend(worker_output_paths or ["No worker output files were recorded."])
    if scout_summary.strip():
        sections.extend(["", "Scout summary:", scout_summary.strip()])
    parent_summaries = parent_summaries or []
    if parent_summaries:
        sections.append("")
        sections.append("Parent summaries:")
        sections.extend(
            f"{index}. {summary.strip()}"
            for index, summary in enumerate(parent_summaries, start=1)
            if summary.strip()
        )
    sections.extend([
        "",
        "Territory summaries:",
    ])
    for index, summary in enumerate(territory_summaries, start=1):
        if summary.strip():
            sections.append(f"{index}. {summary.strip()}")
    if not any(summary.strip() for summary in territory_summaries):
        sections.append("No territory summaries were produced; use the sources directly.")
    if worker_disagreements:
        sections.extend(["", "Worker disagreements that must be addressed:"])
        sections.extend(
            f"- {disagreement.topic}: {disagreement.agent_a_position} || {disagreement.agent_b_position}"
            for disagreement in worker_disagreements
        )
    sections.extend(["", "Numbered sources:"])
    sections.extend(numbered_sources(source_pairs, max_chars=1500))
    return "\n".join(sections).strip()


def worker_output_paths_from_runs(runs: list[TerritoryRun]) -> list[str]:
    return [run.output_path for run in runs if run.output_path]


def write_worker_output_note(
    output_path: str,
    territory: Territory,
    summary: str,
    sources: list[SourceRecord],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Territory: {territory.territory_id}",
        f"Role: {territory.assigned_agent_role.value}",
        f"Worker model: {territory.assigned_worker_model.value}",
        "",
        "Summary:",
        summary.strip() or "No summary produced.",
        "",
        "Source paths:",
    ]
    if sources:
        lines.extend(str(source.raw_text_path) for source in sources)
    else:
        lines.append("No sources recorded.")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def numbered_sources(
    source_pairs: list[tuple[SourceRecord, str]],
    *,
    max_chars: int,
) -> list[str]:
    sections: list[str] = []
    for index, (source, full_text) in enumerate(source_pairs, start=1):
        sections.append(f"[{index}] {source.title}")
        sections.append(full_text[:max_chars])
        sections.append("")
    return sections


def build_complete_session(
    question: str,
    *,
    answer: str,
    agent: str,
    queries_run: list[QueryCall],
    source_pairs: list[tuple[SourceRecord, str]],
    router,
) -> ResearchSession:
    sources = [source for source, _ in source_pairs]
    evidence_chunks = [
        evidence_chunk(source, full_text, answer)
        for source, full_text in source_pairs
    ]
    decision = session_answer_decision(sources, evidence_chunks, router=router)
    rerank_passed_count = sum(
        1 for chunk in evidence_chunks if chunk.rerank_score >= EVIDENCE_OVERLAP_PASS_THRESHOLD
    )
    rerank_failed_count = len(evidence_chunks) - rerank_passed_count
    return ResearchSession(
        protocol=Protocol.SEARCH,
        question=question,
        triggered_by=agent,
        final_status=decision.final_status,
        answer=answer if decision.answer_kind != AnswerKind.ABSTAIN else None,
        answer_kind=decision.answer_kind,
        confidence=decision.confidence,
        answer_confidence=decision.confidence,
        sources=sources,
        evidence_chunks=evidence_chunks,
        rerank_threshold_used=EVIDENCE_OVERLAP_PASS_THRESHOLD,
        rerank_passed_count=rerank_passed_count,
        rerank_failed_count=rerank_failed_count,
        queries_run=queries_run,
        open_questions=list(decision.open_questions),
    )


def build_and_save_research_session(
    question: str,
    *,
    protocol: Protocol,
    agent: str,
    territories: list[Territory],
    runs: list[TerritoryRun],
    llm_prefer: str | None,
    scout_attempt: GeminiInterlockAttempt,
    router,
    topic: str,
    iteration_count: int = 0,
) -> tuple[ResearchSession, Path | None, str, bool]:
    if scout_attempt.record is not None:
        final_attempt = run_gemini_final_synthesis(
            question,
            router=router,
            protocol=protocol,
            topic=topic,
            runs=runs,
        )
        if final_attempt.record is None:
            reason = gemini_unavailable_reason(final_attempt)
            session = build_abstain_session(
                question,
                protocol=protocol,
                agent=agent,
                queries_run=queries_from_runs(runs),
                sources=sources_from_runs(runs),
                territories=territories,
                open_question=f"final synthesis failed: {reason}",
                gemini_pro_runs=[scout_attempt.record],
            )
            return session, write_session_directly(session), "none", True
        session, backend = build_research_session_from_runs(
            question,
            protocol=protocol,
            agent=agent,
            territories=territories,
            runs=runs,
            llm_prefer=llm_prefer,
            router=router,
            iteration_count=iteration_count,
            final_answer_override=final_attempt.output_text,
            final_backend_override=final_attempt.record.model_id,
        )
        attach_gemini_record(session, scout_attempt.record)
        attach_gemini_record(session, final_attempt.record)
        saved_session, path = save_session_with_fallback(session)
        return saved_session, path, backend, False

    fallback_attempt = run_gemini_pro_synthesis_fallback(
        question,
        router=router,
        protocol=protocol,
        topic=topic,
        runs=runs,
    )
    if fallback_attempt.record is not None:
        final_answer = (
            fallback_attempt.output_text if source_pairs_from_runs(runs) else None
        )
        session, backend = build_research_session_from_runs(
            question,
            protocol=protocol,
            agent=agent,
            territories=territories,
            runs=runs,
            llm_prefer=llm_prefer,
            router=router,
            iteration_count=iteration_count,
            final_answer_override=final_answer,
            final_backend_override=fallback_attempt.record.model_id,
        )
        attach_gemini_record(session, fallback_attempt.record)
        saved_session, path = save_session_with_fallback(session)
        return saved_session, path, backend, False

    reason = gemini_unavailable_reason(scout_attempt, fallback_attempt)
    session = build_abstain_session(
        question,
        protocol=protocol,
        agent=agent,
        queries_run=queries_from_runs(runs),
        sources=sources_from_runs(runs),
        territories=territories,
        open_question=f"evidence gate bypassed: Gemini unavailable; {reason}",
    )
    print("evidence gate bypassed: Gemini unavailable", file=sys.stderr)
    return session, write_session_directly(session), "none", True


def attach_gemini_record(
    session: ResearchSession,
    record: GeminiProRunRecord,
) -> ResearchSession:
    session.gemini_pro_runs = [*session.gemini_pro_runs, record]
    return session


def gemini_unavailable_reason(*attempts: GeminiInterlockAttempt) -> str:
    reasons = [attempt.failure_reason for attempt in attempts if attempt.failure_reason]
    return "; ".join(reasons) or "unknown Gemini failure"


def source_pairs_from_runs(runs: list[TerritoryRun]) -> list[tuple[SourceRecord, str]]:
    return [
        (source, full_text)
        for run in runs
        for source, full_text in zip(run.sources, run.full_texts, strict=True)
    ]


def sources_from_runs(runs: list[TerritoryRun]) -> list[SourceRecord]:
    return [source for source, _full_text in source_pairs_from_runs(runs)]


def queries_from_runs(runs: list[TerritoryRun]) -> list[QueryCall]:
    return [query for run in runs for query in run.queries_run]


def build_research_session_from_runs(
    question: str,
    *,
    protocol: Protocol,
    agent: str,
    territories: list[Territory],
    runs: list[TerritoryRun],
    llm_prefer: str | None,
    router,
    iteration_count: int = 0,
    final_answer_override: str | None = None,
    final_backend_override: str | None = None,
) -> tuple[ResearchSession, str]:
    queries_run = queries_from_runs(runs)
    source_pairs = source_pairs_from_runs(runs)
    if not source_pairs:
        return (
            build_abstain_session(
                question,
                protocol=protocol,
                agent=agent,
                queries_run=queries_run,
                territories=territories,
                open_question="no usable sources retrieved",
            ),
            "none",
        )

    if final_answer_override is not None:
        answer = final_answer_override
        backend = final_backend_override or "agy_cli"
    else:
        try:
            answer, backend = llm_call.llm_complete(
                final_research_prompt(
                    question,
                    territory_summaries=[run.summary for run in runs],
                    worker_disagreements=detect_worker_disagreements(question, runs),
                    source_pairs=source_pairs,
                    worker_output_paths=worker_output_paths_from_runs(runs),
                ),
                prefer=llm_prefer,
            )
            if backend == "gemini":
                answer = apply_anti_hallucination_gate(
                    answer,
                    label="final_synthesis_gemini",
                )
        except Exception as exc:  # noqa: BLE001 - synthesis failure becomes abstain
            return (
                build_abstain_session(
                    question,
                    protocol=protocol,
                    agent=agent,
                    queries_run=queries_run,
                    sources=[source for source, _full_text in source_pairs],
                    territories=territories,
                    open_question=f"synthesis failed: {type(exc).__name__}: {exc}",
                ),
                "none",
            )

    sources = [source for source, _full_text in source_pairs]
    evidence_chunks = [
        evidence_chunk(source, full_text, answer)
        for source, full_text in source_pairs
    ]
    decision = session_answer_decision(sources, evidence_chunks, router=router)
    rerank_passed_count = sum(
        1 for chunk in evidence_chunks if chunk.rerank_score >= EVIDENCE_OVERLAP_PASS_THRESHOLD
    )
    rerank_failed_count = len(evidence_chunks) - rerank_passed_count
    session = ResearchSession(
        protocol=protocol,
        question=question,
        triggered_by=agent,
        final_status=decision.final_status,
        territories=territories,
        sources=sources,
        evidence_chunks=evidence_chunks,
        rerank_threshold_used=EVIDENCE_OVERLAP_PASS_THRESHOLD,
        rerank_passed_count=rerank_passed_count,
        rerank_failed_count=rerank_failed_count,
        queries_run=queries_run,
        answer=answer if decision.answer_kind != AnswerKind.ABSTAIN else None,
        answer_kind=decision.answer_kind,
        confidence=decision.confidence,
        answer_confidence=decision.confidence,
        open_questions=list(decision.open_questions),
        iteration_count=iteration_count,
        agent_disagreements=detect_worker_disagreements(question, runs),
    )
    return session, backend


def build_abstain_session(
    question: str,
    *,
    protocol: Protocol = Protocol.SEARCH,
    agent: str | None,
    queries_run: list[QueryCall],
    open_question: str,
    sources: list[SourceRecord] | None = None,
    territories: list[Territory] | None = None,
    gemini_pro_runs: list[GeminiProRunRecord] | None = None,
) -> ResearchSession:
    return ResearchSession(
        protocol=protocol,
        question=question,
        triggered_by=agent or "harness",
        final_status=FinalStatus.INSUFFICIENT_EVIDENCE,
        answer=None,
        answer_kind=AnswerKind.ABSTAIN,
        confidence=0.0,
        answer_confidence=0.0,
        sources=sources or [],
        territories=territories or [],
        evidence_chunks=[],
        queries_run=queries_run,
        open_questions=[open_question or "no usable sources retrieved"],
        gemini_pro_runs=gemini_pro_runs or [],
    )


def evidence_chunk(
    source: SourceRecord,
    full_text: str,
    answer: str,
    *,
    claim: str | None = None,
) -> EvidenceChunk:
    paragraph, matched_claim, overlap_score = best_supporting_paragraph(
        full_text,
        answer,
        claim=claim,
    )
    return EvidenceChunk(
        source_id=source.source_id,
        paragraph_text=paragraph,
        char_offset=0,
        char_length=max(1, len(paragraph)),
        rerank_score=overlap_score,
        supports_claim=matched_claim,
        crystal_check_passed=overlap_score >= EVIDENCE_OVERLAP_PASS_THRESHOLD,
        crystal_check_score=overlap_score,
    )


def detect_worker_disagreements(
    question: str,
    runs: list[TerritoryRun],
) -> list[Disagreement]:
    summaries = [
        (run, run.summary.strip())
        for run in runs
        if run.summary.strip()
    ]
    if len(summaries) < 2:
        return []

    negative_markers = (
        "no study",
        "does not exist",
        "doesn't exist",
        "not found",
        "nothing found",
        "measurement vacuum",
        "couldn't find",
    )
    affirmative_markers = (
        "arxiv",
        "doi",
        "paper",
        "study",
        "comparison",
        "vs",
        "0.",
    )

    negative_run = next(
        (
            run
            for run, summary in summaries
            if any(marker in summary.lower() for marker in negative_markers)
        ),
        None,
    )
    affirmative_run = next(
        (
            run
            for run, summary in summaries
            if any(marker in summary.lower() for marker in affirmative_markers)
        ),
        None,
    )
    if negative_run is None or affirmative_run is None or negative_run is affirmative_run:
        return []

    return [
        Disagreement(
            topic=question,
            agent_a_role=negative_run.territory.assigned_agent_role,
            agent_a_position=negative_run.summary.strip(),
            agent_a_evidence=[],
            agent_b_role=affirmative_run.territory.assigned_agent_role,
            agent_b_position=affirmative_run.summary.strip(),
            agent_b_evidence=[],
        )
    ]


def _looks_like_worker_failure(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    failure_markers = (
        "error:",
        "timeout waiting for response",
        "authentication failed",
        "invalid_grant",
        "no response received",
    )
    return any(marker in lowered for marker in failure_markers)


def build_query_calls(
    question: str,
    *,
    sx: dict[str, Any],
    px: dict[str, Any],
    chosen_provider: str,
    api_payloads: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[QueryCall]:
    now = datetime.now(timezone.utc)
    calls = [
        QueryCall(
            query_text=question,
            lane="searxng_general",
            worker_model=WorkerModel.HAIKU,
            started_at=now,
            duration_ms=0,
            result_count=len(_results(sx)),
            error=sx.get("error"),
        )
    ]
    for lane, payload in api_payloads or []:
        calls.append(
            QueryCall(
                query_text=question,
                lane=lane,
                worker_model=WorkerModel.HAIKU,
                started_at=now,
                duration_ms=0,
                result_count=len(_results(payload)),
                error=payload.get("error") if isinstance(payload.get("error"), str) else None,
            )
        )
    if not px.get("skipped"):
        calls.append(
            QueryCall(
                query_text=question,
                lane=chosen_provider,
                worker_model=WorkerModel.HAIKU,
                started_at=now,
                duration_ms=0,
                result_count=len(_results(px)),
                error=px.get("error"),
            )
        )
    return calls


def build_local_query_calls(
    question: str,
    *,
    local_payloads: list[tuple[str, dict[str, Any]]],
) -> list[QueryCall]:
    now = datetime.now(timezone.utc)
    return [
        QueryCall(
            query_text=question,
            lane=lane,
            worker_model=WorkerModel.HAIKU,
            started_at=now,
            duration_ms=0,
            result_count=len(_results(payload)),
            error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        )
        for lane, payload in local_payloads
    ]


def needs_deep_iteration(session: ResearchSession) -> bool:
    return (
        session.answer_kind == AnswerKind.ABSTAIN
        or session.final_status in {FinalStatus.WEAK_SOURCES, FinalStatus.INSUFFICIENT_EVIDENCE}
        or len(session.sources) < 2
    )


def weakest_territory_run(runs: list[TerritoryRun]) -> TerritoryRun:
    return min(runs, key=lambda run: len(run.sources))


def save_session_with_fallback(session: ResearchSession) -> tuple[ResearchSession, Path | None]:
    try:
        return session, persistence.save_session(session, root=persistence.DEFAULT_ROOT)
    except Exception as exc:  # noqa: BLE001 - rebuild as abstain and retry
        abstain = build_abstain_session(
            session.question,
            protocol=session.protocol,
            agent=session.triggered_by,
            queries_run=session.queries_run,
            sources=session.sources,
            territories=session.territories,
            open_question=f"save_session failed: {type(exc).__name__}: {exc}",
            gemini_pro_runs=session.gemini_pro_runs,
        )
        try:
            return abstain, persistence.save_session(abstain, root=persistence.DEFAULT_ROOT)
        except Exception:
            return abstain, write_session_directly(abstain)


def save_session_safely(session: ResearchSession) -> Path | None:
    _saved_session, path = save_session_with_fallback(session)
    return path


def write_session_directly(session: ResearchSession) -> Path | None:
    try:
        path = session.to_jsonl_path(persistence.DEFAULT_ROOT.expanduser().resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = session.model_dump_json(indent=2)
        if not payload.endswith("\n"):
            payload += "\n"
        path.write_text(payload, encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning("direct session write failed: %s", exc)
        return None


def telemetry_safely() -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            telemetry_observer.run()
    except Exception as exc:
        logger.warning("telemetry observer failed: %s", exc)
        return


def print_result(result: SearchRunResult) -> None:
    if result.session.answer:
        print(result.session.answer)
    else:
        detail = "; ".join(result.session.open_questions) or ABSTAIN_MESSAGE
        print(f"{ABSTAIN_MESSAGE} {detail}")
    print(f"session: {result.path or 'not saved'}")
    print(
        f"logged: {result.n_sources} sources, "
        f"{len(result.session.queries_run)} api calls, backend={result.backend}"
    )


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results") if isinstance(payload, dict) else None
    return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


def first_error_or(fallback: str, errors: list[str]) -> str:
    return errors[0] if errors else fallback


if __name__ == "__main__":
    raise SystemExit(main())
