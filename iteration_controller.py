"""Dormant auto-iteration subsystem for `research_engine`.

This module is currently not called from the live `/research` or
`/deep-research` pipeline. The intended entry point is
`decide_next_iteration(question, session_so_far)`.

Callers must pass the current question string plus a `ResearchSession`
containing the session's sources, evidence chunks, rerank counts,
iteration count, and cost telemetry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

import research_engine.evidence_gate as evidence_gate
from research_engine.dispatcher import trim_output
from research_engine.router import DEFAULT_CONFIG_PATH, Router, RoutingDecision, load_router
from research_engine.schema import (
    FinalStatus,
    Gap,
    GapSeverity,
    Protocol,
    ResearchSession,
    WorkerModel,
)


LOGGER = logging.getLogger("iteration_controller")


@dataclass(frozen=True)
class IterationDecision:
    should_iterate: bool
    reason: str
    gaps_to_address: list[Gap]
    recommended_lanes: list[str]
    recommended_worker_model: str
    iteration_index: int
    cost_remaining_usd: float


@dataclass(frozen=True)
class RecursionTierConfig:
    protocol: Protocol
    tier_name: str
    max_depth_below_scout: int
    subquestions_per_node: int
    leaf_search_cap: int
    budget_ceiling_usd: float


@dataclass
class RecursiveNode:
    question: str
    depth: int
    parent_summary: str
    budget_slice_usd: float
    recommended_lanes: list[str]
    recommended_worker_model: str
    children: list["RecursiveNode"] = field(default_factory=list)


@dataclass
class RecursivePlan:
    session: ResearchSession
    protocol: Protocol
    tier: RecursionTierConfig
    distinct_llm_count: int
    roots: list[RecursiveNode]
    asked_questions: list[str]
    dropped_repeated_questions: list[str]
    leaf_count: int
    budget_remaining_usd: float


@dataclass(frozen=True)
class JudgeResult:
    kept_chunk_ids: list[UUID]
    dropped_chunk_ids: list[UUID]
    drop_reasons: dict[UUID, str]
    worker_model: str


def detect_gaps(session: ResearchSession) -> list[Gap]:
    """Apply the /research-check rules to a session.

    Raises on unexpected internal errors so real bugs surface instead of
    silently returning an empty gap list. The freshness rule is degraded
    (skipped) if the router cannot be loaded — that is the only soft-fail.
    """
    try:
        routed = load_router(DEFAULT_CONFIG_PATH).route(session.question)
    except Exception:
        LOGGER.warning("Could not load router for detect_gaps; skipping freshness rule")
        routed = None
    return _detect_gaps(session, routed)


def decide_next_iteration(
    question: str,
    session_so_far: ResearchSession,
) -> IterationDecision:
    """Load the router and decide whether the session should iterate again.

    This is the intended entry point for a live caller that has a question
    string and the current in-memory session state.
    """

    session = session_so_far
    if session.question != question:
        session = session_so_far.model_copy(update={"question": question})
    decision = decide_iteration(session, load_router(DEFAULT_CONFIG_PATH))
    if session is not session_so_far:
        session_so_far.gaps_detected = session.gaps_detected
    return decision


def decide_iteration(session: ResearchSession, router: Router) -> IterationDecision:
    """Apply router iteration policy to a session and return the next-step decision.

    SIDE EFFECT: mutates `session.gaps_detected` in place — replaces it with the
    detected gaps where each gap's `triggered_iteration` flag is set True for the
    gaps selected for the next loop. Callers should `save_session(session)` after
    this call if they want the marked gaps persisted to disk.
    """
    try:
        policy = router.iteration_policy
        cost_remaining = max(
            0.0,
            float(policy.cost_cap_usd_per_session) - float(session.total_cost_usd_estimate),
        )
        next_iteration = int(session.iteration_count) + 1

        try:
            routed = router.route(session.question)
        except Exception:
            LOGGER.warning("Could not route question for iteration decision")
            routed = None
        detected = _detect_gaps(session, routed)
        worker_model = _worker_model(router)

        triggering = [
            gap
            for gap in detected
            if gap.severity.value in {value.lower() for value in policy.trigger_on_gap_severities}
        ]
        selected_topics = {gap.gap_topic for gap in triggering}
        session.gaps_detected = [
            gap.model_copy(update={"triggered_iteration": gap.gap_topic in selected_topics})
            for gap in detected
        ]

        if session.iteration_count >= policy.max_loops:
            return IterationDecision(
                False, "max loops reached", [], [], worker_model, next_iteration, cost_remaining
            )

        if session.total_cost_usd_estimate >= policy.cost_cap_usd_per_session:
            return IterationDecision(
                False, "cost cap hit", [], [], worker_model, next_iteration, cost_remaining
            )

        if not triggering:
            return IterationDecision(
                False,
                "no triggering gaps",
                [],
                [],
                worker_model,
                next_iteration,
                cost_remaining,
            )

        lanes = _recommended_lanes(
            triggering,
            routed.lanes if routed is not None else [],
            policy.per_iteration_lane_cap,
        )
        reason = (
            "triggering gaps detected: "
            + ", ".join(gap.detection_reason for gap in triggering[:3])
        )
        return IterationDecision(
            should_iterate=True,
            reason=reason,
            gaps_to_address=triggering,
            recommended_lanes=lanes,
            recommended_worker_model=worker_model,
            iteration_index=next_iteration,
            cost_remaining_usd=cost_remaining,
        )
    except Exception as exc:  # pragma: no cover - defensive guardrail
        LOGGER.exception("decide_iteration failed for session %s", session.session_id)
        return IterationDecision(
            should_iterate=False,
            reason=f"error: {exc}",
            gaps_to_address=[],
            recommended_lanes=[],
            recommended_worker_model=WorkerModel.HAIKU.value,
            iteration_index=int(getattr(session, "iteration_count", 0)) + 1,
            cost_remaining_usd=max(
                0.0,
                float(getattr(router.iteration_policy, "cost_cap_usd_per_session", 0.0))
                - float(getattr(session, "total_cost_usd_estimate", 0.0)),
            ),
        )


def count_distinct_answering_llms(session: ResearchSession) -> int:
    """Count distinct worker models that actually returned results this run."""

    answered = {
        call.worker_model.value
        for call in session.queries_run
        if call.result_count > 0 and not call.error
    }
    return len(answered)


def select_recursion_tier(
    session: ResearchSession,
    router: Router,
    *,
    distinct_llm_count: int | None = None,
) -> RecursionTierConfig:
    """Pick Ian's two-tier recursion limits from actual model diversity."""

    protocol = session.protocol
    if protocol not in (Protocol.RESEARCH, Protocol.DEEP_RESEARCH):
        return RecursionTierConfig(protocol, "flat", 0, 0, 0, 0.0)

    model_count = (
        count_distinct_answering_llms(session)
        if distinct_llm_count is None
        else distinct_llm_count
    )
    config = router.recursion_config()
    threshold = int(config.get("distinct_llm_threshold_for_deep_tier", 3))
    protocol_key = "deep_research" if protocol == Protocol.DEEP_RESEARCH else "research"
    tier_name = "deep" if model_count >= threshold else "one_llm"
    raw_tier = config.get(protocol_key, {}).get(tier_name, {})

    return RecursionTierConfig(
        protocol=protocol,
        tier_name=tier_name,
        max_depth_below_scout=int(raw_tier.get("max_depth_below_scout", 0)),
        subquestions_per_node=int(raw_tier.get("subquestions_per_node", 0)),
        leaf_search_cap=int(raw_tier.get("leaf_search_cap", 0)),
        budget_ceiling_usd=float(raw_tier.get("budget_ceiling_usd", 0.0)),
    )


def plan_recursive_research(
    session: ResearchSession,
    router: Router,
    scout_subareas: list[str],
    *,
    scout_summary: str = "",
    distinct_llm_count: int | None = None,
) -> RecursivePlan:
    """Create level-1 recursive work from the Gemini 3.7 Flash scout's depth targets."""

    tier = select_recursion_tier(
        session,
        router,
        distinct_llm_count=distinct_llm_count,
    )
    budget_remaining = max(
        0.0,
        tier.budget_ceiling_usd - float(session.total_cost_usd_estimate),
    )
    if session.total_cost_usd_estimate >= tier.budget_ceiling_usd:
        return _empty_recursive_plan(
            session=session,
            tier=tier,
            distinct_llm_count=distinct_llm_count,
            budget_remaining_usd=budget_remaining,
        )
    if tier.max_depth_below_scout <= 0 or tier.leaf_search_cap <= 0:
        return _empty_recursive_plan(
            session=session,
            tier=tier,
            distinct_llm_count=distinct_llm_count,
            budget_remaining_usd=budget_remaining,
        )
    routed = _safe_route(router, session.question)
    lanes = routed.lanes if routed is not None else ["searxng_general"]
    worker_model = _worker_model(router)

    roots: list[RecursiveNode] = []
    asked_questions: list[str] = []
    dropped: list[str] = []
    max_roots = max(0, tier.leaf_search_cap)
    candidates = _dedupe_questions(
        scout_subareas,
        asked_questions,
        dropped,
        router,
        limit=max_roots,
    )
    slice_count = max(1, len(candidates))
    budget_slice = budget_remaining / slice_count if budget_remaining else 0.0

    for question in candidates:
        asked_questions.append(question)
        roots.append(
            RecursiveNode(
                question=question,
                depth=1,
                parent_summary=scout_summary,
                budget_slice_usd=budget_slice,
                recommended_lanes=lanes[: router.iteration_policy.per_iteration_lane_cap],
                recommended_worker_model=worker_model,
            )
        )

    return RecursivePlan(
        session=session,
        protocol=session.protocol,
        tier=tier,
        distinct_llm_count=(
            count_distinct_answering_llms(session)
            if distinct_llm_count is None
            else distinct_llm_count
        ),
        roots=roots,
        asked_questions=asked_questions,
        dropped_repeated_questions=dropped,
        leaf_count=len(roots),
        budget_remaining_usd=budget_remaining,
    )


def expand_recursive_node(
    plan: RecursivePlan,
    parent: RecursiveNode,
    proposed_subquestions: list[str],
    *,
    parent_summary: str,
    router: Router,
) -> list[RecursiveNode]:
    """Add child work under one node, respecting depth, leaf, budget and loop guards."""

    if plan.session.total_cost_usd_estimate >= plan.tier.budget_ceiling_usd:
        return []
    if parent.depth >= plan.tier.max_depth_below_scout:
        return []
    if plan.leaf_count >= plan.tier.leaf_search_cap:
        return []

    remaining_leaf_slots = plan.tier.leaf_search_cap - plan.leaf_count
    limit = min(plan.tier.subquestions_per_node, remaining_leaf_slots)
    accepted = _dedupe_questions(
        proposed_subquestions,
        plan.asked_questions,
        plan.dropped_repeated_questions,
        router,
        limit=limit,
    )
    if not accepted:
        return []

    routed = _safe_route(router, parent.question)
    lanes = routed.lanes if routed is not None else parent.recommended_lanes
    slice_count = max(1, len(accepted))
    budget_slice = parent.budget_slice_usd / slice_count if parent.budget_slice_usd else 0.0
    children = [
        RecursiveNode(
            question=question,
            depth=parent.depth + 1,
            parent_summary=parent_summary,
            budget_slice_usd=budget_slice,
            recommended_lanes=lanes[: router.iteration_policy.per_iteration_lane_cap],
            recommended_worker_model=parent.recommended_worker_model,
        )
        for question in accepted
    ]
    parent.children.extend(children)
    plan.asked_questions.extend(accepted)
    plan.leaf_count += len(children)
    return children


def judge_findings(session: ResearchSession, router: Router) -> JudgeResult:
    """Compatibility wrapper around the enforced sufficiency gate.

    For committed answers, this runs the same gate enforced by save_session()
    and reports whether the current source pool remains usable.
    For draft sessions where the gate would not run yet, it returns a truthful
    keep-all result and leaves the evidence pool untouched.
    """

    config = router.judge_config()
    worker_model = str(config.get("worker_model", WorkerModel.HAIKU.value))
    original_chunks = list(session.evidence_chunks)
    original_chunk_ids = [chunk.chunk_id for chunk in original_chunks]

    if not evidence_gate.should_enforce_evidence_gate(session):
        return JudgeResult(
            kept_chunk_ids=original_chunk_ids,
            dropped_chunk_ids=[],
            drop_reasons={},
            worker_model=worker_model,
        )

    evidence_gate.enforce_evidence_gate(session)
    decision = session.evidence_gate_decision or {}
    if bool(decision.get("overridden", False)):
        reason = str(decision.get("gate_reason") or decision.get("reason") or "evidence_gate_rejected")
        return JudgeResult(
            kept_chunk_ids=[],
            dropped_chunk_ids=original_chunk_ids,
            drop_reasons={chunk_id: reason for chunk_id in original_chunk_ids},
            worker_model=worker_model,
        )

    return JudgeResult(
        kept_chunk_ids=[chunk.chunk_id for chunk in session.evidence_chunks],
        dropped_chunk_ids=[],
        drop_reasons={},
        worker_model=worker_model,
    )


def _detect_gaps(
    session: ResearchSession,
    routed: RoutingDecision | None,
) -> list[Gap]:
    return [
        *_single_source_claim_gaps(session),
        *_weak_source_pool_gaps(session),
        *_counter_evidence_gaps(session),
        *_stale_data_gaps(session, routed),
        *_subtopic_gaps(session),
        *_three_lens_gaps(session),
    ]


def _single_source_claim_gaps(session: ResearchSession) -> list[Gap]:
    claims: dict[str, set] = {}
    for chunk in session.evidence_chunks:
        claims.setdefault(chunk.supports_claim, set()).add(chunk.source_id)
    return [
        Gap(
            gap_topic=f"single-source claim about {trim_output(claim, limit=80)}",
            severity=GapSeverity.HIGH,
            detection_reason="single-source claim",
            recommended_lane="paid_proxy",
            triggered_iteration=False,
        )
        for claim, source_ids in claims.items()
        if len(source_ids) == 1
    ]


def _weak_source_pool_gaps(session: ResearchSession) -> list[Gap]:
    if session.rerank_failed_count <= session.rerank_passed_count:
        return []
    return [
        Gap(
            gap_topic="weak post-rerank source pool",
            severity=GapSeverity.HIGH,
            detection_reason="weak source pool",
            recommended_lane="paid_proxy",
            triggered_iteration=False,
        )
    ]


def _counter_evidence_gaps(session: ResearchSession) -> list[Gap]:
    if any(source.counter_evidence_flagged for source in session.sources) or session.agent_disagreements:
        return []
    return [
        Gap(
            gap_topic="counter-evidence not checked",
            severity=GapSeverity.MEDIUM,
            detection_reason="no counter-evidence",
            recommended_lane="reddit_failures",
            triggered_iteration=False,
        )
    ]


def _stale_data_gaps(
    session: ResearchSession,
    routed: RoutingDecision | None,
) -> list[Gap]:
    if routed is None or routed.freshness_max_age_hours is None:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=routed.freshness_max_age_hours)
    gaps: list[Gap] = []
    for source in session.sources:
        published = _parse_published_date(source.published_date)
        if published is not None and published < cutoff:
            gaps.append(
                Gap(
                    gap_topic=f"stale source {source.domain}: {trim_output(source.title, limit=80)}",
                    severity=GapSeverity.MEDIUM,
                    detection_reason="stale data",
                    recommended_lane="x_pulse",
                    triggered_iteration=False,
                )
            )
    return gaps


def _subtopic_gaps(session: ResearchSession) -> list[Gap]:
    if not session.open_questions:
        return []
    return [
        Gap(
            gap_topic=f"sub-topic missed: {trim_output(question, limit=80)}",
            severity=GapSeverity.LOW,
            detection_reason="sub-topic missed",
            recommended_lane="searxng_general",
            triggered_iteration=False,
        )
        for question in session.open_questions
    ]


def _three_lens_gaps(session: ResearchSession) -> list[Gap]:
    if session.protocol != Protocol.DEEP_RESEARCH or session.cross_model_verifications:
        return []
    return [
        Gap(
            gap_topic="missing 3-lens cross-model verification",
            severity=GapSeverity.MEDIUM,
            detection_reason="no 3-lens verification",
            recommended_lane="paid_proxy",
            triggered_iteration=False,
        )
    ]


def _worker_model(router: Router) -> str:
    candidate = router.config.get("worker_routing", {}).get(
        "search_and_read",
        WorkerModel.HAIKU.value,
    )
    for model in WorkerModel:
        if candidate == model.value or str(candidate).lower() == model.name.lower():
            return model.value
    return WorkerModel.HAIKU.value


def _safe_route(router: Router, question: str) -> RoutingDecision | None:
    try:
        return router.route(question)
    except Exception:
        LOGGER.warning("Could not route recursive research question")
        return None


def _empty_recursive_plan(
    session: ResearchSession,
    tier: RecursionTierConfig,
    *,
    distinct_llm_count: int | None,
    budget_remaining_usd: float,
) -> RecursivePlan:
    return RecursivePlan(
        session=session,
        protocol=session.protocol,
        tier=tier,
        distinct_llm_count=(
            count_distinct_answering_llms(session)
            if distinct_llm_count is None
            else distinct_llm_count
        ),
        roots=[],
        asked_questions=[],
        dropped_repeated_questions=[],
        leaf_count=0,
        budget_remaining_usd=budget_remaining_usd,
    )


def _dedupe_questions(
    questions: list[str],
    asked_questions: list[str],
    dropped: list[str],
    router: Router,
    *,
    limit: int,
) -> list[str]:
    accepted: list[str] = []
    threshold = float(
        router.recursion_config().get("similarity_drop_threshold", 0.86)
    )
    for question in questions:
        normalized = " ".join(str(question).split())
        if not normalized:
            continue
        prior_questions = [*asked_questions, *accepted]
        if any(_question_similarity(normalized, prior) >= threshold for prior in prior_questions):
            dropped.append(normalized)
            continue
        accepted.append(normalized)
        if len(accepted) >= limit:
            break
    return accepted


def _question_similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return min(len(left_tokens), len(right_tokens)) / max(
            len(left_tokens),
            len(right_tokens),
        )
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re_split_tokens(text)
        if len(token) > 2
        and token not in {"the", "and", "for", "with", "from", "that", "this"}
    }


def re_split_tokens(text: str) -> list[str]:
    return [
        token
        for token in "".join(
            char.lower() if char.isalnum() else " "
            for char in text
        ).split()
        if token
    ]


def _recommended_lanes(
    gaps: list[Gap],
    routed_lanes: list[str],
    lane_cap: int,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for gap in gaps:
        for lane in _lane_candidates(gap, routed_lanes):
            if lane not in seen:
                ordered.append(lane)
                seen.add(lane)
            if len(ordered) >= lane_cap:
                return ordered
    return ordered


def _lane_candidates(gap: Gap, routed_lanes: list[str]) -> list[str]:
    by_reason = {
        "single-source claim": ["paid_proxy", "searxng_general"],
        "weak source pool": ["paid_proxy", "searxng_general"],
        "no counter-evidence": ["reddit_failures", "searxng_forums"],
        "stale data": ["x_pulse", "bluesky_jetstream", "searxng_general"],
        "sub-topic missed": ["searxng_general", "paid_proxy"],
        "no 3-lens verification": ["paid_proxy", "searxng_general"],
    }
    candidates = by_reason.get(gap.detection_reason, [])
    fallback = [gap.recommended_lane] if gap.recommended_lane else []
    return [lane for lane in [*candidates, *fallback, *routed_lanes] if lane]


def _parse_published_date(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    for candidate in (
        normalized,
        f"{normalized}-01" if len(normalized) == 7 else "",
        f"{normalized}-01-01" if len(normalized) == 4 else "",
    ):
        if not candidate:
            continue
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    LOGGER.warning("Could not parse published_date=%r", value)
    return None


if __name__ == "__main__":
    ok = True
    failures: list[str] = []
    router = load_router(DEFAULT_CONFIG_PATH)

    base = ResearchSession(
        protocol=Protocol.RESEARCH,
        question="General market question with no explicit freshness requirement",
        final_status=FinalStatus.COMPLETE,
        answer="A complete answer exists.",
        iteration_count=0,
    )
    gaps = detect_gaps(base)
    if not any(gap.detection_reason == "no counter-evidence" for gap in gaps):
        ok = False
        failures.append("detect_gaps did not flag no counter-evidence")

    first_decision = decide_iteration(base, router)
    medium_triggers = "medium" in router.iteration_policy.trigger_on_gap_severities
    if first_decision.should_iterate != medium_triggers:
        ok = False
        failures.append("decide_iteration MEDIUM-trigger behavior mismatch")

    capped = base.model_copy(
        update={"iteration_count": router.iteration_policy.max_loops}
    )
    capped_decision = decide_iteration(capped, router)
    if capped_decision.should_iterate:
        ok = False
        failures.append("max loops guard failed")

    if ok:
        print("PASS: iteration_controller self-test")
    else:
        print("FAIL: " + "; ".join(failures))
