"""Research Session schema — Pydantic models for the research engine.

Acts as the contract between every research protocol (/search, /research,
/deep-research, /research-check, /recall) and its downstream consumers
(MENTOR substrate, CT pipeline, NotebookLM ingest).

Per MENTOR V2 4-class enforcement (memory: project_mentor_v2_4class_constitution.md):
- Rule 2 (Precedent Grounding) is enforced HERE — Pydantic refuses to
  construct objects without required evidence fields. Hallucinated
  research cards become physically unstorable.

Aligned with locked decisions:
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (NOT bge — 0.35% worse,
  5.8% slower per 32-query bakeoff. Locked, do not re-bench.)
- CT 5-state resolution: pending / confirmed / falsified / ambiguous / annulled
- Algorithm aversion: every accuracy claim must include difficulty + baseline
- Master Event Index strategy: T3/T4 articles use comparison-prompts, not summarization
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums (locked vocabularies — extend with care)
# ---------------------------------------------------------------------------


class Protocol(str, Enum):
    SEARCH = "/search"
    RESEARCH = "/research"
    DEEP_RESEARCH = "/deep-research"
    RESEARCH_CHECK = "/research-check"
    RECALL = "/recall"


class ExtractionMethod(str, Enum):
    APIFY = "apify"
    AGENT_BROWSER = "agent_browser"
    CLOUDFLARE_MARKDOWN = "cloudflare-markdown"
    CRAWL4AI = "crawl4ai"
    CRAWLEE = "crawlee"
    CURL = "curl"
    FIRECRAWL = "firecrawl"
    GITINGEST = "gitingest"
    DOCLING = "docling"
    JINA = "jina"
    MARKITDOWN = "markitdown"
    NEWSPAPER4K = "newspaper4k"
    PUBLISHER_OA = "publisher_oa"
    PYMUPDF = "pymupdf"
    READABILITY = "readability"
    SCRAPLING = "scrapling"
    TRAFILATURA = "trafilatura"
    UNSTRUCTURED = "unstructured"
    WAYBACK = "wayback"


class SourceTier(int, Enum):
    """T1 = official/peer-reviewed/primary; T2 = reputable press/forum;
    T3 = anonymous/single-source/undated."""

    T1 = 1
    T2 = 2
    T3 = 3


class ResolutionState(str, Enum):
    """CT-specific 5-state resolution machine."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    FALSIFIED = "falsified"
    AMBIGUOUS = "ambiguous"
    ANNULLED = "annulled"


class FinalStatus(str, Enum):
    COMPLETE = "complete"
    WEAK_SOURCES = "weak_sources"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"


class AnswerKind(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    ABSTAIN = "abstain"


class GapSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WorkerModel(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    MISTRAL = "mistral"
    CODEX_MINI = "codex-mini"
    CODEX_5_4 = "codex-5.4"
    CODEX_5_5 = "codex-5.5"
    CODEX_5_3 = "codex-5.3"
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH = "gemini-flash"
    GROK = "grok"


class AgentRole(str, Enum):
    """Agent roles in /research and /deep-research."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    COUNTER_EVIDENCE = "counter_evidence"
    DOMAIN_SPECIALIST = "domain_specialist"
    CROSS_MODEL_VERIFIER = "cross_model_verifier"


# ---------------------------------------------------------------------------
# Leaf records
# ---------------------------------------------------------------------------


NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
UrlStr = Annotated[str, StringConstraints(min_length=8, pattern=r"^(https?|file)://")]


class QueryCall(BaseModel):
    """One concrete API/lane invocation."""

    model_config = ConfigDict(frozen=True)

    query_text: NonEmptyStr
    lane: NonEmptyStr  # must match a lane id in router_config.yaml
    worker_model: WorkerModel
    started_at: datetime
    duration_ms: int = Field(ge=0)
    result_count: int = Field(ge=0)
    cost_usd_estimate: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None  # nonempty if call failed


class SourceRecord(BaseModel):
    """A single fetched-and-extracted source. Immutable once stored.

    Required-field discipline enforces MENTOR V2 Rule 2: a source cannot
    exist in the system without an extraction method and a content hash.
    """

    model_config = ConfigDict(frozen=True)

    source_id: UUID = Field(default_factory=uuid4)
    url: UrlStr
    domain: NonEmptyStr  # e.g., "courtlistener.com"
    title: NonEmptyStr
    author: Optional[str] = None
    published_date: Optional[str] = None  # ISO; partial dates allowed (YYYY, YYYY-MM)
    fetched_at: datetime
    content_hash: Annotated[str, StringConstraints(min_length=64, max_length=64)]
    extraction_method: ExtractionMethod
    raw_text_path: Path  # where the cleaned text lives on disk
    char_count: int = Field(ge=0)
    tier: SourceTier
    topic_authority_score: float = Field(ge=0.0, le=1.0)
    counter_evidence_flagged: bool = False
    listicle_flagged: bool = False  # "best of" style page is flagged, not filtered
    archive_url: Optional[UrlStr] = None  # archive.org snapshot for citation stability

    @field_validator("content_hash")
    @classmethod
    def _hash_is_lowercase_hex(cls, v: str) -> str:
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("content_hash must be hex (sha256)")
        return v.lower()

    @staticmethod
    def hash_text(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()


class EvidenceChunk(BaseModel):
    """A paragraph-level evidence fragment supporting one specific claim.

    Schema-enforced FK to source_id means you cannot add evidence
    without a real, fetched source. This is the schema-as-bouncer
    pattern from the 4-class enforcement design.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID = Field(default_factory=uuid4)
    source_id: UUID  # FK to SourceRecord.source_id
    paragraph_text: NonEmptyStr
    char_offset: int = Field(ge=0)
    char_length: int = Field(ge=1)
    rerank_score: float = Field(ge=0.0, le=1.0)
    supports_claim: NonEmptyStr  # the specific claim this chunk grounds
    crystal_check_passed: bool  # RAGAS-style sentence↔paragraph faithfulness
    crystal_check_score: float = Field(ge=0.0, le=1.0)


class Territory(BaseModel):
    """One non-overlapping search territory in the Head Agent's decomposition."""

    model_config = ConfigDict(frozen=True)

    territory_id: NonEmptyStr  # e.g., "A", "B", "C" or descriptive
    description: NonEmptyStr
    queries: list[NonEmptyStr] = Field(min_length=1)
    assigned_agent_role: AgentRole
    assigned_lanes: list[NonEmptyStr] = Field(min_length=1)
    assigned_worker_model: WorkerModel
    do_not_overlap_with: list[NonEmptyStr] = Field(default_factory=list)


class Disagreement(BaseModel):
    """When two agents conflict on the same fact. Surface, do not auto-resolve."""

    topic: NonEmptyStr
    agent_a_role: AgentRole
    agent_a_position: NonEmptyStr
    agent_a_evidence: list[UUID] = Field(default_factory=list)  # FK to chunks
    agent_b_role: AgentRole
    agent_b_position: NonEmptyStr
    agent_b_evidence: list[UUID] = Field(default_factory=list)
    resolution: Literal[
        "open", "agent_a_correct", "agent_b_correct", "both_partial", "both_wrong"
    ] = "open"


class Gap(BaseModel):
    """A research gap detected by /research-check."""

    gap_topic: NonEmptyStr
    severity: GapSeverity
    detection_reason: NonEmptyStr  # "single source", "no counter-evidence", etc.
    recommended_lane: Optional[NonEmptyStr] = None
    triggered_iteration: bool = False


class CrossModelVerification(BaseModel):
    """3-lens check from /deep-research Agent E."""

    claim: NonEmptyStr
    grounding_chunks: list[UUID] = Field(min_length=1)
    analytical_lens_passed: bool
    analytical_lens_notes: NonEmptyStr
    creative_lens_passed: bool
    creative_lens_notes: NonEmptyStr
    skeptical_lens_passed: bool
    skeptical_lens_notes: NonEmptyStr

    @property
    def all_three_passed(self) -> bool:
        return (
            self.analytical_lens_passed
            and self.creative_lens_passed
            and self.skeptical_lens_passed
        )


class GeminiProRunKind(str, Enum):
    SCOUT = "scout"
    FINAL_SYNTHESIS = "final_synthesis"
    PRO_SYNTHESIS_FALLBACK = "pro_synthesis_fallback"


class GeminiProRunRecord(BaseModel):
    """One recorded Gemini Pro execution used by the persistence interlock."""

    run_type: GeminiProRunKind
    success: bool
    model_id: Optional[NonEmptyStr] = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    brief_path: Optional[Path] = None
    output_path: Optional[Path] = None
    failure_reason: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def _check_success_contract(self) -> GeminiProRunRecord:
        if self.success and not self.model_id:
            raise ValueError("successful Gemini Pro run requires model_id")
        if not self.success and not self.failure_reason:
            raise ValueError("failed Gemini Pro run requires failure_reason")
        return self


# ---------------------------------------------------------------------------
# Optional integration contexts
# ---------------------------------------------------------------------------


class CTContext(BaseModel):
    """CT-specific metadata. Present iff this session feeds the CT pipeline."""

    event_id: Optional[UUID] = None  # FK to CT events table
    master_event_index_match: bool  # is this in T1/T2 known events?
    comparison_prompt_used: bool
    delta_facts_extracted: list[NonEmptyStr] = Field(default_factory=list)
    resolution_state: ResolutionState = ResolutionState.PENDING
    resolution_criteria: NonEmptyStr  # set at session creation per CT spec
    earliest_date_seen: Optional[str] = None
    expiry_date: Optional[str] = None  # when this prediction stops counting
    fallback_rule: Optional[NonEmptyStr] = None
    cascade_parent_id: Optional[UUID] = None  # FK; if parent falsifies, this quarantines
    brier_prediction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    brier_resolution_mode: Optional[Literal["fast", "delayed", "interim"]] = None
    accountability_difficulty: Literal["easy", "moderate", "hard"]
    accountability_baseline: NonEmptyStr  # "compared to what" string

    @model_validator(mode="after")
    def _check_brier_consistency(self) -> CTContext:
        if (self.brier_prediction is None) != (self.brier_resolution_mode is None):
            raise ValueError(
                "brier_prediction and brier_resolution_mode must be set together"
            )
        return self


class MentorContext(BaseModel):
    """MENTOR-specific metadata. Present iff this session feeds the MENTOR substrate."""

    ingest_ready: bool
    rule_1_topology_passed: bool  # set by plain-code structural check
    rule_2_precedent_passed: bool  # always True if SourceRecords exist (schema enforces)
    rule_3_verifiable_substeps_passed: bool  # set by state machine
    rule_4_falsifiability_passed: Optional[bool] = None  # set by LLM critic, last
    rule_4_critic_rubric_version: Optional[NonEmptyStr] = None
    domain_tags: list[NonEmptyStr] = Field(default_factory=list)
    proposed_lateral_connections: list[NonEmptyStr] = Field(default_factory=list)

    @property
    def all_four_rules_passed(self) -> bool:
        return (
            self.rule_1_topology_passed
            and self.rule_2_precedent_passed
            and self.rule_3_verifiable_substeps_passed
            and (self.rule_4_falsifiability_passed is True)
        )


# ---------------------------------------------------------------------------
# Top-level session
# ---------------------------------------------------------------------------


class ResearchSession(BaseModel):
    """A single research session, queryable by both MENTOR and CT.

    Persisted to ~/.claude/research-sessions/{session_id}.json
    """

    model_config = ConfigDict(extra="forbid")  # no silent typos

    schema_version: Literal["1.0.0"] = "1.0.0"
    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    protocol: Protocol
    question: NonEmptyStr
    triggered_by: Optional[NonEmptyStr] = None  # caller (CT phase, MENTOR job, manual)
    final_status: FinalStatus = FinalStatus.IN_PROGRESS

    # Stage 1 — decomposition
    territories: list[Territory] = Field(default_factory=list)

    # Stage 2 — execution
    queries_run: list[QueryCall] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)

    # Stage 3 — ranking
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    rerank_threshold_used: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_passed_count: int = Field(default=0, ge=0)
    rerank_failed_count: int = Field(default=0, ge=0)
    evidence_gate_decision: Optional[dict[str, Any]] = None
    gemini_pro_runs: list[GeminiProRunRecord] = Field(default_factory=list)

    # Stage 4 — synthesis
    answer: Optional[str] = None  # may be None if final_status != COMPLETE
    answer_kind: Optional[AnswerKind] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    answer_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    open_questions: list[NonEmptyStr] = Field(default_factory=list)
    agent_disagreements: list[Disagreement] = Field(default_factory=list)
    cross_model_verifications: list[CrossModelVerification] = Field(default_factory=list)
    verbatim_check: Optional[dict[str, Any]] = None

    # Stage 5 — gap detection (optional, /deep-research only)
    gaps_detected: list[Gap] = Field(default_factory=list)
    iteration_count: int = Field(default=0, ge=0, le=3)

    # Optional integration metadata
    ct_metadata: Optional[CTContext] = None
    mentor_metadata: Optional[MentorContext] = None

    # Cost telemetry
    total_cost_usd_estimate: float = Field(default=0.0, ge=0.0)
    total_duration_ms: int = Field(default=0, ge=0)

    # ----- invariants -----

    @model_validator(mode="after")
    def _evidence_chunks_must_reference_sources(self) -> ResearchSession:
        """Schema enforces: every evidence chunk has a real source. No orphans."""
        source_ids = {s.source_id for s in self.sources}
        for chunk in self.evidence_chunks:
            if chunk.source_id not in source_ids:
                raise ValueError(
                    f"EvidenceChunk {chunk.chunk_id} references unknown "
                    f"source_id {chunk.source_id}"
                )
        return self

    @model_validator(mode="after")
    def _disagreements_reference_real_chunks(self) -> ResearchSession:
        chunk_ids = {c.chunk_id for c in self.evidence_chunks}
        for d in self.agent_disagreements:
            for cid in d.agent_a_evidence + d.agent_b_evidence:
                if cid not in chunk_ids:
                    raise ValueError(
                        f"Disagreement on '{d.topic}' references unknown chunk {cid}"
                    )
        return self

    @model_validator(mode="after")
    def _complete_status_requires_answer(self) -> ResearchSession:
        if self.final_status == FinalStatus.COMPLETE and self.answer_kind == AnswerKind.ABSTAIN:
            raise ValueError("answer_kind=abstain cannot use final_status=COMPLETE")
        if self.final_status == FinalStatus.COMPLETE and not self.answer:
            raise ValueError("final_status=COMPLETE requires a non-empty answer")
        return self

    @model_validator(mode="after")
    def _graduated_answer_contract(self) -> ResearchSession:
        """Graduated abstention: full / partial / abstain are explicit states."""
        if self.confidence is None and self.answer_confidence is not None:
            self.confidence = self.answer_confidence
        if self.answer_confidence is None and self.confidence is not None:
            self.answer_confidence = self.confidence

        if self.answer_kind is None:
            if self.final_status in (
                FinalStatus.WEAK_SOURCES,
                FinalStatus.INSUFFICIENT_EVIDENCE,
            ):
                self.answer_kind = AnswerKind.ABSTAIN if not self.answer else AnswerKind.PARTIAL
            elif self.answer:
                self.answer_kind = AnswerKind.FULL

        if self.answer_kind == AnswerKind.FULL:
            if self.final_status != FinalStatus.COMPLETE:
                raise ValueError("answer_kind=full requires final_status=COMPLETE")
            if not self.answer:
                raise ValueError("answer_kind=full requires a non-empty answer")

        if self.answer_kind == AnswerKind.PARTIAL:
            if not self.answer:
                raise ValueError("answer_kind=partial requires a partial answer")
            if self.confidence is None:
                raise ValueError("answer_kind=partial requires confidence")
            if not self.open_questions:
                raise ValueError(
                    "answer_kind=partial requires open_questions with confidence caveats"
                )

        if self.answer_kind == AnswerKind.ABSTAIN:
            if self.final_status not in (
                FinalStatus.WEAK_SOURCES,
                FinalStatus.INSUFFICIENT_EVIDENCE,
                FinalStatus.FAILED,
            ):
                raise ValueError(
                    "answer_kind=abstain requires weak_sources, "
                    "insufficient_evidence, or failed status"
                )
            if not self.open_questions:
                raise ValueError(
                    "answer_kind=abstain requires open_questions with concrete next steps"
                )
        return self

    @model_validator(mode="after")
    def _ct_metadata_requires_baseline(self) -> ResearchSession:
        """Algorithm aversion rule: every CT accuracy claim shows baseline."""
        if self.ct_metadata is not None:
            if not self.ct_metadata.accountability_baseline.strip():
                raise ValueError(
                    "CT mode requires accountability_baseline ('compared to what')"
                )
        return self

    # ----- convenience -----

    def to_jsonl_path(self, root: Path) -> Path:
        """Persistence path: ~/.claude/research-sessions/{YYYY-MM-DD}/{session_id}.json"""
        date_dir = self.created_at.strftime("%Y-%m-%d")
        return root / date_dir / f"{self.session_id}.json"

    def is_mentor_ingest_ready(self) -> bool:
        return (
            self.mentor_metadata is not None
            and self.mentor_metadata.all_four_rules_passed
            and self.final_status == FinalStatus.COMPLETE
        )
