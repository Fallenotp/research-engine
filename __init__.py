"""Research engine — shared orchestrator used by /search, /research, /deep-research."""
from research_engine.schema import (
    ResearchSession, SourceRecord, EvidenceChunk, Territory,
    Disagreement, Gap, CrossModelVerification, QueryCall,
    CTContext, MentorContext, Protocol, ExtractionMethod,
    SourceTier, FinalStatus, AgentRole, WorkerModel, ResolutionState,
)
from research_engine.router import Router, RoutingDecision, load_router
from research_engine.dispatcher import WorkerSpec, dispatch, routing_table
from research_engine.extractor import compact_search_results, extract_clean_text
from research_engine.iteration_controller import detect_gaps, decide_iteration, IterationDecision
from research_engine.persistence import save_session, load_session, list_sessions, delete_session
from research_engine.verbatim_check import (
    HONEST_SCOPE_NOTE,
    VerbatimResult,
    check_verbatim,
    result_to_markdown,
    source_texts_from_paths,
)

__all__ = [
    "ResearchSession", "SourceRecord", "EvidenceChunk", "Territory",
    "Disagreement", "Gap", "CrossModelVerification", "QueryCall",
    "CTContext", "MentorContext", "Protocol", "ExtractionMethod",
    "SourceTier", "FinalStatus", "AgentRole", "WorkerModel", "ResolutionState",
    "Router", "RoutingDecision", "load_router",
    "WorkerSpec", "dispatch", "routing_table",
    "compact_search_results", "extract_clean_text",
    "detect_gaps", "decide_iteration", "IterationDecision",
    "save_session", "load_session", "list_sessions", "delete_session",
    "HONEST_SCOPE_NOTE", "VerbatimResult", "check_verbatim",
    "result_to_markdown", "source_texts_from_paths",
]
