from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_engine import research_cli
from research_engine import paths
import research_engine.dispatcher as dispatcher
from research_engine.persistence import CANONICAL_GEMINI_PRO_MODEL_ID
from research_engine.schema import (
    AnswerKind,
    FinalStatus,
    GeminiProRunKind,
    GeminiProRunRecord,
    Protocol,
    WorkerModel,
)


def _install_common_fakes(monkeypatch, tmp_path: Path, *, extract_result, grok_calls=None):
    saved_sessions = []
    if grok_calls is None:
        grok_calls = []

    class FakeRouter:
        def route(self, question: str):
            return SimpleNamespace(topic="test", lanes=["tavily_direct"])

        def fleet_worker_models(self, fleet_name: str):
            if fleet_name == research_cli.RESEARCH_FLEET_NAME:
                return [model.value for model in (WorkerModel.CODEX_5_4, WorkerModel.MISTRAL, WorkerModel.GROK)]
            if fleet_name == research_cli.DEEP_RESEARCH_FLEET_NAME:
                return (
                    [WorkerModel.HAIKU.value] * 5
                    + [WorkerModel.CODEX_5_4.value] * 5
                    + [WorkerModel.HAIKU.value] * 5
                    + [WorkerModel.GROK.value]
                )
            raise AssertionError(f"unexpected fleet {fleet_name}")

    def fake_save_session(session, root):
        saved_sessions.append(session)
        path = Path(root) / session.created_at.strftime("%Y-%m-%d") / f"{session.session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    monkeypatch.setattr(research_cli, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *args, **kwargs: {"results": [{"url": "https://ex.com/a"}]},
    )
    monkeypatch.setattr(
        research_cli.logged_search,
        "proxy",
        lambda *args, **kwargs: {"results": [{"url": "https://ex.com/a"}]},
    )
    monkeypatch.setattr(research_cli, "extract_clean_text", lambda *args, **kwargs: extract_result)
    monkeypatch.setattr(
        research_cli,
        "execute_grok_worker_spec",
        lambda spec: grok_calls.append(spec) or "X post by @example on 2026-05-27. https://x.com/example/status/1",
    )
    monkeypatch.setattr(research_cli.persistence, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(research_cli.persistence, "save_session", fake_save_session)
    monkeypatch.setattr(research_cli.telemetry_observer, "run", lambda: {"added": 1})
    return saved_sessions


def _successful_gemini_attempt(
    run_type: GeminiProRunKind,
    output_text: str = "Gemini scout context",
    *,
    model_id: str = CANONICAL_GEMINI_PRO_MODEL_ID,
):
    return research_cli.GeminiInterlockAttempt(
        run_type=run_type,
        record=GeminiProRunRecord(
            run_type=run_type,
            success=True,
            model_id=model_id,
        ),
        output_text=output_text,
    )


def _failed_gemini_attempt(run_type: GeminiProRunKind, reason: str):
    return research_cli.GeminiInterlockAttempt(
        run_type=run_type,
        failure_reason=reason,
    )


def _install_research_fakes(
    monkeypatch,
    tmp_path: Path,
    *,
    no_sources=False,
    unique_urls=True,
    fake_save=True,
    grok_calls=None,
):
    saved_sessions = []
    search_calls = []
    if grok_calls is None:
        grok_calls = []

    class FakeRouter:
        def route(self, question: str):
            return SimpleNamespace(topic="test", lanes=["tavily_direct"])

        def fleet_worker_models(self, fleet_name: str):
            if fleet_name == research_cli.RESEARCH_FLEET_NAME:
                return [model.value for model in (WorkerModel.CODEX_5_4, WorkerModel.MISTRAL, WorkerModel.GROK)]
            if fleet_name == research_cli.DEEP_RESEARCH_FLEET_NAME:
                return (
                    [WorkerModel.HAIKU.value] * 5
                    + [WorkerModel.CODEX_5_4.value] * 5
                    + [WorkerModel.HAIKU.value] * 5
                    + [WorkerModel.GROK.value]
                )
            raise AssertionError(f"unexpected fleet {fleet_name}")

    def fake_save_session(session, root):
        saved_sessions.append(session)
        path = Path(root) / session.created_at.strftime("%Y-%m-%d") / f"{session.session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def fake_search(query, **kwargs):
        lane = kwargs.get("provider") or "searxng_general"
        search_calls.append({"lane": lane, "query": query})
        suffix = len(search_calls) if unique_urls else "only"
        return {"results": [{"url": f"https://ex.com/{suffix}"}]}

    def fake_extract(url, *args, **kwargs):
        if no_sources:
            return None
        safe = url.rsplit("/", 1)[-1].replace("?", "-")
        raw_text_path = tmp_path / f"source-{safe}.txt"
        full_text = f"Grounded source text for {url}. It directly supports the answer."
        raw_text_path.write_text(full_text, encoding="utf-8")
        return {
            "url": url,
            "domain": "ex.com",
            "title": f"Source {safe}",
            "extraction_method": "curl",
            "raw_text_path": str(raw_text_path),
            "char_count": len(full_text),
            "char_text_preview": full_text[:200],
        }

    def fake_execute_grok_worker_spec(spec):
        grok_calls.append(spec)
        output_path = Path(spec.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("Fake Grok summary [grok]\n", encoding="utf-8")
        return "Fake Grok summary [grok]"

    def fake_execute_cli_worker_spec(spec, *args, **kwargs):
        output_path = Path(spec.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = f"Fake {spec.worker_model} summary [1]"
        output_path.write_text(output + "\n", encoding="utf-8")
        return output

    monkeypatch.setattr(research_cli, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(research_cli.logged_search, "searxng", fake_search)
    monkeypatch.setattr(research_cli.logged_search, "proxy", fake_search)
    monkeypatch.setattr(research_cli, "extract_clean_text", fake_extract)
    monkeypatch.setattr(research_cli, "execute_grok_worker_spec", fake_execute_grok_worker_spec)
    monkeypatch.setattr(research_cli, "execute_codex_worker_spec", fake_execute_cli_worker_spec)
    monkeypatch.setattr(research_cli, "execute_gemini_worker_spec", fake_execute_cli_worker_spec)
    monkeypatch.setattr(research_cli, "execute_mistral_worker_spec", fake_execute_cli_worker_spec)
    monkeypatch.setattr(research_cli.persistence, "DEFAULT_ROOT", tmp_path)
    if fake_save:
        monkeypatch.setattr(research_cli.persistence, "save_session", fake_save_session)
    else:
        monkeypatch.setattr(research_cli.persistence, "enforce_evidence_gate", lambda session: session)
        monkeypatch.setattr(
            research_cli,
            "write_session_directly",
            lambda session: (_ for _ in ()).throw(
                AssertionError("direct write fallback should not be used")
            ),
        )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_scout",
        lambda *args, **kwargs: _successful_gemini_attempt(GeminiProRunKind.SCOUT),
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_pro_synthesis_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Pro synthesis fallback should not run after successful scout")
        ),
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_final_synthesis",
        lambda *args, **kwargs: _successful_gemini_attempt(
            GeminiProRunKind.FINAL_SYNTHESIS,
            "Final grounded answer [1]\nCaveats/disagreements: none.",
            model_id=WorkerModel.OPUS.value,
        ),
    )
    monkeypatch.setattr(research_cli.telemetry_observer, "run", lambda: {"added": 1})
    return saved_sessions, search_calls


def test_run_search_builds_session_with_source_answer_and_path(monkeypatch, tmp_path):
    grok_calls = []
    raw_text_path = tmp_path / "source.txt"
    full_text = "Whisper.cpp is a fast local speech recognizer."
    raw_text_path.write_text(full_text, encoding="utf-8")
    extract_result = {
        "url": "https://ex.com/a",
        "domain": "ex.com",
        "title": "Example",
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": "",
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": len(full_text),
        "char_text_preview": full_text[:200],
    }
    saved_sessions = _install_common_fakes(
        monkeypatch,
        tmp_path,
        extract_result=extract_result,
        grok_calls=grok_calls,
    )
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *args, **kwargs: ("whisper.cpp is an option [1]", "codex"),
    )

    result = research_cli.run_search(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.path is not None
    assert result.path.exists()
    assert result.session.sources
    assert result.session.answer
    # Graded answer_kind is incidental here; thin synthetic sources may be PARTIAL.
    assert result.session.answer_kind in {AnswerKind.FULL, AnswerKind.PARTIAL}
    assert result.backend == "codex"
    assert len(saved_sessions) == 1
    assert len(grok_calls) == 1
    assert grok_calls[0].model_id == dispatcher.GROK_RESEARCH_MODEL
    assert any(call.lane == "grok_x_search" for call in result.session.queries_run)
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["sources"]
    assert payload["answer"]


def test_run_search_skips_paid_proxy_when_searxng_has_enough_results(
    monkeypatch,
    tmp_path,
):
    raw_text_path = tmp_path / "source.txt"
    full_text = "SearXNG returned enough free source candidates."
    raw_text_path.write_text(full_text, encoding="utf-8")
    extract_result = {
        "url": "https://ex.com/free-0",
        "domain": "ex.com",
        "title": "Free result",
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": "",
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": len(full_text),
        "char_text_preview": full_text[:200],
    }
    _install_common_fakes(monkeypatch, tmp_path, extract_result=extract_result)

    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *args, **kwargs: {
            "results": [{"url": f"https://ex.com/free-{index}"} for index in range(5)]
        },
    )
    monkeypatch.setattr(
        research_cli.logged_search,
        "proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paid proxy should be skipped when free results are sufficient")
        ),
    )
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *args, **kwargs: ("free-first answer [1]", "codex"),
    )

    result = research_cli.run_search("free-first query", topic="free-first", agent="pytest")

    assert result.session.answer == "free-first answer [1]"
    assert [call.lane for call in result.session.queries_run] == [
        "searxng_general",
        "grok_x_search",
    ]


def test_run_search_uses_paid_proxy_when_searxng_is_thin(monkeypatch, tmp_path):
    proxy_calls = []
    raw_text_path = tmp_path / "source.txt"
    full_text = "Paid fallback source text."
    raw_text_path.write_text(full_text, encoding="utf-8")
    extract_result = {
        "url": "https://ex.com/paid",
        "domain": "ex.com",
        "title": "Paid fallback",
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": "",
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": len(full_text),
        "char_text_preview": full_text[:200],
    }
    _install_common_fakes(monkeypatch, tmp_path, extract_result=extract_result)

    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *args, **kwargs: {"results": [{"url": "https://ex.com/free-only"}]},
    )

    def fake_proxy(*args, **kwargs):
        proxy_calls.append(kwargs)
        return {"results": [{"url": "https://ex.com/paid"}]}

    monkeypatch.setattr(research_cli.logged_search, "proxy", fake_proxy)
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *args, **kwargs: ("paid fallback answer [1]", "codex"),
    )

    result = research_cli.run_search("thin query", topic="thin", agent="pytest")

    assert proxy_calls
    assert result.session.answer == "paid fallback answer [1]"
    assert [call.lane for call in result.session.queries_run] == [
        "searxng_general",
        "tavily",
        "grok_x_search",
    ]


def test_run_search_uses_free_specialty_lane_before_paid_proxy(
    monkeypatch,
    tmp_path,
):
    raw_text_path = tmp_path / "source.txt"
    full_text = "Arxiv returned enough free source candidates."
    raw_text_path.write_text(full_text, encoding="utf-8")
    extract_result = {
        "url": "https://arxiv.org/abs/1234.00001",
        "domain": "arxiv.org",
        "title": "Free arxiv result",
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": "",
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": len(full_text),
        "char_text_preview": full_text[:200],
    }
    _install_common_fakes(monkeypatch, tmp_path, extract_result=extract_result)

    class FakeRouter:
        def route(self, question: str):
            return SimpleNamespace(topic="academic", lanes=["arxiv", "paid_proxy"])

        def lane_endpoint(self, lane_name: str):
            if lane_name == "arxiv":
                return {
                    "type": "api",
                    "endpoint": "https://export.arxiv.org/api/query?search_query=all:{query}",
                    "auth": "none",
                    "cost_per_call_usd": 0.0,
                }
            if lane_name == "paid_proxy":
                return {
                    "type": "api",
                    "endpoint": "http://localhost:18791/search",
                    "cost_per_call_usd": 0.005,
                }
            raise KeyError(lane_name)

    api_lane_calls = []
    monkeypatch.setattr(research_cli, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *args, **kwargs: {"results": [{"url": "https://ex.com/thin"}]},
    )

    def fake_api_lane(lane, request, **kwargs):
        api_lane_calls.append((lane, request, kwargs))
        return {
            "results": [
                {"url": f"https://arxiv.org/abs/1234.0000{index}"}
                for index in range(4)
            ]
        }

    monkeypatch.setattr(research_cli.logged_search, "api_lane", fake_api_lane)
    monkeypatch.setattr(
        research_cli.logged_search,
        "proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paid proxy should wait for free specialty lanes")
        ),
    )
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *args, **kwargs: ("free specialty answer [1]", "codex"),
    )

    result = research_cli.run_search(
        "transformer attention papers",
        topic="academic-free",
        agent="pytest",
    )

    assert [call[0] for call in api_lane_calls] == ["arxiv"]
    assert result.session.answer == "free specialty answer [1]"
    assert [call.lane for call in result.session.queries_run] == [
        "searxng_general",
        "arxiv",
        "grok_x_search",
    ]


def test_search_synthesis_prompt_includes_query_phrasing_playbook() -> None:
    prompt = research_cli.synthesis_prompt("How does X connect to Y?", [])

    assert "Query phrasing rules:" in prompt
    assert "X connects to Y" in prompt
    assert "counter-case" in prompt


def test_decomposition_prompt_includes_query_phrasing_playbook(monkeypatch) -> None:
    prompts = []

    def fake_complete(prompt, **kwargs):
        prompts.append(prompt)
        return ('["one", "two", "three"]', "mock")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_complete)

    assert research_cli.decompose_question("How does X connect to Y?", 3) == [
        "one",
        "two",
        "three",
    ]
    assert prompts
    assert "Query phrasing rules:" in prompts[0]
    assert "Cap reformulation at 2 retries" in prompts[0]


def test_worker_territory_brief_includes_query_phrasing_playbook() -> None:
    territory = research_cli.Territory(
        territory_id="semantic",
        description="Semantic territory",
        queries=["How does X connect to Y?"],
        assigned_agent_role=research_cli.AgentRole.SEMANTIC,
        assigned_lanes=["semantic_scholar"],
        assigned_worker_model=WorkerModel.HAIKU,
    )

    brief = research_cli.worker_territory_brief(
        "How does X connect to Y?",
        territory,
        protocol=Protocol.RESEARCH,
        source_pairs=[],
        worker_model=WorkerModel.HAIKU,
    )

    assert "Query phrasing rules:" in brief
    assert "semantic/full-question" in brief


def test_run_search_abstains_without_usable_sources(monkeypatch, tmp_path):
    saved_sessions = _install_common_fakes(monkeypatch, tmp_path, extract_result=None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called without usable sources")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fail_if_called)

    result = research_cli.run_search(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.path is not None
    assert result.path.exists()
    assert result.session.final_status == FinalStatus.INSUFFICIENT_EVIDENCE
    assert result.session.answer_kind == AnswerKind.ABSTAIN
    assert result.session.answer is None
    assert result.session.open_questions
    assert len(saved_sessions) == 1


def test_run_research_builds_three_territory_session(monkeypatch, tmp_path):
    grok_calls = []
    saved_sessions, _search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        grok_calls=grok_calls,
    )
    llm_calls = []

    def fake_llm(prompt, **kwargs):
        llm_calls.append(prompt)
        if "exactly 3 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [
                        "current options",
                        "selection criteria",
                        "known problems",
                    ]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return (f"Territory summary {len(llm_calls)} [1]", "mock")
        return ("Final grounded answer [1]\nCaveats/disagreements: none.", "mock")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.session.protocol == Protocol.RESEARCH
    assert len(result.session.territories) == 3
    assert result.session.sources
    assert result.session.answer
    # Purpose is three-territory session build; kind is graded from fixtures, not forced FULL.
    assert result.session.answer_kind in {AnswerKind.FULL, AnswerKind.PARTIAL}
    assert len(saved_sessions) == 1
    assert len(llm_calls) == 1
    assert len(grok_calls) == 1
    assert grok_calls[0].model_id == dispatcher.GROK_RESEARCH_MODEL


def test_run_research_attaches_gemini_record_before_gated_save(monkeypatch, tmp_path):
    _saved_sessions, _search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        fake_save=False,
    )
    llm_calls = []

    def fake_llm(prompt, **kwargs):
        llm_calls.append(prompt)
        if "exactly 3 non-overlapping sub-questions" in prompt:
            assert "Gemini scout context" in prompt
            return (
                json.dumps(
                    [
                        "current options",
                        "selection criteria",
                        "known problems",
                    ]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return (f"Territory summary {len(llm_calls)} [1]", "mock")
        return ("Final grounded answer [1]\nCaveats/disagreements: none.", "mock")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.path is not None
    assert result.path.exists()
    # Purpose is scout attachment before gated save, not COMPLETE from fail-open FULL.
    assert result.session.gemini_pro_runs
    assert any(run.run_type == GeminiProRunKind.SCOUT for run in result.session.gemini_pro_runs)
    run = result.session.gemini_pro_runs[0]
    assert run.run_type == GeminiProRunKind.SCOUT
    assert run.success is True
    assert run.model_id == CANONICAL_GEMINI_PRO_MODEL_ID
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["gemini_pro_runs"][0]["run_type"] == GeminiProRunKind.SCOUT.value
    assert payload["gemini_pro_runs"][0]["model_id"] == CANONICAL_GEMINI_PRO_MODEL_ID


def test_run_deep_research_thin_evidence_iteration_does_not_crash(monkeypatch, tmp_path):
    grok_calls = []
    saved_sessions, search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        unique_urls=False,
        grok_calls=grok_calls,
    )
    llm_calls = []

    def fake_llm(prompt, **kwargs):
        llm_calls.append(prompt)
        if "exactly 16 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [f"deep research angle {index}" for index in range(1, 17)]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return (f"Deep territory summary {len(llm_calls)} [1]", "mock")
        return ("Deep grounded answer [1]\nCaveats/disagreements: thin evidence.", "mock")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_deep_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    worker_queries = [
        call
        for call in search_calls
        if call["lane"] == "searxng_general" and "failure OR problem" not in call["query"]
    ]
    assert result.session.protocol == Protocol.DEEP_RESEARCH
    assert len(result.session.territories) == 16
    assert len(worker_queries) == 17
    assert len(saved_sessions) == 2
    assert result.path is not None
    assert result.path.exists()
    # unique_urls=False is deliberately thin; measured grading may ABSTAIN (answer=None).
    assert result.session.answer_kind in {AnswerKind.PARTIAL, AnswerKind.ABSTAIN}
    assert result.session.iteration_count >= 1
    assert {call.worker_model for call in result.session.queries_run} >= {
        WorkerModel.HAIKU,
        WorkerModel.CODEX_5_4,
        WorkerModel.GROK,
    }
    assert len(llm_calls) >= 1
    assert len(grok_calls) == 1
    assert grok_calls[0].model_id == dispatcher.GROK_RESEARCH_MODEL


def test_run_deep_research_records_worker_disagreement(monkeypatch, tmp_path):
    _saved_sessions, _search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        fake_save=False,
    )
    summary_calls = {"n": 0}

    def fake_llm(prompt, **kwargs):
        if "exactly 16 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [f"deep research angle {index}" for index in range(1, 17)]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            summary_calls["n"] += 1
            if summary_calls["n"] == 1:
                return (
                    "GA4-practitioner says published accuracy data does not exist.",
                    "mock",
                )
            return ("Neutral territory summary [1]", "mock")
        return ("Final grounded answer [1]\nCaveats/disagreements: none.", "mock")

    def fake_execute_grok_worker_spec(spec):
        output_path = Path(spec.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = "Another worker found arXiv 2411.10109 with a 0.74 vs 0.83 comparison."
        output_path.write_text(output + "\n", encoding="utf-8")
        return output

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)
    monkeypatch.setattr(research_cli, "execute_grok_worker_spec", fake_execute_grok_worker_spec)

    result = research_cli.run_deep_research(
        "Which demographic personas improve GA4 accuracy?",
        topic="ga4-personas",
        agent="pytest",
    )

    assert result.session.agent_disagreements
    disagreement = result.session.agent_disagreements[0]
    assert "does not exist" in disagreement.agent_a_position.lower()
    assert "arxiv" in disagreement.agent_b_position.lower()


def test_research_worker_models_follow_router_fleets_config():
    class FakeRouter:
        def fleet_worker_models(self, fleet_name: str):
            if fleet_name == research_cli.RESEARCH_FLEET_NAME:
                return ["sonnet", "opus", "haiku"]
            if fleet_name == research_cli.DEEP_RESEARCH_FLEET_NAME:
                return (
                    ["sonnet"] * 5
                    + ["opus"] * 5
                    + ["haiku"] * 5
                    + ["grok"]
                )
            raise AssertionError(f"unexpected fleet {fleet_name}")

    research_specs = research_cli.research_territory_specs(router=FakeRouter())
    deep_specs = research_cli.deep_research_territory_specs(router=FakeRouter())

    assert [spec[3] for spec in research_specs] == [
        WorkerModel.SONNET,
        WorkerModel.OPUS,
        WorkerModel.HAIKU,
    ]
    assert [spec[3] for spec in deep_specs[:5]] == [WorkerModel.SONNET] * 5
    assert [spec[3] for spec in deep_specs[5:10]] == [WorkerModel.OPUS] * 5
    assert [spec[3] for spec in deep_specs[10:15]] == [WorkerModel.HAIKU] * 5
    assert deep_specs[15][3] is WorkerModel.GROK

    assert [spec[:3] for spec in research_specs] == list(research_cli.RESEARCH_LANE_PLAN)
    assert [spec[:3] for spec in deep_specs] == list(research_cli.DEEP_RESEARCH_LANE_PLAN)


def test_broken_fleets_config_is_loud_not_silent(monkeypatch, caplog, tmp_path: Path):
    warning_fragment = "fleet 'research' router fleets config unusable"

    class BrokenRouter:
        def fleet_worker_models(self, fleet_name: str):
            raise ValueError(f"bad config for {fleet_name}")

        def route(self, question: str):
            return SimpleNamespace(topic="test", lanes=["tavily_direct"])

    saved_sessions = _install_research_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(research_cli, "load_router", lambda: BrokenRouter())
    monkeypatch.setattr(
        research_cli,
        "run_gemini_scout",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.SCOUT,
            "disabled for test",
        ),
    )
    monkeypatch.setattr(
        research_cli,
        "decompose_question",
        lambda question, territory_count, **kwargs: [
            f"sub-question {index}" for index in range(territory_count)
        ],
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_pro_synthesis_fallback",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
            "disabled for test",
        ),
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_final_synthesis",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.FINAL_SYNTHESIS,
            "disabled for test",
        ),
    )

    with caplog.at_level("ERROR"):
        result = research_cli.run_research(
            "Which workers should this test use?",
            topic="fleet-warning",
            agent="pytest",
        )

    assert result.fleet_warning is not None
    assert warning_fragment in result.fleet_warning
    assert "ROUTER FLEETS ERROR" in caplog.text
    assert warning_fragment in caplog.text
    assert result.session.final_status == FinalStatus.INSUFFICIENT_EVIDENCE
    assert result.session.territories == []
    assert saved_sessions


def test_business_question_routes_deep_research_to_web_social_lanes():
    question = (
        "Name 3 well-known companies often called 'the Uber of X' and what "
        "industry each brought the model into"
    )
    deep_sub_questions = [f"example-seeking angle {index}" for index in range(1, 17)]

    territories = research_cli.build_territories(
        deep_sub_questions,
        research_cli.deep_research_territory_specs(),
        provider="linkup",
        protocol=Protocol.DEEP_RESEARCH,
        original_question=question,
    )

    assigned_lanes = {
        lane for territory in territories for lane in territory.assigned_lanes
    }
    forbidden_lanes = {
        "arxiv",
        "pubmed",
        "semantic_scholar",
        "papers_with_code",
        "github_code",
        "sourcegraph",
        "stack_exchange",
    }
    direct_provider_lanes = {
        "linkup_direct",
        "tavily_direct",
        "youcom_direct",
        "exa_direct",
    }

    assert forbidden_lanes.isdisjoint(assigned_lanes)
    assert all("searxng_general" in territory.assigned_lanes for territory in territories)
    assert all(
        direct_provider_lanes.intersection(territory.assigned_lanes)
        for territory in territories
    )
    assert [territory.assigned_worker_model for territory in territories].count(
        WorkerModel.HAIKU
    ) == 10
    assert [territory.assigned_worker_model for territory in territories].count(
        WorkerModel.CODEX_5_4
    ) == 5
    assert [territory.assigned_worker_model for territory in territories].count(
        WorkerModel.GEMINI_FLASH
    ) == 0
    assert [territory.assigned_worker_model for territory in territories].count(
        WorkerModel.GROK
    ) == 1


def test_example_seeking_decomposition_removes_majority_meta_questions(monkeypatch):
    question = (
        "Name 3 well-known companies often called 'the Uber of X' and what "
        "industry each brought the model into"
    )

    def fake_llm(prompt, **kwargs):
        assert "concrete example-seeking search queries" in prompt
        return (
            json.dumps(
                [
                    "What does the phrase 'the Uber of X' mean in this context?",
                    "What counts as a well-known company for this question?",
                    "Which companies are commonly described by journalists as the Uber of X?",
                    "Which three companies are the strongest examples to use?",
                    "How should the answer be formatted?",
                ]
            ),
            "mock",
        )

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    sub_questions = research_cli.decompose_question(question, 5)
    meta_questions = [
        sub_question
        for sub_question in sub_questions
        if research_cli.is_meta_subquestion(sub_question)
    ]

    assert len(sub_questions) == 5
    assert len(meta_questions) == 0
    assert all(
        any(term in sub_question.lower() for term in ("company", "companies", "startup"))
        for sub_question in sub_questions
    )
    assert any("uber of x" in sub_question.lower() for sub_question in sub_questions)


def test_run_deep_research_attaches_gemini_record_before_gated_save(monkeypatch, tmp_path):
    _saved_sessions, _search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        fake_save=False,
    )
    llm_calls = []

    def fake_llm(prompt, **kwargs):
        llm_calls.append(prompt)
        if "exactly 16 non-overlapping sub-questions" in prompt:
            assert "Gemini scout context" in prompt
            return (
                json.dumps(
                    [f"deep research angle {index}" for index in range(1, 17)]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return (f"Deep territory summary {len(llm_calls)} [1]", "mock")
        return ("Deep grounded answer [1]\nCaveats/disagreements: none.", "mock")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_deep_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.path is not None
    assert result.path.exists()
    # Purpose is scout attachment before gated save, not COMPLETE from fail-open FULL.
    assert result.session.gemini_pro_runs
    assert any(run.run_type == GeminiProRunKind.SCOUT for run in result.session.gemini_pro_runs)
    run = result.session.gemini_pro_runs[0]
    assert run.run_type == GeminiProRunKind.SCOUT
    assert run.success is True
    assert run.model_id == CANONICAL_GEMINI_PRO_MODEL_ID
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["gemini_pro_runs"][0]["run_type"] == GeminiProRunKind.SCOUT.value
    assert payload["gemini_pro_runs"][0]["model_id"] == CANONICAL_GEMINI_PRO_MODEL_ID


def test_run_research_uses_pro_synthesis_fallback_record_when_scout_fails(
    monkeypatch,
    tmp_path,
):
    _saved_sessions, _search_calls = _install_research_fakes(
        monkeypatch,
        tmp_path,
        fake_save=False,
    )

    monkeypatch.setattr(
        research_cli,
        "run_gemini_scout",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.SCOUT,
            "scout unavailable",
        ),
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_pro_synthesis_fallback",
        lambda *args, **kwargs: _successful_gemini_attempt(
            GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
            "Sonnet fallback answer [1]\nCaveats/disagreements: none.",
            model_id=WorkerModel.SONNET.value,
        ),
    )

    def fake_llm(prompt, **kwargs):
        if "exactly 3 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [
                        "current options",
                        "selection criteria",
                        "known problems",
                    ]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return ("Territory summary [1]", "mock")
        raise AssertionError("final synthesis should use Gemini fallback")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.backend == WorkerModel.SONNET.value
    assert result.session.answer == "Sonnet fallback answer [1]\nCaveats/disagreements: none."
    run = result.session.gemini_pro_runs[0]
    assert run.run_type == GeminiProRunKind.PRO_SYNTHESIS_FALLBACK
    assert run.success is True
    assert run.model_id == WorkerModel.SONNET.value


def test_run_final_synthesis_falls_back_opus_then_codex_then_sonnet(monkeypatch) -> None:
    calls = []

    def fake_complete(prompt, **kwargs):
        calls.append((kwargs["backend"], kwargs.get("model")))
        if kwargs["backend"] == WorkerModel.OPUS.value:
            raise RuntimeError("opus unavailable")
        if kwargs["backend"] == "codex":
            raise RuntimeError("codex unavailable")
        return ("Sonnet answer [1]\nCaveats/disagreements: none.", kwargs["backend"])

    monkeypatch.setattr(research_cli.llm_call, "llm_complete_with_backend", fake_complete)

    attempt = research_cli.run_gemini_final_synthesis(
        "current open source alternatives to OpenAI Whisper",
        router=None,
        protocol=Protocol.RESEARCH,
        topic="whisper-alternatives",
        runs=[],
    )

    assert calls == [
        (WorkerModel.OPUS.value, None),
        ("codex", "gpt-5.5"),
        (WorkerModel.SONNET.value, None),
    ]
    assert attempt.record is not None
    assert attempt.record.model_id == WorkerModel.SONNET.value
    assert attempt.output_text == "Sonnet answer [1]\nCaveats/disagreements: none."


def test_run_research_uses_actual_final_backend_when_opus_falls_back_to_codex(
    monkeypatch,
    tmp_path,
):
    _saved_sessions, _search_calls = _install_research_fakes(monkeypatch, tmp_path, fake_save=False)

    monkeypatch.setattr(
        research_cli,
        "run_gemini_final_synthesis",
        lambda *args, **kwargs: _successful_gemini_attempt(
            GeminiProRunKind.FINAL_SYNTHESIS,
            "Codex final answer [1]\nCaveats/disagreements: none.",
            model_id=WorkerModel.CODEX_5_5.value,
        ),
    )

    def fake_llm(prompt, **kwargs):
        if "exactly 3 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [
                        "current options",
                        "selection criteria",
                        "known problems",
                    ]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return ("Territory summary [1]", "mock")
        raise AssertionError("final synthesis should use the injected fallback answer")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    assert result.backend == WorkerModel.CODEX_5_5.value
    assert result.session.answer == "Codex final answer [1]\nCaveats/disagreements: none."


def test_run_research_gemini_unavailable_direct_abstain_does_not_crash(
    monkeypatch,
    tmp_path,
    capsys,
):
    _saved_sessions, _search_calls = _install_research_fakes(monkeypatch, tmp_path)

    monkeypatch.setattr(
        research_cli,
        "run_gemini_scout",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.SCOUT,
            "scout unavailable",
        ),
    )
    monkeypatch.setattr(
        research_cli,
        "run_gemini_pro_synthesis_fallback",
        lambda *args, **kwargs: _failed_gemini_attempt(
            GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
            "fallback unavailable",
        ),
    )

    def fake_llm(prompt, **kwargs):
        if "exactly 3 non-overlapping sub-questions" in prompt:
            return (
                json.dumps(
                    [
                        "current options",
                        "selection criteria",
                        "known problems",
                    ]
                ),
                "mock",
            )
        if "Summarize this territory" in prompt:
            return ("Territory summary [1]", "mock")
        raise AssertionError("final synthesis should not run when Gemini is unavailable")

    monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

    result = research_cli.run_research(
        "current open source alternatives to OpenAI Whisper",
        topic="whisper-alternatives",
        agent="pytest",
    )

    captured = capsys.readouterr()
    assert "evidence gate bypassed: Gemini unavailable" in captured.err
    assert result.path is not None
    assert result.path.exists()
    assert result.session.final_status == FinalStatus.INSUFFICIENT_EVIDENCE
    assert result.session.answer_kind == AnswerKind.ABSTAIN
    assert result.session.answer is None
    assert not result.session.gemini_pro_runs
    assert "evidence gate bypassed: Gemini unavailable" in result.session.open_questions[0]


def test_execute_gemini_worker_spec_uses_agy_with_explicit_gemini_model(monkeypatch, tmp_path):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("Scout this question.", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="gemini response https://example.com/source\n",
            stderr="",
        )

    class Router:
        def scout_config(self):
            return {
                "cli_home": "/tmp/gemini-home",
                "health_check_timeout_seconds": 12,
            }

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", tmp_path / "counter.json")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
        model_id="Gemini 3.7 Flash (Medium)",
    )

    output = research_cli.execute_gemini_worker_spec(spec, router=Router())

    assert output == "gemini response https://example.com/source"
    assert output_path.read_text(encoding="utf-8") == (
        "gemini response https://example.com/source\n"
    )
    assert calls
    cmd, kwargs = calls[0]
    assert cmd == [
        dispatcher.AGY_CLI,
        "--dangerously-skip-permissions",
        "-p",
        "Scout this question.",
        "--model",
        "Gemini 3.7 Flash (Medium)",
    ]
    assert "input" not in kwargs
    assert "env" not in kwargs
    assert kwargs["stdin"] == research_cli.subprocess.DEVNULL
    assert kwargs["timeout"] == research_cli.GEMINI_TIMEOUT_SECONDS
    assert json.loads((tmp_path / "counter.json").read_text(encoding="utf-8"))["used"] == 1


def test_execute_gemini_worker_spec_routes_to_fallback_when_daily_cap_hit(
    monkeypatch,
    tmp_path,
):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    counter_path = tmp_path / "counter.json"
    brief_path.write_text("Scout this question.", encoding="utf-8")
    counter_path.write_text(
        json.dumps({"date": datetime.now().date().isoformat(), "used": 1}) + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="fallback response https://example.com/source\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", counter_path)
    monkeypatch.setenv("RESEARCH_ENGINE_GEMINI_DAILY_BUDGET", "1")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
        model_id="Gemini 3.7 Flash (Medium)",
    )

    output = research_cli.execute_gemini_worker_spec(spec, router=object())

    assert output == "fallback response https://example.com/source"
    cmd, _kwargs = calls[0]
    assert cmd[cmd.index("--model") + 1] == "GPT-OSS 120B (Medium)"
    assert json.loads(counter_path.read_text(encoding="utf-8"))["used"] == 1


def test_execute_gemini_worker_spec_routes_unattended_run_to_full_quota_model(
    monkeypatch,
    tmp_path,
):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    counter_path = tmp_path / "counter.json"
    brief_path.write_text("Nightly research.", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="nightly response https://example.com/source\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", counter_path)
    monkeypatch.setenv("RESEARCH_ENGINE_UNATTENDED", "launchd")
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
        model_id="Gemini 3.7 Flash (Medium)",
    )

    output = research_cli.execute_gemini_worker_spec(spec, router=object())

    assert output == "nightly response https://example.com/source"
    cmd, _kwargs = calls[0]
    assert cmd[cmd.index("--model") + 1] == "GPT-OSS 120B (Medium)"
    assert not counter_path.exists()


def test_execute_gemini_worker_spec_prefixes_dash_prefixed_prompt(monkeypatch, tmp_path):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("-start with a flag-like line", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="grounded response https://example.com/source\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", tmp_path / "counter.json")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
    )

    output = research_cli.execute_gemini_worker_spec(spec, router=object())

    assert output == "grounded response https://example.com/source"
    cmd, _kwargs = calls[0]
    assert cmd == [
        dispatcher.AGY_CLI,
        "--dangerously-skip-permissions",
        "-p",
        "Brief:\n-start with a flag-like line",
        "--model",
        "Gemini 3.7 Flash (Medium)",
    ]


def test_execute_gemini_worker_spec_rejects_oversized_prompt(tmp_path):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("a" * 200_001, encoding="utf-8")
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
    )

    with pytest.raises(research_cli.GeminiProScoutError) as excinfo:
        research_cli.execute_gemini_worker_spec(spec, router=object())

    assert str(excinfo.value) == "brief too large for agy argv: 200001 bytes; trim the brief"


def test_execute_gemini_worker_spec_writes_stdout_only_and_warns_on_stderr(
    monkeypatch, tmp_path
):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("Scout this question.", encoding="utf-8")
    warnings = []

    def fake_run(cmd, **kwargs):
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="stdout answer https://example.com/source\n",
            stderr="non-fatal stderr noise",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", tmp_path / "counter.json")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    monkeypatch.setattr(
        research_cli.logger,
        "warning",
        lambda message, detail: warnings.append((message, detail)),
    )
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
    )

    output = research_cli.execute_gemini_worker_spec(spec, router=object())

    assert output == "stdout answer https://example.com/source"
    assert output_path.read_text(encoding="utf-8") == (
        "stdout answer https://example.com/source\n"
    )
    assert warnings == [("agy Gemini worker stderr: %s", "non-fatal stderr noise")]


def test_execute_gemini_worker_spec_rejects_failure_stub(monkeypatch, tmp_path):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("Scout this question.", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Error: timeout waiting for response\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", tmp_path / "counter.json")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["gemini_pro_scout"],
        rationale="test",
    )

    with pytest.raises(research_cli.GeminiProScoutError) as excinfo:
        research_cli.execute_gemini_worker_spec(spec, router=object())

    assert "failure stub" in str(excinfo.value)
    assert not output_path.exists()


def test_failed_gemini_exit_releases_budget_without_charge(monkeypatch, tmp_path) -> None:
    brief_path = tmp_path / "brief.md"
    brief_path.write_text("Scout this question.", encoding="utf-8")
    counter_path = tmp_path / "counter.json"

    def fake_run(cmd, **kwargs):
        return research_cli.subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="authentication failed",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(research_cli, "GEMINI_DAILY_COUNTER_FILE", counter_path)
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(tmp_path / "output.md"),
        lanes=["gemini_pro_scout"],
        rationale="test",
        model_id="Gemini 3.7 Flash (Medium)",
    )

    with pytest.raises(research_cli.GeminiProScoutError):
        research_cli.execute_gemini_worker_spec(spec, router=object())

    payload = json.loads(counter_path.read_text(encoding="utf-8"))
    assert payload["used"] == 0
    assert payload["reserved"] == 0


def test_scout_record_uses_model_actually_run(monkeypatch, tmp_path) -> None:
    brief_path = tmp_path / "brief.md"
    brief_path.write_text("Nightly scout.", encoding="utf-8")
    spec = research_cli.WorkerSpec(
        worker_model="gemini-flash",
        provider="agy_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(tmp_path / "output.md"),
        lanes=["gemini_pro_scout"],
        rationale="test",
        model_id="Gemini 3.7 Flash (Medium)",
    )

    def fake_run(cmd, **kwargs):
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="nightly result https://example.com/source\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli, "dispatch_scout", lambda *args, **kwargs: spec)
    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        research_cli,
        "GEMINI_DAILY_COUNTER_FILE",
        tmp_path / "counter.json",
    )
    monkeypatch.setenv("MENTOR_NIGHTLY_RUN", "1")

    attempt = research_cli.run_gemini_scout(
        "nightly question",
        router=object(),
        protocol=Protocol.RESEARCH,
        topic="nightly",
    )

    assert attempt.record is not None
    assert attempt.record.model_id == "GPT-OSS 120B (Medium)"


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("RESEARCH_ENGINE_UNATTENDED", "nightly"),
        ("MENTOR_NIGHTLY_RUN", "1"),
    ],
)
def test_cli_and_dispatcher_share_unattended_detection(
    monkeypatch,
    env_name,
    env_value,
) -> None:
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)
    monkeypatch.setenv(env_name, env_value)

    assert research_cli.is_unattended_research_run()
    assert dispatcher.is_unattended_research_run()


def test_mistral_free_keys_rotate(monkeypatch, tmp_path) -> None:
    key_path = tmp_path / "mistral-keys.env"
    key_path.write_text(
        "MISTRAL_FREE_KEY_1=key-one\n"
        "MISTRAL_FREE_KEY_2=key-two\n"
        "MISTRAL_FREE_KEY_3=key-three\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(research_cli, "MISTRAL_FREE_KEYS_PATH", key_path)
    monkeypatch.setattr(research_cli, "_MISTRAL_KEY_COUNTER", itertools.count())

    selected = [research_cli.next_mistral_free_key() for _ in range(3)]

    assert set(selected) == {"key-one", "key-two", "key-three"}
    assert selected[0] != selected[1]


def test_execute_grok_worker_spec_uses_hermes_without_shell(monkeypatch, tmp_path):
    brief_path = tmp_path / "brief.md"
    output_path = tmp_path / "output.md"
    brief_path.write_text("Run the Grok territory.", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return research_cli.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="grok response https://example.com/source\n",
            stderr="",
        )

    monkeypatch.setattr(research_cli.subprocess, "run", fake_run)
    spec = research_cli.WorkerSpec(
        worker_model="grok",
        provider="grok_cli",
        invocation_hint="",
        brief_path=str(brief_path),
        output_path=str(output_path),
        lanes=["counter_evidence"],
        rationale="test",
        model_id="grok-test-model",
    )

    output = research_cli.execute_grok_worker_spec(spec)

    assert output == "grok response https://example.com/source"
    assert output_path.read_text(encoding="utf-8") == "grok response https://example.com/source\n"
    assert calls
    cmd, kwargs = calls[0]
    assert cmd == [
        paths.executable(paths.GROK_BIN_ENV, "grok") or "grok",
        "--single",
        "Run the Grok territory.",
    ]
    assert kwargs["stdin"] == research_cli.subprocess.DEVNULL
    assert kwargs["timeout"] == research_cli.GROK_TIMEOUT_SECONDS


def test_research_and_deep_research_abstain_without_sources(monkeypatch, tmp_path):
    for runner, protocol, n_questions in (
        (research_cli.run_research, Protocol.RESEARCH, 3),
        (research_cli.run_deep_research, Protocol.DEEP_RESEARCH, 16),
    ):
        saved_sessions, _search_calls = _install_research_fakes(
            monkeypatch,
            tmp_path,
            no_sources=True,
        )

        def fake_llm(prompt, **kwargs):
            if f"exactly {n_questions} non-overlapping sub-questions" in prompt:
                return (json.dumps([f"sub-question {i}" for i in range(n_questions)]), "mock")
            raise AssertionError("LLM should not synthesize without usable sources")

        monkeypatch.setattr(research_cli.llm_call, "llm_complete", fake_llm)

        result = runner(
            "current open source alternatives to OpenAI Whisper",
            topic=f"no-sources-{protocol.value.strip('/')}",
            agent="pytest",
        )

        assert result.session.protocol == protocol
        assert result.session.final_status == FinalStatus.INSUFFICIENT_EVIDENCE
        assert result.session.answer_kind == AnswerKind.ABSTAIN
        assert result.session.answer is None
        assert not result.session.sources
        assert result.session.open_questions
        assert saved_sessions
