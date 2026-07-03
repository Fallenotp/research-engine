from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from research_engine import evidence_gate, sufficiency
from research_engine.evidence_gate import enforce_evidence_gate
from research_engine.persistence import save_session
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


class FakeClient:
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
        raw_text_path=Path(f"/tmp/research-engine-sufficiency/source-{index}.txt"),
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
        gemini_pro_runs=[
            GeminiProRunRecord(
                run_type=GeminiProRunKind.SCOUT,
                success=True,
                model_id="gemini-3-flash",
            )
        ],
    )


class SufficiencyTests(unittest.TestCase):
    def test_flash_cli_client_runs_in_sealed_env_with_temp_cwd(self) -> None:
        client = sufficiency.CLIJsonClient(
            provider_label="flash",
            model_id="gemini-3.1-flash-lite-preview",
            executable="/Users/cleo/bin/agy-cli-1",
            timeout_seconds=5,
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"verdict":"insufficient","reason":"no direct answer","missing":[]}',
            stderr="",
        )

        with patch.dict(os.environ, {"IDENTITY": "leak", "OPENCLAW_RULES": "leak"}, clear=False):
            with patch("research_engine.sufficiency.subprocess.run", return_value=completed) as run_mock:
                client.generate_json("judge prompt")

        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        env = kwargs["env"]

        self.assertEqual(
            command,
            [
                "/Users/cleo/bin/agy-cli-1",
                "--dangerously-skip-permissions",
                "--print",
                "judge prompt",
            ],
        )
        self.assertNotIn("input", kwargs)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotEqual(kwargs["cwd"], "/Users/cleo")
        self.assertTrue(kwargs["cwd"].startswith(tempfile.gettempdir()))
        self.assertEqual(set(env), {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"})
        self.assertEqual(env["HOME"], str(Path.home()))
        self.assertNotIn("IDENTITY", env)
        self.assertNotIn("OPENCLAW_RULES", env)

    def test_ollama_cli_client_runs_with_local_model(self) -> None:
        client = sufficiency.CLIJsonClient(
            provider_label="ollama-fallback",
            model_id="qwen3.5:9b",
            executable="/opt/homebrew/bin/ollama",
            timeout_seconds=5,
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"verdict":"insufficient","reason":"no direct answer","missing":[]}',
            stderr="",
        )

        with patch("research_engine.sufficiency.subprocess.run", return_value=completed) as run_mock:
            client.generate_json("judge prompt")

        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(command, ["/opt/homebrew/bin/ollama", "run", "qwen3.5:9b"])
        self.assertEqual(set(kwargs["env"]), {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"})
        self.assertEqual(kwargs["env"]["HOME"], str(Path.home()))

    def test_get_sufficiency_clients_prefers_gemini_then_ollama(self) -> None:
        with patch.object(
            sufficiency,
            "_resolve_agy_executable",
            return_value="/Users/cleo/bin/agy-cli-1",
        ):
            with patch.object(
                sufficiency,
                "_resolve_ollama_runtime",
                return_value=("/opt/homebrew/bin/ollama", "qwen3.5:9b"),
            ):
                primary, fallback = sufficiency.get_sufficiency_clients()

        self.assertEqual(primary.provider_label, "flash")
        self.assertEqual(primary.model_id, "gemini-3.1-flash-lite-preview")
        self.assertEqual(fallback.provider_label, "ollama-fallback")
        self.assertEqual(fallback.model_id, "qwen3.5:9b")

    def test_get_sufficiency_clients_skips_when_no_provider_is_available(self) -> None:
        with patch.object(sufficiency, "_resolve_agy_executable", return_value=None):
            with patch.object(sufficiency, "_resolve_ollama_runtime", return_value=None):
                primary, fallback = sufficiency.get_sufficiency_clients()

        self.assertEqual(primary.provider_label, "skip")
        self.assertEqual(fallback.provider_label, "skip")

    def test_resolve_agy_executable_accepts_command_on_path(self) -> None:
        with patch.object(sufficiency, "SUFFICIENCY_AGY_BIN", "agy"):
            with patch(
                "research_engine.sufficiency.shutil.which",
                side_effect=lambda command: "/tmp/bin/agy" if command == "agy" else None,
            ):
                self.assertEqual(sufficiency._resolve_agy_executable(), "agy")

    def test_judge_prompt_uses_original_query_and_raw_sources_only(self) -> None:
        prompt = sufficiency.build_judge_prompt(
            "get shit done",
            [
                sufficiency.SourceText(
                    source_id="reddit:1",
                    title="A music mix",
                    domain="reddit.com",
                    text="A playlist for coding focus, not a productivity method.",
                )
            ],
        )

        self.assertIn("Original query:\nget shit done", prompt)
        self.assertIn("Raw text:", prompt)
        self.assertIn("A playlist for coding focus", prompt)
        self.assertIn("must not use any synthesized answer", prompt)
        self.assertNotIn("Checked answer:", prompt)

    def test_relevance_prefilter_drops_off_topic_items_before_judging(self) -> None:
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item describes async I/O changes in PostgreSQL 18.",
                },
                {
                    "verdict": "irrelevant",
                    "reason": "This item is a music playlist and does not answer the database question.",
                },
                {
                    "verdict": "sufficient",
                    "reason": "The remaining raw source directly answers the query.",
                    "missing": [],
                },
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])

        decision = sufficiency.judge_sufficiency(
            query="What changed in PostgreSQL 18 async I/O?",
            source_texts=[
                sufficiency.SourceText(
                    "web:1",
                    "PostgreSQL 18 release",
                    "postgresql.org",
                    "PostgreSQL 18 includes async I/O changes for reads and writes.",
                ),
                sufficiency.SourceText(
                    "web:2",
                    "Coding playlist",
                    "example.com",
                    "A playlist to help you focus while coding.",
                ),
            ],
            clients=(primary, fallback),
        )

        prefilter = decision["query_relevance_prefilter"]
        self.assertEqual(decision["verdict"], "sufficient")
        self.assertEqual(decision["source_count"], 1)
        self.assertEqual(prefilter["input_source_count"], 2)
        self.assertEqual(prefilter["kept_source_count"], 1)
        self.assertEqual(prefilter["dropped_source_count"], 1)
        self.assertEqual(prefilter["dropped"][0]["source_id"], "web:2")
        self.assertNotIn("playlist", primary.prompts[-1].lower())
        self.assertIn("Single retrieved item", primary.prompts[0])

    def test_prefilter_can_be_disabled_for_fast_paths(self) -> None:
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "sufficient",
                    "reason": "The raw sources already answer the query.",
                    "missing": [],
                }
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])

        with patch.object(sufficiency, "SUFFICIENCY_PREFILTER_ENABLED", False):
            decision = sufficiency.judge_sufficiency(
                query="What changed in PostgreSQL 18 async I/O?",
                source_texts=[
                    sufficiency.SourceText(
                        "web:1",
                        "PostgreSQL 18 release",
                        "postgresql.org",
                        "PostgreSQL 18 includes async I/O changes for reads and writes.",
                    ),
                    sufficiency.SourceText(
                        "web:2",
                        "Background article",
                        "example.com",
                        "A generic PostgreSQL explainer that would normally be filtered.",
                    ),
                ],
                clients=(primary, fallback),
            )

        prefilter = decision["query_relevance_prefilter"]
        self.assertEqual(decision["verdict"], "sufficient")
        self.assertTrue(prefilter["skipped"])
        self.assertEqual(prefilter["input_source_count"], 2)
        self.assertEqual(len(primary.prompts), 1)

    def test_recovery_loop_reformulates_and_stops_when_sufficient(self) -> None:
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item mentions PostgreSQL 18 async I/O.",
                },
                {
                    "verdict": "partial",
                    "reason": "The sources mention async I/O but not adoption evidence.",
                    "missing": ["adoption evidence"],
                },
                {"query": "PostgreSQL 18 async I/O adoption evidence"},
                {
                    "verdict": "relevant",
                    "reason": "This item contains the missing adoption evidence.",
                },
                {
                    "verdict": "sufficient",
                    "reason": "The recovered raw source text answers the query.",
                    "missing": [],
                },
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])
        initial = [
            sufficiency.SourceText(
                "web:1",
                "PostgreSQL 18 release",
                "postgresql.org",
                "PostgreSQL 18 includes async I/O.",
            )
        ]
        retriever_queries: list[str] = []

        def retriever(query: str) -> list[sufficiency.SourceText]:
            retriever_queries.append(query)
            return [
                sufficiency.SourceText(
                    "web:2",
                    "PostgreSQL 18 async I/O details",
                    "postgresql.org",
                    "The release notes explain async I/O and describe production adoption evidence.",
                )
            ]

        result = sufficiency.run_sufficiency_loop(
            query="What changed in PostgreSQL 18 async I/O?",
            source_texts=initial,
            retriever=retriever,
            clients=(primary, fallback),
        )

        self.assertEqual(result["terminal_state"], "sufficient")
        self.assertEqual(result["recovery_iterations"], 1)
        self.assertEqual(retriever_queries, ["PostgreSQL 18 async I/O adoption evidence"])
        self.assertEqual(len(primary.prompts), 5)
        self.assertIn("Original query:\nWhat changed", primary.prompts[1])
        self.assertIn("Original query:\nWhat changed", primary.prompts[4])

    def test_same_items_stop_recovery_without_extra_judge(self) -> None:
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item is about the tool's feature list.",
                },
                {
                    "verdict": "partial",
                    "reason": "The sources are missing pricing.",
                    "missing": ["pricing"],
                },
                {"query": "tool pricing details"},
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])
        sources = [
            sufficiency.SourceText("web:1", "Tool page", "example.com", "Feature list only.")
        ]

        result = sufficiency.run_sufficiency_loop(
            query="What does Tool X cost?",
            source_texts=sources,
            retriever=lambda _query: sources,
            clients=(primary, fallback),
        )

        self.assertEqual(result["terminal_state"], "partial")
        self.assertEqual(result["stop_reason"], "same_items")
        self.assertLessEqual(result["recovery_iterations"], 2)
        self.assertEqual(len(primary.prompts), 3)

    def test_skip_client_returns_non_blocking_sufficient_verdict(self) -> None:
        primary = sufficiency.SkipSufficiencyClient()
        fallback = sufficiency.SkipSufficiencyClient()

        result = sufficiency.run_sufficiency_loop(
            query="What changed?",
            source_texts=[
                sufficiency.SourceText(
                    "web:1",
                    "Topic page",
                    "example.com",
                    "A short item that still requires the checker to inspect it.",
                )
            ],
            clients=(primary, fallback),
        )

        self.assertEqual(result["terminal_state"], "sufficient")
        self.assertEqual(result["final_judge"]["checker_route"], "skip")
        self.assertTrue(result["proceed"])

    def test_explicit_broken_clients_still_fail_closed_insufficient(self) -> None:
        primary = FakeClient("flash", "gemini-3.1-flash-lite-preview", [RuntimeError("flash down")])
        fallback = FakeClient("haiku-fallback", "haiku", [RuntimeError("haiku down")])

        decision = sufficiency.judge_sufficiency(
            query="What changed?",
            source_texts=[
                sufficiency.SourceText(
                    "web:1",
                    "Topic page",
                    "example.com",
                    "A short item that still requires the checker to inspect it.",
                )
            ],
            clients=(primary, fallback),
        )

        self.assertEqual(decision["verdict"], "insufficient")
        self.assertEqual(decision["checker_route"], "failed-closed")
        self.assertTrue(decision["fail_closed"])
        self.assertEqual(decision["fail_stage"], "query_relevance_prefilter")

    def test_ambiguous_plain_phrase_needs_independent_confirmation(self) -> None:
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item discusses a GSD framework.",
                },
                {
                    "verdict": "sufficient",
                    "reason": "The Reddit sources mention a GSD framework.",
                    "missing": [],
                }
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])

        decision = sufficiency.judge_sufficiency(
            query="get shit done",
            source_texts=[
                sufficiency.SourceText(
                    "reddit:1",
                    "GSD alternative",
                    "reddit.com",
                    "A Reddit post discusses an agentic framework called GSD.",
                )
            ],
            clients=(primary, fallback),
        )

        self.assertEqual(decision["verdict"], "insufficient")
        self.assertIn("short ambiguous phrase", decision["reason"])

    def test_evidence_gate_partial_preserves_answer_contract(self) -> None:
        session = _session(
            "What changed in the policy?",
            "The policy added an appeal window.",
            [
                "The policy added an appeal window.",
                "A second source confirms the appeal process changed.",
            ],
        )
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item directly addresses the policy change question.",
                },
                {
                    "verdict": "relevant",
                    "reason": "This item independently confirms the policy change.",
                },
                {
                    "verdict": "partial",
                    "reason": "The effective date is missing.",
                    "missing": ["effective date"],
                }
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])

        with patch.object(evidence_gate, "get_sufficiency_clients", return_value=(primary, fallback)):
            result = enforce_evidence_gate(session)

        ResearchSession.model_validate(result.model_dump(mode="python"))
        self.assertEqual(result.answer_kind, AnswerKind.PARTIAL)
        self.assertEqual(result.final_status, FinalStatus.WEAK_SOURCES)
        self.assertTrue(result.open_questions)
        self.assertEqual(result.evidence_gate_decision["terminal_state"], "partial")

    def test_save_session_abstains_exhausted_and_writes_sidecar(self) -> None:
        session = _session(
            "What does this prove?",
            "It proves a broad conclusion.",
            [
                "The source only mentions the topic.",
                "The second source also lacks the requested answer.",
            ],
        )
        primary = FakeClient(
            "flash",
            "gemini-3.1-flash-lite-preview",
            [
                {
                    "verdict": "relevant",
                    "reason": "This item is about the requested proof question.",
                },
                {
                    "verdict": "relevant",
                    "reason": "This second item is also about the requested proof question.",
                },
                {
                    "verdict": "insufficient",
                    "reason": "The raw text does not answer the query.",
                    "missing": ["direct answer"],
                }
            ],
        )
        fallback = FakeClient("haiku-fallback", "haiku", [])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(evidence_gate, "get_sufficiency_clients", return_value=(primary, fallback)):
                path = save_session(session, root=Path(tmpdir))
            sidecar = evidence_gate.evidence_gate_sidecar_path(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            decision = json.loads(sidecar.read_text(encoding="utf-8"))

        self.assertEqual(saved["answer_kind"], AnswerKind.ABSTAIN.value)
        self.assertEqual(saved["final_status"], FinalStatus.INSUFFICIENT_EVIDENCE.value)
        self.assertEqual(decision["terminal_state"], "exhausted")
        self.assertEqual(decision["gate_reason"], "sufficiency_exhausted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
