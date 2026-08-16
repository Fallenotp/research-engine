from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

import pytest

import research_engine.grounding as grounding


def _fake_extract(monkeypatch, tmp_path: Path) -> None:
    def fake_extract(url: str, **kwargs):  # type: ignore[no-untyped-def]
        safe = url.rsplit("/", 1)[-1].replace("?", "-") or "source"
        raw_text_path = tmp_path / f"{safe}.txt"
        full_text = (
            f"Title for {url}\n\n"
            f"This source directly supports the answer for {url}."
        )
        raw_text_path.write_text(full_text, encoding="utf-8")
        return {
            "url": url,
            "domain": "example.com",
            "title": f"Title for {safe}",
            "extraction_method": "trafilatura",
            "raw_text_path": str(raw_text_path),
            "char_count": len(full_text),
        }

    monkeypatch.setattr(grounding, "extract_clean_text", fake_extract)


def _searxng_ready_for_smoke() -> bool:
    probe_url = grounding.logged_search.SEARXNG_URL + "?q=capital+france&format=json"
    try:
        with urlopen(probe_url, timeout=3) as response:
            response.read(1)
    except (OSError, TimeoutError, URLError):
        return False
    return True


def test_ground_uses_searxng_results_without_escalation(monkeypatch, tmp_path: Path) -> None:
    search_payload = {
        "results": [
            {
                "url": "https://example.com/paris",
                "title": "Paris",
                "content": "Paris is the capital of France.",
            },
            {
                "url": "https://example.com/france",
                "title": "France facts",
                "content": "The capital city of France is Paris.",
            },
        ]
    }
    backend_calls: list[str] = []
    searx_queries: list[str] = []

    def fake_searxng(query: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        searx_queries.append(query)
        return search_payload

    monkeypatch.setattr(grounding.logged_search, "searxng", fake_searxng)
    _fake_extract(monkeypatch, tmp_path)

    def fake_backend_lookup(backend: str, query: str):  # type: ignore[no-untyped-def]
        backend_calls.append(backend)
        return grounding.BackendLookupResult(answer="", urls=[])

    monkeypatch.setattr(grounding, "_run_backend_lookup", fake_backend_lookup)
    monkeypatch.setattr(
        grounding,
        "_synthesize_from_sources",
        lambda *args, **kwargs: ("grounded", "Paris is the capital of France.", 0.98),
    )

    result = grounding.ground("What is the capital of France?", topic_slug="capital-france")

    assert result.status == "grounded"
    assert result.answer == "Paris is the capital of France."
    assert result.confidence == pytest.approx(0.98)
    assert result.backends_used == ["searxng"]
    assert backend_calls == []
    assert searx_queries == ["capital france"]
    assert [source.url for source in result.sources] == [
        "https://example.com/paris",
        "https://example.com/france",
    ]
    assert all(Path(source.raw_text_path).exists() for source in result.sources)


def test_ground_escalates_then_returns_not_found_when_all_backends_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend_calls: list[str] = []

    monkeypatch.setattr(
        grounding.logged_search,
        "searxng",
        lambda *args, **kwargs: {"results": []},
    )
    _fake_extract(monkeypatch, tmp_path)

    def fake_backend_lookup(backend: str, query: str):  # type: ignore[no-untyped-def]
        backend_calls.append(backend)
        return grounding.BackendLookupResult(answer="", urls=[])

    monkeypatch.setattr(grounding, "_run_backend_lookup", fake_backend_lookup)
    monkeypatch.setattr(
        grounding,
        "_synthesize_from_sources",
        lambda *args, **kwargs: ("not_found", "", 0.0),
    )

    result = grounding.ground(
        "zzqv orbital teacup parliament 947213",
        topic_slug="nonsense-query",
    )

    assert result.status == "not_found"
    assert result.answer == ""
    assert result.confidence == 0.0
    assert result.sources == []
    assert result.backends_used == ["searxng", "grok", "gemini"]
    assert backend_calls == ["grok", "gemini"]


def test_gemini_grounding_escalation_uses_agy_cli_not_retired_gemini(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return grounding.subprocess.CompletedProcess(
            cmd,
            0,
            stdout="ANSWER:\nReady\n\nSOURCES:\nhttps://example.com\n",
            stderr="",
        )

    monkeypatch.setattr(grounding, "_agy_binary", lambda: "agy-cli-1")
    monkeypatch.setattr(grounding.subprocess, "run", fake_run)
    monkeypatch.setattr(grounding, "_append_telemetry", lambda *args, **kwargs: None)

    output = grounding._run_gemini("lookup prompt")

    assert "ANSWER:" in output
    assert calls
    cmd, kwargs = calls[0]
    assert cmd == [
        "agy-cli-1",
        "--dangerously-skip-permissions",
        "--print",
        "lookup prompt",
    ]
    assert "input" not in kwargs
    assert "env" not in kwargs
    assert kwargs["stdin"] == grounding.subprocess.DEVNULL


def test_ground_returns_not_found_for_irrelevant_noise(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        grounding.logged_search,
        "searxng",
        lambda *args, **kwargs: {
            "results": [
                {
                    "url": "https://example.com/unrelated",
                    "title": "Weather in Denver",
                    "content": "Tomorrow will be sunny and warm.",
                }
            ]
        },
    )
    _fake_extract(monkeypatch, tmp_path)
    monkeypatch.setattr(
        grounding,
        "_run_backend_lookup",
        lambda *args, **kwargs: grounding.BackendLookupResult(answer="", urls=[]),
    )
    monkeypatch.setattr(
        grounding,
        "_synthesize_from_sources",
        lambda *args, **kwargs: ("not_found", "", 0.0),
    )

    result = grounding.ground(
        "zzqxjvfrp orbital teacup parliament 947213",
        topic_slug="nonsense-noise",
    )

    assert result.status == "not_found"
    assert result.answer == ""
    assert result.confidence == 0.0
    assert result.sources == []


def test_candidate_search_urls_prefers_more_relevant_results() -> None:
    results = [
        {
            "url": "https://example.com/france-guide",
            "title": "France travel guide",
            "snippet": "A practical guide for visitors to France.",
        },
        {
            "url": "https://example.com/paris",
            "title": "Paris",
            "snippet": "Paris is the capital of France.",
        },
        {
            "url": "https://example.com/notre-dame",
            "title": "Notre-Dame history",
            "snippet": "A cathedral in Paris, France.",
        },
    ]

    urls = grounding._candidate_search_urls(results, "What is the capital of France?")

    assert urls == ["https://example.com/paris"]


def test_synthesize_uses_llm_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDING_DISABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.setattr(
        grounding,
        "_llm_synthesis",
        lambda *args, **kwargs: ("grounded", "Paris is the capital of France.", 0.91),
    )
    monkeypatch.setattr(
        grounding,
        "_heuristic_answer",
        lambda *args, **kwargs: pytest.fail("heuristic fallback should not run when LLM succeeds"),
    )

    result = grounding._synthesize_from_sources(
        "What is the capital of France?",
        [object()],
        search_results=[],
        backend_answers=[],
    )

    assert result == ("grounded", "Paris is the capital of France.", 0.91)


def test_synthesize_falls_back_to_heuristic_when_llm_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDING_DISABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.setattr(grounding, "_llm_synthesis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        grounding,
        "_heuristic_answer",
        lambda *args, **kwargs: "Paris is the capital of France.",
    )
    source = grounding._VerifiedSource(
        source=grounding.GroundSource(
            url="https://example.com/paris",
            tier=grounding.SourceTier.T2,
            raw_text_path="/tmp/paris.txt",
        ),
        record=SimpleNamespace(topic_authority_score=0.5),
        full_text="Paris is the capital of France.",
        snippet_hint="",
    )

    result = grounding._synthesize_from_sources(
        "What is the capital of France?",
        [source],
        search_results=[],
        backend_answers=[],
    )

    assert result == (
        "partial",
        "Paris is the capital of France.",
        pytest.approx(0.43333333333333335),
    )


def _verified_source(
    tmp_path: Path,
    *,
    url: str,
    authority: float,
) -> grounding._VerifiedSource:
    raw_text_path = tmp_path / f"{url.rsplit('/', 1)[-1]}.txt"
    full_text = f"Evidence for {url}."
    raw_text_path.write_text(full_text, encoding="utf-8")
    record = grounding.SourceRecord(
        url=url,
        domain="example.com",
        title=f"Title for {url}",
        fetched_at=grounding.datetime.now(grounding.timezone.utc),
        content_hash=grounding.SourceRecord.hash_text(full_text),
        extraction_method=grounding.ExtractionMethod.CURL,
        raw_text_path=raw_text_path,
        char_count=len(full_text),
        tier=grounding.SourceTier.T2,
        topic_authority_score=authority,
    )
    return grounding._VerifiedSource(
        source=grounding.GroundSource(
            url=url,
            tier=record.tier,
            raw_text_path=str(raw_text_path),
        ),
        record=record,
        full_text=full_text,
        snippet_hint="",
    )


def test_synthesize_confidence_changes_with_authority_same_source_count(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GROUNDING_DISABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.setattr(grounding, "_llm_synthesis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        grounding,
        "_heuristic_answer",
        lambda *args, **kwargs: "Measured answer.",
    )
    monkeypatch.setattr(
        grounding,
        "_grounding_status_from_confidence",
        lambda confidence: "grounded" if confidence >= 0.7 else "partial",
    )

    low_authority_sources = [
        _verified_source(
            tmp_path,
            url="https://example.com/low-1",
            authority=0.2,
        ),
        _verified_source(
            tmp_path,
            url="https://example.com/low-2",
            authority=0.2,
        ),
    ]
    high_authority_sources = [
        _verified_source(
            tmp_path,
            url="https://example.com/high-1",
            authority=0.9,
        ),
        _verified_source(
            tmp_path,
            url="https://example.com/high-2",
            authority=0.9,
        ),
    ]

    low_result = grounding._synthesize_from_sources(
        "question",
        low_authority_sources,
        search_results=[],
        backend_answers=[],
    )
    high_result = grounding._synthesize_from_sources(
        "question",
        high_authority_sources,
        search_results=[],
        backend_answers=[],
    )

    assert low_result == ("partial", "Measured answer.", pytest.approx(0.38666666666666666))
    assert high_result == ("grounded", "Measured answer.", pytest.approx(0.8066666666666666))


def test_grounding_status_fails_down_when_thresholds_unloadable(
    monkeypatch,
) -> None:
    def fail_load_router():
        raise RuntimeError("boom")

    monkeypatch.setattr(grounding, "load_router", fail_load_router)

    assert grounding._grounding_status_from_confidence(0.95) == "not_found"


def test_ground_live_smoke_capital_of_france(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDING_DISABLE_LLM_SYNTHESIS", raising=False)
    if not _searxng_ready_for_smoke():
        pytest.skip("SearXNG at http://localhost:8888 is unreachable")

    result = grounding.ground(
        "What is the capital of France?",
        topic_slug="capital-france-live-smoke",
    )

    assert result.status in {"grounded", "partial"}
    assert result.answer
    assert "paris" in result.answer.lower()
    assert result.sources
    assert all(Path(source.raw_text_path).exists() for source in result.sources)


def test_run_grok_uses_grok_cli_with_locked_model(monkeypatch) -> None:
    commands: list[list[str]] = []
    telemetry_calls: list[tuple[str, str, str, str]] = []

    def fake_run(argv: list[str], **kwargs):  # type: ignore[no-untyped-def]
        commands.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(grounding, "_is_executable", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        grounding,
        "_append_telemetry",
        lambda *args: telemetry_calls.append(args),
    )
    monkeypatch.setattr(grounding.subprocess, "run", fake_run)

    monkeypatch.setenv("GROUNDING_GROK_BIN", str(grounding.DEFAULT_GROK_BIN))
    monkeypatch.delenv("GROUNDING_GROK_MODEL", raising=False)
    assert grounding._run_grok("Grok CLI prompt") == "ok"
    assert commands[0] == [
        str(grounding.DEFAULT_GROK_BIN),
        "-p",
        "Grok CLI prompt",
    ]
    assert telemetry_calls == [
        ("grounding_escalation_grok", "grok-4.5", "Grok CLI prompt", "ok")
    ]
