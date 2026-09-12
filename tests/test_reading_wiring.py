from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from research_engine import research_cli
from research_engine.schema import ExtractionMethod, SourceRecord, SourceTier, WorkerModel


def _result(number: int, *, matching: bool = False) -> dict[str, str]:
    title = "rare orchid conservation" if matching else "unrelated result"
    snippet = "rare orchid habitat facts" if matching else "other topic"
    return {"url": f"https://example.test/{number}", "title": title, "snippet": snippet}


def _extracted(url: str, tmp_path: Path, text: str = "source text") -> dict[str, str | int]:
    path = tmp_path / f"{url.rsplit('/', 1)[-1]}.txt"
    path.write_text(text, encoding="utf-8")
    return {
        "url": url,
        "domain": "example.test",
        "title": url,
        "extraction_method": "curl",
        "raw_text_path": str(path),
        "char_count": len(text),
    }


def _source(tmp_path: Path) -> SourceRecord:
    return SourceRecord(
        url="https://example.test/source",
        domain="example.test",
        title="Source",
        fetched_at=datetime.now(timezone.utc),
        content_hash="a" * 64,
        extraction_method=ExtractionMethod.CURL,
        raw_text_path=str(tmp_path / "source.txt"),
        char_count=1,
        tier=SourceTier.T2,
        topic_authority_score=0.5,
    )


def test_page_budget_table() -> None:
    assert research_cli.PAGE_BUDGET == {"/search": 5, "/research": 6, "/deep-research": 8}
    assert research_cli.page_budget_for("/search") == 5
    assert research_cli.page_budget_for("/research") == 6
    assert research_cli.page_budget_for("/deep-research") == 8
    assert research_cli.page_budget_for("unknown") == 5


def test_run_worker_reads_ranked_candidates_up_to_budget(monkeypatch, tmp_path: Path) -> None:
    results = [_result(number, matching=number == 8) for number in range(12)]
    fetched: list[str] = []

    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *args, **kwargs: {"results": results, "duration_ms": 1},
    )
    monkeypatch.setattr(
        research_cli.logged_search,
        "proxy",
        lambda *args, **kwargs: {"results": [], "duration_ms": 1},
    )

    def fake_extract(url: str, **kwargs):
        fetched.append(url)
        return _extracted(url, tmp_path)

    monkeypatch.setattr(research_cli, "extract_clean_text", fake_extract)

    sources, _texts, _calls = research_cli._run_worker(
        "rare orchid",
        protocol="/research",
        topic="test",
        agent="test",
        provider="proxy",
        worker_model=WorkerModel.HAIKU,
        seen_urls=set(),
    )

    assert "https://example.test/8" in fetched[:3]
    assert len(sources) == research_cli.page_budget_for("/research")


def test_failed_fetch_does_not_consume_budget(monkeypatch, tmp_path: Path) -> None:
    urls = [f"https://example.test/{number}" for number in range(12)]
    fetched: list[str] = []

    def fake_extract(url: str, **kwargs):
        fetched.append(url)
        if len(fetched) <= 3:
            return None
        return _extracted(url, tmp_path)

    monkeypatch.setattr(research_cli, "extract_clean_text", fake_extract)

    source_pairs = research_cli.extract_sources(
        urls,
        errors=[],
        page_budget=research_cli.page_budget_for("/search"),
    )

    assert len(source_pairs) == research_cli.page_budget_for("/search")
    assert len(fetched) == research_cli.page_budget_for("/search") + 3


def test_search_fetches_duplicate_lane_url_once_and_fills_unique_budget(
    monkeypatch, tmp_path: Path
) -> None:
    duplicate_url = "https://example.test/shared"
    searx_urls = [duplicate_url, "https://example.test/one"]
    proxy_urls = [
        duplicate_url,
        "https://example.test/two",
        "https://example.test/three",
        "https://example.test/four",
    ]
    fetched: list[str] = []

    class FakeRouter:
        def route(self, _question: str) -> SimpleNamespace:
            return SimpleNamespace(topic="test", lanes=[])

        def graduated_answer_config(self) -> dict[str, float]:
            return {
                "full_confidence_min": 0.2,
                "partial_confidence_min": 0.1,
                "abstain_confidence_below": 0.05,
            }

    def lane_result(urls: list[str]) -> dict[str, object]:
        return {
            "results": [
                {"url": url, "title": f"source {index}", "snippet": "test"}
                for index, url in enumerate(urls)
            ],
            "duration_ms": 1,
        }

    def fake_extract(url: str, **_kwargs: object) -> dict[str, str | int]:
        fetched.append(url)
        return _extracted(url, tmp_path, text=f"evidence for {url}")

    monkeypatch.setattr(research_cli, "load_router", lambda: FakeRouter())
    monkeypatch.setattr(
        research_cli.logged_search,
        "searxng",
        lambda *_args, **_kwargs: lane_result(searx_urls),
    )
    monkeypatch.setattr(
        research_cli.logged_search,
        "proxy",
        lambda *_args, **_kwargs: lane_result(proxy_urls),
    )
    monkeypatch.setattr(
        research_cli,
        "run_grok_x_search",
        lambda *_args, **_kwargs: (
            "",
            research_cli.query_call(
                "x",
                lane="grok_x_search",
                payload={"duration_ms": 1},
                worker_model=research_cli.WorkerModel.GROK,
            ),
        ),
    )
    monkeypatch.setattr(research_cli, "extract_clean_text", fake_extract)
    monkeypatch.setattr(
        research_cli.llm_call,
        "llm_complete",
        lambda *_args, **_kwargs: ("answer [1]", "codex"),
    )
    monkeypatch.setattr(research_cli, "save_session_safely", lambda _session: None)
    monkeypatch.setattr(research_cli, "telemetry_safely", lambda: None)

    result = research_cli.run_search("duplicate lane test", topic="test", agent="pytest")

    assert fetched.count(duplicate_url) == 1
    assert len(result.session.sources) == research_cli.page_budget_for("/search")


def test_prompt_uses_windows_not_prefix(tmp_path: Path) -> None:
    prefix = "A" * 20_000
    paragraph = "The rare orchid survives through a unique mycorrhizal partnership."
    full_text = prefix + "\n\n" + ("filler " * 2_500) + "\n\n" + paragraph

    prompt = research_cli.synthesis_prompt("rare orchid mycorrhizal partnership", [(_source(tmp_path), full_text)])

    assert paragraph in prompt
    assert prefix[:1_500] not in prompt


def test_evidence_chunk_offsets_are_real(tmp_path: Path) -> None:
    paragraph = "The rare orchid survives through a unique mycorrhizal partnership."
    full_text = ("background " * 500) + "\n\n" + paragraph

    chunk = research_cli.evidence_chunk(
        _source(tmp_path), full_text, "The orchid has a mycorrhizal partnership."
    )

    assert chunk.char_offset > 0
    assert full_text[chunk.char_offset : chunk.char_offset + chunk.char_length] == chunk.paragraph_text
    assert paragraph in chunk.paragraph_text


def test_no_literal_caps_remain() -> None:
    source = Path(research_cli.__file__).read_text(encoding="utf-8")

    assert "[:1500]" not in source
    assert "max_chars=1500" not in source
    assert "max_sources=2" not in source
    assert "max_sources=3" not in source
