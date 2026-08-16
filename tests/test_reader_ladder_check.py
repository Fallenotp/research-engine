from __future__ import annotations

from research_engine import extractor
from research_engine.tools import reader_ladder_check as checker


def test_derive_reader_rungs_from_source_not_hardcoded() -> None:
    rungs = checker.derive_reader_rungs(checker.extractor_source_text())
    methods = {rung.method for rung in rungs}

    assert methods >= {
        "docling",
        "pymupdf",
        "markitdown",
        "unstructured",
        "gitingest",
        "apify",
        "trafilatura",
        "crawl4ai",
        "jina",
        "crawlee",
        "scrapling",
        "firecrawl",
        "publisher_oa",
        "wayback",
    }

    synthetic_source = """
def extract_clean_text():
    payload = _attempt(
        ExtractionMethod.NEW_RUNG.value,
        lambda: _new_rung(),
    )
    return payload
"""
    synthetic = checker.derive_reader_rungs(synthetic_source)
    assert [rung.method for rung in synthetic] == ["new_rung"]


def test_default_mode_makes_no_network_call(tmp_path, monkeypatch) -> None:
    called: list[str] = []

    def forbidden(name: str):
        def _raise(*_args, **_kwargs):
            called.append(name)
            raise AssertionError(f"{name} should not be called in default mode")

        return _raise

    for helper_name in (
        "_pdf_docling",
        "_pdf_pymupdf",
        "_markitdown_payload",
        "_gitingest",
        "_apify_actor_fetch",
        "_trafilatura",
        "_crawl4ai",
        "_jina",
        "_crawlee_http",
        "_scrapling_stealth",
        "_firecrawl",
    ):
        monkeypatch.setattr(extractor, helper_name, forbidden(helper_name))

    monkeypatch.setattr(checker, "REPORT_PATH", tmp_path / "reader_ladder_report.json")

    report = checker.build_report(live=False)

    assert report["live"] is False
    assert called == []

def test_missing_key_reports_without_printing_secret() -> None:
    rung = checker.RungSpec(
        method="apify",
        helper="_apify_actor_fetch",
        line=1634,
        guards=("actor_route_for_url(source_url)",),
    )

    row = checker.classify_rung(rung, {"UNRELATED_SECRET": "sk-test-123"}, {})

    assert row["status"] == "MISSING_KEY"
    assert "sk-test-123" not in row["detail"]
    assert "APIFY_API_KEY" in row["detail"]


def test_is_listicle_matches_titles_and_urls() -> None:
    assert extractor._is_listicle(
        "10 Best Running Shoes",
        "https://example.org/reviews/running-shoes",
    )
    assert extractor._is_listicle(
        "A normal article title",
        "https://example.org/reviews/top-10-running-shoes",
    )
    assert extractor._is_listicle(
        "Another normal article title",
        "https://example.org/compare/nike-vs-adidas",
    )
    assert not extractor._is_listicle(
        "How supply chains changed in 2026",
        "https://example.org/news/supply-chains-2026",
    )
