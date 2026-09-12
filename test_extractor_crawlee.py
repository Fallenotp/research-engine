from pathlib import Path
from unittest.mock import patch

from research_engine import extractor
from research_engine.schema import ExtractionMethod


def test_crawlee_method_exists() -> None:
    assert ExtractionMethod.CRAWLEE.value == "crawlee"


def test_crawlee_proxy_configuration_allows_no_proxy_backend() -> None:
    assert extractor._crawlee_proxy_configuration(None) is None


def test_chain_uses_crawlee_after_prior_rungs_fail(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "crawlee recovered this body text " * 8,
        "author": None,
        "published_date": None,
        "fetch_meta": {
            "reader": "crawlee",
            "proxy_label": "no_proxy",
            "sticky": False,
            "ladder_depth": 6,
        },
    }

    with patch.object(extractor, "_is_document_source", return_value=False), patch.object(
        extractor, "_extract_pdf_or_document", return_value=None
    ), patch.object(extractor, "_rung_availability", return_value=(True, "")), patch.object(
        extractor, "_cloudflare_markdown", return_value={}
    ), patch.object(extractor, "_trafilatura", return_value={}
    ), patch.object(extractor, "_firecrawl", return_value={}
    ), patch.object(
        extractor,
        "_crawl4ai",
        return_value={},
    ) as crawl4ai_mock, patch.object(
        extractor, "_scrapling_stealth", return_value={}
    ) as scrapling_mock, patch.object(
        extractor,
        "_crawlee_http",
        return_value=good,
    ) as crawlee_mock, patch.object(
        extractor, "_jina", side_effect=AssertionError("must not be called")
    ):
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.CRAWLEE.value
    assert "crawlee recovered" in out["char_text_preview"]
    assert "crawlee recovered" in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    assert out["fetch_meta"]["reader"] == "crawlee"
    crawl4ai_mock.assert_called_once_with(url)
    scrapling_mock.assert_called_once_with(url)
    crawlee_mock.assert_called_once()


def test_chain_bypasses_later_rungs_when_crawl4ai_succeeds(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "crawl4ai recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    with patch.object(extractor, "_is_document_source", return_value=False), patch.object(
        extractor, "_extract_pdf_or_document", return_value=None
    ), patch.object(extractor, "_rung_availability", return_value=(True, "")), patch.object(
        extractor, "_cloudflare_markdown", return_value={}
    ), patch.object(extractor, "_trafilatura", return_value={}
    ), patch.object(extractor, "_firecrawl", return_value={}
    ), patch.object(
        extractor,
        "_crawl4ai",
        return_value=good,
    ), patch.object(
        extractor,
        "_crawlee_http",
    ) as crawlee_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.CRAWL4AI.value
    crawlee_mock.assert_not_called()
