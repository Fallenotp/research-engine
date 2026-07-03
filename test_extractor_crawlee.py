from pathlib import Path
from unittest.mock import patch

from research_engine import extractor
from research_engine.schema import ExtractionMethod


def test_crawlee_method_exists() -> None:
    assert ExtractionMethod.CRAWLEE.value == "crawlee"


def test_crawlee_proxy_configuration_allows_no_proxy_backend() -> None:
    assert extractor._crawlee_proxy_configuration(None) is None


def test_chain_uses_crawlee_after_crawl4ai_fails(tmp_path) -> None:
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

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_crawl4ai",
        return_value={},
    ), patch.object(
        extractor,
        "_crawlee_http",
        return_value=good,
    ) as crawlee_mock, patch.object(
        extractor,
        "_scrapling_stealth",
    ) as scrapling_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.CRAWLEE.value
    assert "crawlee recovered" in out["char_text_preview"]
    assert "crawlee recovered" in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    assert out["fetch_meta"]["reader"] == "crawlee"
    crawlee_mock.assert_called_once()
    scrapling_mock.assert_not_called()


def test_chain_bypasses_crawlee_when_crawl4ai_succeeds(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "crawl4ai recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
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
