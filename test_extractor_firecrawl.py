from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from research_engine import extractor
from research_engine.schema import ExtractionMethod


def test_firecrawl_method_exists() -> None:
    assert ExtractionMethod.FIRECRAWL.value == "firecrawl"


def test_chain_uses_firecrawl_after_scrapling_fails(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "firecrawl recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_crawl4ai",
        return_value={},
    ) as crawl4ai_mock, patch.object(
        extractor,
        "_crawlee_http",
        return_value={},
    ) as crawlee_mock, patch.object(
        extractor,
        "_scrapling_stealth",
        return_value={},
    ) as scrapling_mock, patch.object(
        extractor,
        "_firecrawl",
        return_value=good,
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.FIRECRAWL.value
    assert "firecrawl recovered" in out["char_text_preview"]
    assert "firecrawl recovered" in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    crawl4ai_mock.assert_called_once_with(url)
    crawlee_mock.assert_called_once_with(url)
    scrapling_mock.assert_called_once_with(url)
    firecrawl_mock.assert_called_once_with(url)


def test_chain_bypasses_firecrawl_when_crawl4ai_succeeds(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "jina recovered this body text " * 10,
        "author": None,
        "published_date": None,
    }

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_crawl4ai",
        return_value=good,
    ), patch.object(
        extractor,
        "_firecrawl",
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.CRAWL4AI.value
    firecrawl_mock.assert_not_called()


def test_web_ladder_does_not_attempt_removed_rungs(tmp_path) -> None:
    url = "https://protected.example/article"

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_cloudflare_markdown_preflight",
        side_effect=AssertionError("cloudflare-md is not in the web ladder"),
    ), patch.object(
        extractor,
        "_jina",
        side_effect=AssertionError("jina is not in the web ladder"),
    ), patch.object(
        extractor,
        "_trafilatura",
        side_effect=AssertionError("trafilatura is not in the web ladder"),
    ), patch.object(
        extractor,
        "_crawl4ai",
        return_value={},
    ), patch.object(
        extractor,
        "_crawlee_http",
        return_value={},
    ), patch.object(
        extractor,
        "_scrapling_stealth",
        return_value={},
    ), patch.object(
        extractor,
        "_firecrawl",
        return_value={},
    ):
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is None


class _FirecrawlResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict:
        return self._payload


def test_firecrawl_direct_env_keys_rotate_across_calls(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(extractor, "_FIRECRAWL_KEY_INDEX", 0, raising=False)
    monkeypatch.setattr(
        "research_engine.fetch_proxy.load_firecrawl_keys_from_env",
        lambda: [
            ("FIRECRAWL_API_KEY_1", "mock-one"),
            ("FIRECRAWL_API_KEY_2", "mock-two"),
        ],
    )
    monkeypatch.setattr(
        "research_engine.dispatcher.build_api_lane_request",
        lambda *_args, **_kwargs: pytest.fail("proxy lane should not be used"),
    )

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _FirecrawlResponse(
            {
                "success": True,
                "data": {
                    "markdown": "direct firecrawl body " * 20,
                    "metadata": {"title": "Direct Firecrawl"},
                },
            }
        )

    monkeypatch.setattr(extractor.requests, "request", fake_request)

    first = extractor._firecrawl("https://example.com/one")
    second = extractor._firecrawl("https://example.com/two")

    assert first["title"] == "Direct Firecrawl"
    assert second["title"] == "Direct Firecrawl"
    assert [call[2]["headers"]["Authorization"] for call in calls] == [
        "Bearer mock-one",
        "Bearer mock-two",
    ]
    assert all(call[1] == "https://api.firecrawl.dev/v2/scrape" for call in calls)


def test_firecrawl_zero_env_keys_uses_proxy_lane(monkeypatch) -> None:
    url = "https://example.com/proxy"
    build_calls = []
    request_calls = []
    monkeypatch.setattr(extractor, "_FIRECRAWL_KEY_INDEX", 0, raising=False)
    monkeypatch.setattr(
        "research_engine.fetch_proxy.load_firecrawl_keys_from_env",
        lambda: [],
    )
    monkeypatch.setattr(
        "research_engine.router.load_router",
        lambda: SimpleNamespace(config={"lanes": {"firecrawl_direct": {"type": "api"}}}),
    )

    def fake_build(lane_name, lane_config, query):
        build_calls.append((lane_name, lane_config, query))
        return SimpleNamespace(
            method="POST",
            url="http://localhost:18791/search",
            headers={},
            body='{"query":"https://example.com/proxy","provider":"firecrawl"}',
        )

    def fake_request(method, request_url, **kwargs):
        request_calls.append((method, request_url, kwargs))
        return _FirecrawlResponse(
            {
                "results": [
                    {
                        "url": url,
                        "title": "Proxy Firecrawl",
                        "markdown": "proxy firecrawl body " * 20,
                    }
                ]
            }
        )

    monkeypatch.setattr("research_engine.dispatcher.build_api_lane_request", fake_build)
    monkeypatch.setattr(extractor.requests, "request", fake_request)

    result = extractor._firecrawl(url)

    assert result["title"] == "Proxy Firecrawl"
    assert build_calls == [("firecrawl_direct", {"type": "api"}, url)]
    assert request_calls[0][1] == "http://localhost:18791/search"
