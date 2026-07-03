import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_engine import extractor
from research_engine.apify_accounts import ApifyAccount
from research_engine.fetch_proxy import NoProxyBackend
from research_engine.schema import ExtractionMethod


def test_scrapling_method_exists() -> None:
    assert ExtractionMethod.SCRAPLING.value == "scrapling"


def test_apify_method_exists() -> None:
    assert ExtractionMethod.APIFY.value == "apify"


def test_record_carries_fetch_meta() -> None:
    record = extractor._record(
        "https://protected.example",
        ExtractionMethod.SCRAPLING.value,
        {
            "title": "T",
            "text": "body",
            "author": None,
            "published_date": None,
        },
        tier=None,
        extra={
            "fetch_meta": {
                "reader": "scrapling",
                "proxy_label": "apify:a2:session-ab12",
                "sticky": True,
                "ladder_depth": 6,
            }
        },
    )

    assert record["fetch_meta"]["reader"] == "scrapling"
    assert record["fetch_meta"]["ladder_depth"] == 6


def test_scrapling_stealth_returns_payload() -> None:
    fake_page = MagicMock()
    fake_page.html_content = "<html><title>T</title><body>Hello world body</body></html>"
    fake_page.status = 200
    fetcher = MagicMock()
    fetcher.fetch.return_value = fake_page

    with patch.object(extractor, "_get_proxy_backend", return_value=NoProxyBackend()), patch.object(
        extractor,
        "_get_politeness",
    ) as politeness_mock, patch.object(
        extractor,
        "_stealthy_fetcher",
        return_value=fetcher,
    ):
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._scrapling_stealth("https://protected.example", sticky=False)

    assert out is not None
    assert "Hello world body" in (out.get("text") or "")
    fetcher.fetch.assert_called_once()


def test_scrapling_stealth_returns_empty_on_failure() -> None:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = RuntimeError("blocked")

    with patch.object(extractor, "_get_proxy_backend", return_value=NoProxyBackend()), patch.object(
        extractor,
        "_get_politeness",
    ) as politeness_mock, patch.object(
        extractor,
        "_stealthy_fetcher",
        return_value=fetcher,
    ):
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._scrapling_stealth("https://protected.example", sticky=False)

    assert out == {}


def test_chain_uses_scrapling_after_crawl4ai_fails(tmp_path) -> None:
    url = "https://protected.example/article"
    good = {
        "title": "T",
        "text": "scrapling recovered this body text " * 8,
        "author": None,
        "published_date": None,
        "fetch_meta": {
            "reader": "scrapling",
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
        return_value={},
    ), patch.object(
        extractor,
        "_scrapling_stealth",
        return_value=good,
    ):
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.SCRAPLING.value
    assert "scrapling recovered" in out["char_text_preview"]
    assert "scrapling recovered" in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    assert out["fetch_meta"]["reader"] == "scrapling"


def test_apify_actor_fetch_returns_payload(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    class FakePool:
        def next_account(self) -> ApifyAccount:
            return ApifyAccount(id="test-account", token="test-token")

    def fake_request(account, method: str, path: str, *, json_body=None, params=None, timeout=60):
        assert account.id == "test-account"
        calls.append((method, path, params))
        if method == "POST":
            return {"data": {"id": "run-123"}}
        if path == "/v2/actor-runs/run-123":
            return {"data": {"id": "run-123", "status": "SUCCEEDED"}}
        if path == "/v2/actor-runs/run-123/dataset/items":
            return [
                {
                    "url": "https://protected.example/article",
                    "title": "Protected title",
                    "text": "Recovered body " * 30,
                }
            ]
        if path == "/v2/actor-runs/run-123/key-value-store/records/OUTPUT":
            return {}
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(extractor, "_get_apify_account_pool", lambda: FakePool())
    monkeypatch.setattr(extractor, "_apify_api_request", fake_request)

    out = extractor._apify_actor_fetch("https://protected.example/article")

    assert out["title"] == "Protected title"
    assert "Recovered body" in (out.get("text") or "")
    assert out["fetch_meta"] == {
        "reader": "apify",
        "actor": "apify/camoufox-scraper",
        "run_id": "run-123",
        "platform": "web",
        "account_id": "test-account",
        "ladder_depth": 7,
    }
    assert calls[0][0:2] == ("POST", "/v2/acts/apify~camoufox-scraper/runs")


def test_apify_actor_fetch_reads_output_record_when_dataset_empty(monkeypatch) -> None:
    class FakePool:
        def next_account(self) -> ApifyAccount:
            return ApifyAccount(id="test-account", token="test-token")

    def fake_request(account, method: str, path: str, *, json_body=None, params=None, timeout=60):
        if method == "POST":
            return {"data": {"id": "run-456"}}
        if path == "/v2/actor-runs/run-456":
            return {"data": {"id": "run-456", "status": "SUCCEEDED"}}
        if path == "/v2/actor-runs/run-456/dataset/items":
            return []
        if path == "/v2/actor-runs/run-456/key-value-store/records/OUTPUT":
            return {
                "title": "KV Title",
                "text": "Output body " * 30,
            }
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(extractor, "_get_apify_account_pool", lambda: FakePool())
    monkeypatch.setattr(extractor, "_apify_api_request", fake_request)

    out = extractor._apify_actor_fetch("https://protected.example/article")

    assert out["title"] == "KV Title"
    assert "Output body" in (out.get("text") or "")
    assert out["fetch_meta"]["run_id"] == "run-456"


def test_apify_actor_fetch_reads_output_record_when_dataset_item_is_stub(monkeypatch) -> None:
    class FakePool:
        def next_account(self) -> ApifyAccount:
            return ApifyAccount(id="test-account", token="test-token")

    def fake_request(account, method: str, path: str, *, json_body=None, params=None, timeout=60):
        if method == "POST":
            return {"data": {"id": "run-457"}}
        if path == "/v2/actor-runs/run-457":
            return {"data": {"id": "run-457", "status": "SUCCEEDED"}}
        if path == "/v2/actor-runs/run-457/dataset/items":
            return [{"title": "Stub", "text": "too short"}]
        if path == "/v2/actor-runs/run-457/key-value-store/records/OUTPUT":
            return {
                "title": "KV Title",
                "text": "Output body " * 30,
            }
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(extractor, "_get_apify_account_pool", lambda: FakePool())
    monkeypatch.setattr(extractor, "_apify_api_request", fake_request)

    out = extractor._apify_actor_fetch("https://protected.example/article")

    assert out["title"] == "KV Title"
    assert "Output body" in (out.get("text") or "")
    assert out["fetch_meta"]["run_id"] == "run-457"


def test_apify_actor_fetch_returns_empty_on_failed_run(monkeypatch) -> None:
    class FakePool:
        def next_account(self) -> ApifyAccount:
            return ApifyAccount(id="test-account", token="test-token")

    def fake_request(account, method: str, path: str, *, json_body=None, params=None, timeout=60):
        if method == "POST":
            return {"data": {"id": "run-789"}}
        if path == "/v2/actor-runs/run-789":
            return {"data": {"id": "run-789", "status": "FAILED"}}
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(extractor, "_get_apify_account_pool", lambda: FakePool())
    monkeypatch.setattr(extractor, "_apify_api_request", fake_request)

    assert extractor._apify_actor_fetch("https://protected.example/article") == {}


def test_instagram_routes_to_apify_before_general_ladder(tmp_path) -> None:
    url = "https://www.instagram.com/p/example/"
    good = {
        "title": "T",
        "text": "apify recovered this body text " * 8,
        "author": None,
        "published_date": None,
        "fetch_meta": {
            "reader": "apify",
            "actor": "apify/instagram-scraper",
            "run_id": "run-999",
            "platform": "instagram",
            "ladder_depth": 1,
        },
    }

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_crawl4ai",
        side_effect=AssertionError("normal ladder should not run for Instagram"),
    ), patch.object(
        extractor,
        "_apify_actor_fetch",
        return_value=good,
    ) as apify_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.APIFY.value
    assert "apify recovered" in out["char_text_preview"]
    assert out["fetch_meta"]["reader"] == "apify"
    assert out["fetch_meta"]["platform"] == "instagram"
    apify_mock.assert_called_once_with(url)


def test_tiktok_routes_to_apify_before_general_ladder(tmp_path) -> None:
    url = "https://vm.tiktok.com/example/"
    good = {
        "title": "T",
        "text": "tiktok apify recovered this body text " * 8,
        "author": None,
        "published_date": None,
        "fetch_meta": {
            "reader": "apify",
            "actor": "clockworks/tiktok-scraper",
            "run_id": "run-1000",
            "platform": "tiktok",
            "ladder_depth": 1,
        },
    }

    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_crawl4ai",
        side_effect=AssertionError("normal ladder should not run for TikTok"),
    ), patch.object(
        extractor,
        "_apify_actor_fetch",
        return_value=good,
    ) as apify_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.APIFY.value
    assert out["fetch_meta"]["platform"] == "tiktok"
    apify_mock.assert_called_once_with(url)


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SCRAPLING") != "1",
    reason="set RUN_LIVE_SCRAPLING=1 to hit the network",
)
def test_live_scrapling_reads_protected_page(tmp_path) -> None:
    out = extractor.extract_clean_text(
        "https://www.scrapingcourse.com/cloudflare-challenge",
        seen_urls_path=tmp_path / "seen.txt",
    )

    assert out is not None
    assert len(Path(out["raw_text_path"]).read_text(encoding="utf-8")) > 200
