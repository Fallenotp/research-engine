from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from research_engine import extractor
from research_engine import dispatcher
from research_engine import router
from research_engine.politeness import DomainCooldown, Politeness


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_ENGINE_DATA_DIR", str(tmp_path))
    extractor.reset_process_state()
    yield
    extractor.reset_process_state()


def fake_clock_politeness() -> tuple[Politeness, dict[str, float], list[float]]:
    politeness = Politeness(min_interval_s=2.0)
    state = {"now": 100.0}
    sleeps: list[float] = []
    politeness._now = lambda: state["now"]
    politeness._sleep = sleeps.append
    return politeness, state, sleeps


def test_gate_spaces_requests_per_host(monkeypatch) -> None:
    politeness, _state, sleeps = fake_clock_politeness()
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)

    extractor.fetch_gate("https://a.example/1")
    extractor.fetch_gate("https://a.example/2")
    extractor.fetch_gate("https://b.example/")

    assert sleeps == [2.0]


def test_gate_refuses_cooling_service_or_origin(monkeypatch) -> None:
    politeness, _state, _sleeps = fake_clock_politeness()
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    politeness.note_block("origin.example", 429, "60")

    with pytest.raises(DomainCooldown, match="origin.example"):
        extractor.fetch_gate(
            "https://service.example/scrape",
            group_url="https://origin.example/article",
        )

    politeness.cooldown_until.clear()
    politeness.note_block("service.example", 429, "60")
    with pytest.raises(DomainCooldown, match="service.example"):
        extractor.fetch_gate(
            "https://service.example/scrape",
            group_url="https://origin.example/article",
        )


def test_429_sets_cooldown_and_next_call_raises_without_request(monkeypatch) -> None:
    politeness, _state, _sleeps = fake_clock_politeness()
    response = Mock(status_code=429, headers={"Retry-After": "30"})
    get = Mock(return_value=response)
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(extractor.requests, "get", get)

    extractor._get("https://a.example/one")
    with pytest.raises(DomainCooldown):
        extractor._get("https://a.example/two")

    assert get.call_count == 1


@pytest.mark.parametrize("status", [403, 429])
def test_crawl4ai_block_status_starts_origin_cooldown(monkeypatch, status) -> None:
    politeness, _state, _sleeps = fake_clock_politeness()
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(
        extractor.subprocess,
        "run",
        lambda *_args, **_kwargs: Mock(returncode=status, stdout=""),
    )

    extractor._crawl4ai("https://origin.example/article")

    assert politeness.in_cooldown("origin.example") is not None


def test_reddit_aliases_share_one_cooldown(monkeypatch) -> None:
    politeness, _state, _sleeps = fake_clock_politeness()
    response = Mock(status_code=429, headers={})
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(extractor.requests, "get", Mock(return_value=response))

    extractor._get("https://old.reddit.com/x")
    with pytest.raises(DomainCooldown):
        extractor._get("https://www.reddit.com/y")


def test_cooldown_expires(monkeypatch) -> None:
    politeness, state, _sleeps = fake_clock_politeness()
    get = Mock(return_value=Mock(status_code=429, headers={"Retry-After": "1"}))
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(extractor.requests, "get", get)

    extractor._get("https://a.example/one")
    state["now"] += 1.1
    extractor._get("https://a.example/two")

    assert get.call_count == 2


def test_ladder_skips_direct_rungs_on_cooldown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        extractor, "READER_TELEMETRY_LOG", tmp_path / "research-reader-telemetry.jsonl", raising=False
    )
    politeness, _state, _sleeps = fake_clock_politeness()
    politeness.note_block("example.com", 429, "60")
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(extractor, "WEB_RUNGS", (extractor.Rung("trafilatura", lambda url: extractor._trafilatura(url, None), lambda: (True, ""), False),))
    extractor._rung_availability.cache_clear()
    transport = Mock()
    monkeypatch.setattr(extractor.requests, "get", transport)
    result = extractor._extract_web_ladder("https://example.com/a", min_chars=200, tier=None)

    assert result is None
    rows = [json.loads(line) for line in (tmp_path / "research-reader-telemetry.jsonl").read_text().splitlines()]
    cooldown = [row for row in rows if row["status"] == "domain_cooldown"]
    assert cooldown
    assert {row["method"] for row in cooldown} == {"trafilatura"}
    assert all(row["reason"].startswith("example.com ") for row in cooldown)
    transport.assert_not_called()


def test_head_content_type_is_memoised_per_url(monkeypatch) -> None:
    head = Mock(return_value=Mock(status_code=200, headers={"content-type": "text/html"}))
    monkeypatch.setattr(extractor, "fetch_gate", Mock())
    monkeypatch.setattr(extractor.requests, "head", head)
    url = "https://a.example/a"

    assert [extractor._head_content_type(url) for _ in range(3)] == ["text/html"] * 3
    assert head.call_count == 1


def test_blocked_head_is_not_cached_after_cooldown(monkeypatch) -> None:
    politeness, state, _sleeps = fake_clock_politeness()
    head = Mock(
        side_effect=[
            Mock(status_code=429, headers={"Retry-After": "1"}),
            Mock(status_code=200, headers={"content-type": "application/pdf"}),
        ]
    )
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    monkeypatch.setattr(extractor.requests, "head", head)
    url = "https://a.example/a"

    assert extractor._head_content_type(url) is None
    state["now"] += 1.1
    assert extractor._head_content_type(url) == "application/pdf"
    assert head.call_count == 2


def test_reset_process_state_clears_cooldown_and_caches(monkeypatch) -> None:
    head = Mock(return_value=Mock(status_code=200, headers={"content-type": "text/html"}))
    monkeypatch.setattr(extractor.requests, "head", head)
    url = "https://a.example/a"

    politeness = extractor._get_politeness()
    extractor._rung_availability("trafilatura")
    assert extractor._rung_availability.cache_info().currsize == 1
    assert extractor._head_content_type(url) == "text/html"
    assert head.call_count == 1
    politeness.note_block("a.example", 429, "30")
    assert politeness.in_cooldown("a.example") is not None

    extractor.reset_process_state()

    assert extractor._POLITENESS is None
    assert extractor._rung_availability.cache_info().currsize == 0
    assert extractor._head_content_type(url) == "text/html"
    assert head.call_count == 2
    assert extractor._get_politeness().in_cooldown("a.example") is None


@pytest.mark.parametrize("rung", ["trafilatura", "jina", "firecrawl", "crawl4ai", "scrapling", "crawlee", "cloudflare_markdown"])
def test_every_page_request_passes_the_gate(monkeypatch, rung) -> None:
    politeness, _state, _sleeps = fake_clock_politeness()
    monkeypatch.setattr(extractor, "_get_politeness", lambda: politeness)
    url = "https://a.example/a"
    payload = extractor._payload("Title", "body " * 100)
    if rung == "trafilatura":
        monkeypatch.setattr(extractor.requests, "get", lambda *_args, **_kwargs: Mock(status_code=200, headers={}, text="<p>body</p>", raise_for_status=lambda: None))
        extractor._trafilatura(url, None)
    elif rung == "jina":
        monkeypatch.setattr(extractor.requests, "get", lambda *_args, **_kwargs: Mock(status_code=200, headers={}, text="Title: T\nbody", raise_for_status=lambda: None))
        extractor._jina(url)
    elif rung == "firecrawl":
        monkeypatch.setattr(
            router,
            "load_router",
            lambda: Mock(config={"lanes": {"firecrawl_direct": {"type": "api"}}}),
        )
        monkeypatch.setattr(
            dispatcher,
            "build_api_lane_request",
            lambda *_args: Mock(
                method="POST",
                url="http://localhost:18791/search",
                headers={},
                body="{}",
            ),
        )
        monkeypatch.setattr(
            extractor.requests,
            "request",
            Mock(
                return_value=Mock(
                    status_code=200,
                    headers={},
                    json=lambda: {"results": [{"url": url, "markdown": payload["text"]}]},
                    raise_for_status=lambda: None,
                )
            ),
        )
        extractor._firecrawl_proxy(url)
    elif rung == "crawl4ai":
        monkeypatch.setattr(extractor.subprocess, "run", lambda *_args, **_kwargs: Mock(stdout="body"))
        extractor._crawl4ai(url)
    elif rung == "scrapling":
        monkeypatch.setattr(extractor, "_get_proxy_backend", lambda: Mock(acquire=lambda **_: Mock(proxy_url=None), release=lambda *_args, **_kwargs: None))
        monkeypatch.setattr(extractor, "_stealthy_fetcher", lambda: Mock(fetch=lambda **_: Mock(html_content="<p>body</p>", status=200)))
        extractor._scrapling_stealth(url)
    elif rung == "crawlee":
        monkeypatch.setattr(extractor, "_get_proxy_backend", lambda: Mock(acquire=lambda **_: Mock(proxy_url=None), release=lambda *_args, **_kwargs: None))
        async def fake_fetch(*_args, **_kwargs):
            return payload
        monkeypatch.setattr(extractor, "_crawlee_http_fetch", fake_fetch)
        extractor._crawlee_http(url)
    else:
        monkeypatch.setattr(extractor.requests, "get", lambda *_args, **_kwargs: Mock(status_code=404, headers={}))
        extractor._cloudflare_markdown_preflight(url)
    assert politeness._last_hit
