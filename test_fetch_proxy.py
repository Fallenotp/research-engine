from unittest.mock import patch

from research_engine.apify_accounts import AccountPool, ApifyAccount
from research_engine.fetch_proxy import (
    ApifyResidentialBackend,
    NoProxyBackend,
    OwnProxyBackend,
    ProxySession,
    ScraperAPIBackend,
    load_firecrawl_keys_from_env,
    load_proxy_backend,
)


def test_no_proxy_backend_returns_none_url() -> None:
    backend = NoProxyBackend()
    session = backend.acquire(domain="example.com", sticky=False)

    assert isinstance(session, ProxySession)
    assert session.proxy_url is None
    assert session.label == "no_proxy"

    backend.release(session, ok=True)


def test_apify_backend_builds_residential_sticky_url() -> None:
    pool = AccountPool([ApifyAccount(id="a0", token="t0", proxy_pass="SECRET")])
    backend = ApifyResidentialBackend(pool)

    with patch.object(pool, "_usage_usd", return_value=0.0):
        session = backend.acquire(domain="example.com", sticky=True)

    assert session.proxy_url is not None
    assert session.proxy_url.startswith("http://groups-RESIDENTIAL,session-")
    assert "SECRET@proxy.apify.com:8000" in session.proxy_url
    assert session.account_id == "a0"


def test_apify_backend_rotates_when_not_sticky() -> None:
    pool = AccountPool([ApifyAccount(id="a0", token="t0", proxy_pass="P")])
    backend = ApifyResidentialBackend(pool)

    with patch.object(pool, "_usage_usd", return_value=0.0):
        session = backend.acquire(domain="example.com", sticky=False)

    assert session.proxy_url is not None
    assert "session-" not in session.proxy_url
    assert session.proxy_url.startswith("http://groups-RESIDENTIAL:")


def test_own_proxy_backend_cycles_list() -> None:
    backend = OwnProxyBackend(["http://a:1", "socks5://b:2"])

    first = backend.acquire(domain="x.com", sticky=False).proxy_url
    second = backend.acquire(domain="x.com", sticky=False).proxy_url

    assert {first, second} == {"http://a:1", "socks5://b:2"}


def test_factory_defaults_to_no_proxy(monkeypatch) -> None:
    monkeypatch.delenv("APIFY_ACCOUNTS", raising=False)

    assert isinstance(load_proxy_backend({"proxy": {"backend": "none"}}), NoProxyBackend)


def test_scraperapi_backend_builds_proxy_mode_url(monkeypatch) -> None:
    class FakeRotator:
        def reserve_key(self) -> dict[str, object]:
            return {
                "index": 1,
                "value": "scraper-key",
                "masked": "scraper-key",
                "env_var": "SCRAPERAPI_KEY_2",
            }

    monkeypatch.setattr("research_engine.fetch_proxy._get_scraperapi_rotator", lambda: FakeRotator())
    backend = ScraperAPIBackend()

    session = backend.acquire(domain="example.com", sticky=False)

    assert session.proxy_url == (
        "http://scraperapi:scraper-key@proxy-server.scraperapi.com:8001"
    )
    assert session.label == "scraperapi:SCRAPERAPI_KEY_2"
    assert session.sticky is False


def test_factory_loads_scraperapi_backend() -> None:
    assert isinstance(load_proxy_backend({"proxy": {"backend": "scraperapi"}}), ScraperAPIBackend)


def test_scraperapi_backend_falls_back_to_direct_fetch_when_rotator_fails(monkeypatch) -> None:
    class BrokenRotator:
        def reserve_key(self) -> dict[str, object]:
            raise RuntimeError("no keys")

    monkeypatch.setattr("research_engine.fetch_proxy._get_scraperapi_rotator", lambda: BrokenRotator())
    backend = ScraperAPIBackend()

    session = backend.acquire(domain="example.com", sticky=False)

    assert session.proxy_url is None
    assert session.label == "scraperapi_unavailable"


def test_firecrawl_key_loader_uses_present_slots_and_warns(monkeypatch, caplog) -> None:
    for idx in range(1, 7):
        monkeypatch.delenv(f"FIRECRAWL_API_KEY_{idx}", raising=False)
    monkeypatch.setenv("FIRECRAWL_API_KEY_1", "test-firecrawl-one")
    monkeypatch.setenv("FIRECRAWL_API_KEY_3", "test-firecrawl-three")

    keys = load_firecrawl_keys_from_env()

    assert [env_var for env_var, _value in keys] == [
        "FIRECRAWL_API_KEY_1",
        "FIRECRAWL_API_KEY_3",
    ]
    assert "firecrawl key rotation loaded 2/6 keys" in caplog.text
