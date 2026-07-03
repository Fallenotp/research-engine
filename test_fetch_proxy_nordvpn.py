import pytest

from research_engine.fetch_proxy import NoProxyBackend, NordVPNSocksBackend, load_proxy_backend


class FakeNordVPNResponse:
    def __init__(self, hosts: list[str]):
        self._hosts = hosts

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, str]]:
        return [{"hostname": host, "status": "online"} for host in self._hosts]


def write_creds(tmp_path, text: str = "NORDVPN_SERVICE_USER=user\nNORDVPN_SERVICE_PASS=pass\n"):
    path = tmp_path / "nordvpn.env"
    path.write_text(text)
    return path


def mock_nordvpn_api(monkeypatch, hosts: list[str]) -> None:
    def fake_get(url, *, params, timeout):
        assert url == "https://api.nordvpn.com/v1/servers"
        assert params == {"filters[servers_technologies][identifier]": "socks", "limit": 100}
        assert timeout == 10
        return FakeNordVPNResponse(hosts)

    monkeypatch.setattr("research_engine.fetch_proxy.httpx.get", fake_get)


def test_nordvpn_acquire_returns_socks5_url_with_creds(tmp_path, monkeypatch) -> None:
    mock_nordvpn_api(monkeypatch, ["socks-nl1.nordvpn.com"])
    backend = NordVPNSocksBackend(credentials_path=write_creds(tmp_path))

    session = backend.acquire(domain="example.com", sticky=False)

    assert session.proxy_url == "socks5://user:pass@socks-nl1.nordvpn.com:1080"
    assert session.label == "socks-nl1.nordvpn.com"


def test_nordvpn_acquire_rotates_servers(tmp_path, monkeypatch) -> None:
    mock_nordvpn_api(monkeypatch, ["socks-nl1.nordvpn.com", "socks-nl2.nordvpn.com"])
    backend = NordVPNSocksBackend(credentials_path=write_creds(tmp_path))

    first = backend.acquire(domain="example.com", sticky=False)
    second = backend.acquire(domain="example.com", sticky=False)

    assert first.label == "socks-nl1.nordvpn.com"
    assert second.label == "socks-nl2.nordvpn.com"


def test_nordvpn_release_bad_skips_server(tmp_path, monkeypatch) -> None:
    mock_nordvpn_api(monkeypatch, ["socks-nl1.nordvpn.com", "socks-nl2.nordvpn.com"])
    backend = NordVPNSocksBackend(credentials_path=write_creds(tmp_path))
    first = backend.acquire(domain="example.com", sticky=False)

    backend.release(first, ok=False)
    second = backend.acquire(domain="example.com", sticky=False)

    assert second.label == "socks-nl2.nordvpn.com"


def test_nordvpn_missing_creds_raises_clearly(tmp_path) -> None:
    creds = write_creds(tmp_path, "NORDVPN_SERVICE_USER=user\n")

    with pytest.raises(RuntimeError, match="NORDVPN_SERVICE_PASS"):
        NordVPNSocksBackend(credentials_path=creds)


def test_factory_loads_nordvpn_backend(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr("research_engine.fetch_proxy.NordVPNSocksBackend", lambda: sentinel)

    assert load_proxy_backend({"proxy": {"backend": "nordvpn"}}) is sentinel


def test_factory_falls_back_to_scraperapi_when_nordvpn_creds_missing(monkeypatch) -> None:
    sentinel = object()

    def missing_creds():
        raise RuntimeError("NordVPN credentials missing")

    monkeypatch.setattr("research_engine.fetch_proxy.NordVPNSocksBackend", missing_creds)
    monkeypatch.setattr("research_engine.fetch_proxy.ScraperAPIBackend", lambda: sentinel)

    assert load_proxy_backend({"proxy": {"backend": "nordvpn"}}) is sentinel


def test_factory_falls_back_to_none_when_nordvpn_and_scraperapi_fail(monkeypatch) -> None:
    def missing_creds():
        raise RuntimeError("NordVPN credentials missing")

    def scraperapi_down():
        raise RuntimeError("scraperapi unavailable")

    monkeypatch.setattr("research_engine.fetch_proxy.NordVPNSocksBackend", missing_creds)
    monkeypatch.setattr("research_engine.fetch_proxy.ScraperAPIBackend", scraperapi_down)

    assert isinstance(load_proxy_backend({"proxy": {"backend": "nordvpn"}}), NoProxyBackend)
