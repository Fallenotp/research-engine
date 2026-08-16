"""Swappable proxy backends for the stealth fetcher."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import importlib.util
import itertools
import logging
import os
from pathlib import Path
import secrets
import time
from urllib.parse import quote

import httpx
from research_engine.apify_accounts import AccountPool, load_accounts_from_env

from . import paths


_SCRAPERAPI_PROXY_USERNAME = "scraperapi"
_SCRAPERAPI_PROXY_MODULE = None
_SCRAPERAPI_PROXY_MODULE_PATH = paths.optional_path(paths.APIFY_PROXY_MODULE_ENV) or paths.home_path(
    "apify-proxy-app",
    "apify-proxy.py",
)
_NORDVPN_CREDENTIALS_PATH = paths.optional_path(paths.NORDVPN_ENV_FILE_ENV) or paths.home_path(
    ".secrets",
    "nordvpn.env",
)
_NORDVPN_SERVERS_URL = "https://api.nordvpn.com/v1/servers"
_NORDVPN_SOCKS_PORT = 1080
_NORDVPN_BAD_TTL_SECONDS = 300
_NORDVPN_FALLBACK_HOSTS = [
    "socks-nl1.nordvpn.com",
    "socks-nl2.nordvpn.com",
]
logger = logging.getLogger("fetch_proxy")
FIRECRAWL_ENV_VARS = tuple(f"FIRECRAWL_API_KEY_{idx}" for idx in range(1, 9))


@dataclass(frozen=True)
class ProxySession:
    proxy_url: str | None
    label: str
    sticky: bool = False
    account_id: str | None = None


class ProxyBackend:
    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        raise NotImplementedError

    def release(self, session: ProxySession, *, ok: bool) -> None:
        return None


class NoProxyBackend(ProxyBackend):
    def __init__(self, label: str = "no_proxy"):
        self._label = label

    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        return ProxySession(proxy_url=None, label=self._label, sticky=sticky)


@functools.lru_cache(maxsize=None)
def env_file_values(path: Path | None = None) -> dict[str, str]:
    """Parse the engine's env file. Cached. NEVER writes to os.environ."""
    path = path or paths.env_file()
    try:
        if path is None or not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = value
    return values


def env_value(name: str) -> str:
    """os.environ wins; fall back to the env file. Read-only."""
    return (os.environ.get(name) or env_file_values().get(name) or "").strip()


def load_firecrawl_keys_from_env(max_keys: int = 8) -> list[tuple[str, str]]:
    keys = [
        (env_var, value)
        for env_var in FIRECRAWL_ENV_VARS[:max_keys]
        if (value := env_value(env_var))
    ]
    if not keys:
        logger.warning(
            "firecrawl key rotation loaded %s/%s keys; checked %s",
            len(keys),
            max_keys,
            ", ".join(FIRECRAWL_ENV_VARS[:max_keys]),
        )
    else:
        logger.info("firecrawl key rotation loaded %s keys", len(keys))
    return keys


class ApifyResidentialBackend(ProxyBackend):
    def __init__(self, pool: AccountPool):
        self._pool = pool

    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        account = self._pool.next_account()
        if sticky:
            session_id = f"session-{secrets.token_hex(4)}"
            username = f"groups-RESIDENTIAL,{session_id}"
            label = f"apify:{account.id}:{session_id}"
        else:
            username = "groups-RESIDENTIAL"
            label = f"apify:{account.id}:rotate"

        proxy_url = (
            f"http://{username}:{account.proxy_pass}@proxy.apify.com:8000"
        )
        return ProxySession(
            proxy_url=proxy_url,
            label=label,
            sticky=sticky,
            account_id=account.id,
        )


class OwnProxyBackend(ProxyBackend):
    def __init__(self, proxies: list[str]):
        if not proxies:
            raise ValueError("OwnProxyBackend needs at least one proxy URL")
        self._cycle = itertools.cycle(proxies)

    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        proxy_url = next(self._cycle)
        return ProxySession(
            proxy_url=proxy_url,
            label=f"own:{proxy_url.split('@')[-1]}",
            sticky=sticky,
        )


def _load_nordvpn_credentials(path: Path | None = None) -> tuple[str, str]:
    path = path or _NORDVPN_CREDENTIALS_PATH
    if not path.exists():
        msg = f"NordVPN credentials file missing: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    user = values.get("NORDVPN_SERVICE_USER", "")
    password = values.get("NORDVPN_SERVICE_PASS", "")
    missing = [
        key
        for key, value in (
            ("NORDVPN_SERVICE_USER", user),
            ("NORDVPN_SERVICE_PASS", password),
        )
        if not value
    ]
    if missing:
        msg = f"NordVPN credentials missing in {path}: {', '.join(missing)}"
        logger.error(msg)
        raise RuntimeError(msg)
    return user, password


def _fetch_nordvpn_socks_hosts() -> list[str]:
    response = httpx.get(
        _NORDVPN_SERVERS_URL,
        params={"filters[servers_technologies][identifier]": "socks", "limit": 100},
        timeout=10,
    )
    response.raise_for_status()
    hosts = [
        str(server["hostname"])
        for server in response.json()
        if server.get("status") == "online" and server.get("hostname")
    ]
    if not hosts:
        raise RuntimeError("NordVPN API returned no online SOCKS5 hosts")
    return hosts


class NordVPNSocksBackend(ProxyBackend):
    def __init__(self, credentials_path: Path | None = None):
        self._user, self._password = _load_nordvpn_credentials(credentials_path)
        try:
            self._hosts = _fetch_nordvpn_socks_hosts()
        except httpx.HTTPError as exc:
            logger.warning("NordVPN API unreachable, using fallback SOCKS5 hosts: %s", exc)
            self._hosts = list(_NORDVPN_FALLBACK_HOSTS)
        self._next = 0
        self._bad_until: dict[str, float] = {}

    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        del domain
        now = time.monotonic()
        for _ in range(len(self._hosts)):
            host = self._hosts[self._next % len(self._hosts)]
            self._next += 1
            if self._bad_until.get(host, 0) > now:
                continue
            user = quote(self._user, safe="")
            password = quote(self._password, safe="")
            return ProxySession(
                proxy_url=f"socks5://{user}:{password}@{host}:{_NORDVPN_SOCKS_PORT}",
                label=host,
                sticky=sticky,
            )
        raise RuntimeError("no NordVPN SOCKS5 servers available")

    def release(self, session: ProxySession, *, ok: bool) -> None:
        if not ok and session.label:
            self._bad_until[session.label] = time.monotonic() + _NORDVPN_BAD_TTL_SECONDS


def _load_scraperapi_proxy_module():
    global _SCRAPERAPI_PROXY_MODULE
    if _SCRAPERAPI_PROXY_MODULE is not None:
        return _SCRAPERAPI_PROXY_MODULE

    proxy_module_path = _SCRAPERAPI_PROXY_MODULE_PATH
    if not proxy_module_path.exists():
        raise FileNotFoundError(
            f"ScraperAPI proxy module missing: {proxy_module_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "apify_proxy_runtime",
        proxy_module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"unable to load ScraperAPI proxy module at {proxy_module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ScraperAPI proxy module missing: {proxy_module_path}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"unable to load ScraperAPI proxy module at {proxy_module_path}: {exc}"
        ) from exc
    _SCRAPERAPI_PROXY_MODULE = module
    return module


class _ScraperAPIProxyRotator:
    def __init__(self, module):
        self._module = module

    def reserve_key(self) -> dict[str, object]:
        selected = self._module._reserve_next_key(self._module.SCRAPERAPI, attempted=set())
        if not selected:
            raise RuntimeError("no live ScraperAPI keys available")
        self._module._record_attempt(
            self._module.SCRAPERAPI,
            int(selected["index"]),
            str(selected["masked"]),
            None,
        )
        return dict(selected)


def _get_scraperapi_rotator() -> _ScraperAPIProxyRotator:
    return _ScraperAPIProxyRotator(_load_scraperapi_proxy_module())


def _load_scraperapi_backend(*, fail_open: bool) -> ProxyBackend:
    try:
        return ScraperAPIBackend()
    except Exception as exc:
        msg = f"scraperapi backend unavailable: {exc}"
        if fail_open:
            logger.warning("%s", msg)
            return NoProxyBackend()
        raise RuntimeError(msg) from exc


class ScraperAPIBackend(ProxyBackend):
    def __init__(self, rotator: _ScraperAPIProxyRotator | None = None):
        self._rotator = rotator or _get_scraperapi_rotator()

    def acquire(self, *, domain: str, sticky: bool) -> ProxySession:
        del domain
        try:
            key = self._rotator.reserve_key()
        except Exception as exc:
            logger.warning("scraperapi proxy unavailable, falling back to direct fetch: %s", exc)
            return ProxySession(proxy_url=None, label="scraperapi_unavailable", sticky=sticky)
        proxy_url = f"{_SCRAPERAPI_PROXY_USERNAME}:{key['value']}@proxy-server.scraperapi.com:8001"
        return ProxySession(
            proxy_url=f"http://{proxy_url}",
            label=f"scraperapi:{key['env_var']}",
            sticky=sticky,
        )


def load_proxy_backend(config: dict) -> ProxyBackend:
    choice = ((config.get("proxy") or {}) if isinstance(config, dict) else {}).get(
        "backend", "none"
    )
    if choice == "apify":
        accounts = [account for account in load_accounts_from_env() if account.proxy_pass]
        if not accounts:
            return NoProxyBackend()
        return ApifyResidentialBackend(AccountPool(accounts))
    if choice == "own":
        proxies = [item for item in os.environ.get("OWN_PROXIES", "").split(",") if item]
        return OwnProxyBackend(proxies) if proxies else NoProxyBackend()
    if choice == "scraperapi":
        return _load_scraperapi_backend(fail_open=False)
    if choice == "nordvpn":
        try:
            return NordVPNSocksBackend()
        except Exception as exc:
            logger.warning("nordvpn backend unavailable, trying scraperapi backend: %s", exc)
        try:
            return ScraperAPIBackend()
        except Exception as scraper_exc:
            logger.warning(
                "nordvpn backend unavailable, no proxy will be used after scraperapi fallback failed: %s",
                scraper_exc,
            )
            return NoProxyBackend("no_proxy:nordvpn_and_scraperapi_unavailable")
    return NoProxyBackend()
