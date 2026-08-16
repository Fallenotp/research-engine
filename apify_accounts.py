"""Round-robin helpers for shared Apify accounts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import logging
from urllib.parse import urlparse

import requests

from . import paths


CAP_USD = 4.90
USAGE_TTL_S = 300
MAX_ACCOUNTS = 12
_API = "https://api.apify.com/v2/users/me"
logger = logging.getLogger("apify_accounts")


class AllAccountsExhausted(RuntimeError):
    """Raised when every configured account is over the cap."""


@dataclass
class ApifyAccount:
    id: str
    token: str
    proxy_pass: str = ""
    last_used: float = 0.0
    _usage: float = field(default=0.0)
    _usage_at: float = field(default=0.0)


@dataclass(frozen=True)
class ApifyActorRoute:
    platform: str
    actor_id: str
    env_var: str

    @property
    def runs_path(self) -> str:
        return f"/v2/acts/{self.actor_id.replace('/', '~')}/runs"


def _default_secret_files() -> tuple[Path, ...]:
    configured = paths.env_file()
    if configured is not None:
        return (configured,)
    return (
        paths.home_path(".secrets", "apify.env"),
        paths.home_path(".secrets", "apify-keys.env"),
        paths.home_path(".openclaw", ".env"),
        paths.home_path("consequence-tracker", ".env"),
    )


def _read_secret_files(paths_to_read: tuple[Path, ...] | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    paths_to_read = paths_to_read or _default_secret_files()
    for path in paths_to_read:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values.setdefault(key.strip(), value.strip().strip("\"'"))
        except FileNotFoundError:
            continue
    return values


def _merged_values() -> dict[str, str]:
    values = _read_secret_files()
    values.update(os.environ)
    return values


def _append_account(
    accounts: list[ApifyAccount],
    *,
    account_id: str,
    token: str,
    proxy_pass: str = "",
) -> None:
    if not token or any(account.token == token for account in accounts):
        return
    accounts.append(ApifyAccount(id=account_id, token=token, proxy_pass=proxy_pass))


def load_accounts_from_env(max_accounts: int = MAX_ACCOUNTS) -> list[ApifyAccount]:
    values = _merged_values()
    raw = values.get("APIFY_ACCOUNTS", "").strip()
    accounts: list[ApifyAccount] = []

    for chunk in (part for part in raw.split(",") if part):
        parts = chunk.split(":", 2)
        if len(parts) < 2:
            continue
        proxy_pass = parts[2] if len(parts) == 3 else ""
        _append_account(
            accounts,
            account_id=parts[0],
            token=parts[1],
            proxy_pass=proxy_pass,
        )
        if len(accounts) >= max_accounts:
            break

    for idx in range(1, max_accounts + 1):
        if len(accounts) >= max_accounts:
            break
        token = values.get(f"APIFY_TOKEN_{idx}", "") or values.get(
            f"APIFY_KEY_{idx}",
            "",
        )
        proxy_pass = values.get(f"APIFY_PROXY_PASS_{idx}", "") or values.get(
            f"APIFY_PROXY_PASSWORD_{idx}",
            "",
        )
        _append_account(
            accounts,
            account_id=values.get(f"APIFY_ACCOUNT_ID_{idx}", f"apify_{idx}"),
            token=token,
            proxy_pass=proxy_pass,
        )

    if len(accounts) < max_accounts:
        _append_account(
            accounts,
            account_id=values.get("APIFY_ACCOUNT_ID", "apify_api_key"),
            token=values.get("APIFY_API_KEY", ""),
        )

    if len(accounts) < max_accounts:
        logger.warning(
            "apify account rotation loaded %s/%s accounts; checked APIFY_ACCOUNTS, "
            "APIFY_TOKEN_1..%s or APIFY_KEY_1..%s, APIFY_API_KEY and %s",
            len(accounts),
            max_accounts,
            max_accounts,
            max_accounts,
            paths.ENV_FILE_ENV,
        )

    return accounts


def social_platform_for_url(source_url: str) -> str | None:
    host = urlparse(source_url).netloc.lower().split(":", 1)[0]
    if host == "instagram.com" or host.endswith(".instagram.com"):
        return "instagram"
    if host == "tiktok.com" or host.endswith(".tiktok.com"):
        return "tiktok"
    return None


def actor_route_for_url(source_url: str) -> ApifyActorRoute | None:
    platform = social_platform_for_url(source_url)
    if platform == "instagram":
        env_var = "APIFY_INSTAGRAM_ACTOR"
        return ApifyActorRoute(
            platform=platform,
            actor_id=os.environ.get(env_var, "apify/instagram-scraper"),
            env_var=env_var,
        )
    if platform == "tiktok":
        env_var = "APIFY_TIKTOK_ACTOR"
        return ApifyActorRoute(
            platform=platform,
            actor_id=os.environ.get(env_var, "clockworks/tiktok-scraper"),
            env_var=env_var,
        )
    return None


class AccountPool:
    def __init__(self, accounts: list[ApifyAccount]):
        if not accounts:
            raise ValueError("AccountPool needs at least one account")
        self._accounts = accounts

    def _usage_usd(self, account: ApifyAccount) -> float:
        now = time.monotonic()
        if now - account._usage_at < USAGE_TTL_S:
            return account._usage

        response = requests.get(
            _API,
            headers={"Authorization": f"Bearer {account.token}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json().get("data", {})
        usage = float(data.get("monthlyUsageCycleUsdSpend", 0.0) or 0.0)
        account._usage = usage
        account._usage_at = now
        return usage

    def next_account(self) -> ApifyAccount:
        for account in sorted(self._accounts, key=lambda item: item.last_used):
            try:
                if self._usage_usd(account) < CAP_USD:
                    account.last_used = time.monotonic()
                    return account
            except requests.RequestException:
                continue

        raise AllAccountsExhausted(
            f"all {len(self._accounts)} Apify accounts at/over ${CAP_USD}"
        )
