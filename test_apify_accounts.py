import os
from unittest.mock import patch

import pytest

from research_engine.apify_accounts import (
    CAP_USD,
    AccountPool,
    AllAccountsExhausted,
    ApifyAccount,
    actor_route_for_url,
    _read_secret_files,
    load_accounts_from_env,
)


def _accounts() -> list[ApifyAccount]:
    return [ApifyAccount(id=f"a{i}", token=f"t{i}", proxy_pass=f"p{i}") for i in range(3)]


def test_round_robin_picks_least_used_under_cap() -> None:
    pool = AccountPool(_accounts())

    with patch.object(pool, "_usage_usd", return_value=0.0):
        first = pool.next_account()
        second = pool.next_account()

    assert first.id != second.id


def test_skips_account_over_cap() -> None:
    pool = AccountPool(_accounts())
    usage = {"a0": CAP_USD + 0.01, "a1": 0.0, "a2": 0.0}

    with patch.object(pool, "_usage_usd", side_effect=lambda acct: usage[acct.id]):
        account = pool.next_account()

    assert account.id in {"a1", "a2"}


def test_all_capped_raises() -> None:
    pool = AccountPool(_accounts())

    with patch.object(pool, "_usage_usd", return_value=CAP_USD + 1):
        with pytest.raises(AllAccountsExhausted):
            pool.next_account()


def test_load_accounts_uses_up_to_twelve_numbered_tokens(monkeypatch, caplog) -> None:
    monkeypatch.setattr("research_engine.apify_accounts._read_secret_files", lambda: {})
    monkeypatch.delenv("APIFY_ACCOUNTS", raising=False)
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    for idx in range(1, 14):
        monkeypatch.delenv(f"APIFY_ACCOUNT_ID_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASS_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASSWORD_{idx}", raising=False)
        monkeypatch.setenv(f"APIFY_TOKEN_{idx}", f"test-token-{idx}")

    accounts = load_accounts_from_env()

    assert [account.id for account in accounts] == [f"apify_{idx}" for idx in range(1, 13)]
    assert "apify account rotation loaded" not in caplog.text


def test_load_accounts_warns_when_fewer_than_twelve(monkeypatch, caplog) -> None:
    monkeypatch.setattr("research_engine.apify_accounts._read_secret_files", lambda: {})
    monkeypatch.delenv("APIFY_ACCOUNTS", raising=False)
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    for idx in range(1, 13):
        monkeypatch.delenv(f"APIFY_TOKEN_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_ACCOUNT_ID_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASS_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASSWORD_{idx}", raising=False)
    monkeypatch.setenv("APIFY_TOKEN_1", "test-token-1")

    accounts = load_accounts_from_env()

    assert len(accounts) == 1
    assert "apify account rotation loaded 1/12 accounts" in caplog.text


def test_load_accounts_does_not_mutate_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "research_engine.apify_accounts._read_secret_files",
        lambda: {"APIFY_TOKEN_1": "file-token-1"},
    )
    monkeypatch.delenv("APIFY_ACCOUNTS", raising=False)
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    monkeypatch.delenv("APIFY_ACCOUNT_ID", raising=False)
    for idx in range(1, 14):
        monkeypatch.delenv(f"APIFY_TOKEN_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_KEY_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_ACCOUNT_ID_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASS_{idx}", raising=False)
        monkeypatch.delenv(f"APIFY_PROXY_PASSWORD_{idx}", raising=False)
    baseline = dict(os.environ)

    accounts = load_accounts_from_env()

    assert [account.token for account in accounts] == ["file-token-1"]
    assert dict(os.environ) == baseline


def test_read_secret_files_skips_missing_entries(tmp_path) -> None:
    present = tmp_path / "apify-present.env"
    present.write_text("APIFY_TOKEN_1=token-one\n", encoding="utf-8")
    missing = tmp_path / "missing.env"

    values = _read_secret_files((missing, present))

    assert values == {"APIFY_TOKEN_1": "token-one"}


def test_social_actor_routes_cover_instagram_and_tiktok(monkeypatch) -> None:
    monkeypatch.delenv("APIFY_INSTAGRAM_ACTOR", raising=False)
    monkeypatch.delenv("APIFY_TIKTOK_ACTOR", raising=False)

    instagram = actor_route_for_url("https://m.instagram.com/p/example/")
    tiktok = actor_route_for_url("https://vm.tiktok.com/example/")

    assert instagram is not None
    assert instagram.platform == "instagram"
    assert instagram.runs_path == "/v2/acts/apify~instagram-scraper/runs"
    assert tiktok is not None
    assert tiktok.platform == "tiktok"
    assert tiktok.runs_path == "/v2/acts/clockworks~tiktok-scraper/runs"
