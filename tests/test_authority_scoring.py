from pathlib import Path

import pytest

import research_engine.apify_accounts as apify_accounts
import research_engine.router as router_module
from research_engine.grounding import _source_record_from_extract
from research_engine.research_cli import source_record


@pytest.fixture
def extracted_source(tmp_path: Path) -> dict[str, object]:
    raw_text_path = tmp_path / "source.txt"
    raw_text_path.write_text("verified source text", encoding="utf-8")
    return {
        "url": "https://courtlistener.com/opinion/123/example/",
        "domain": "courtlistener.com",
        "title": "Example opinion",
        "extraction_method": "curl",
        "raw_text_path": str(raw_text_path),
        "char_count": 20,
    }


@pytest.mark.parametrize("builder", [_source_record_from_extract, source_record])
def test_source_without_topic_keeps_flat_default(builder, extracted_source) -> None:
    record = builder(extracted_source, "verified source text", topic=None)

    assert record.topic_authority_score == 0.6


@pytest.mark.parametrize("builder", [_source_record_from_extract, source_record])
def test_known_authoritative_domain_uses_table_value(builder, extracted_source) -> None:
    record = builder(extracted_source, "verified source text", topic="legal")

    assert record.topic_authority_score == 1.0


@pytest.mark.parametrize("builder", [_source_record_from_extract, source_record])
def test_unknown_domain_ranks_below_old_flat_default(builder, extracted_source) -> None:
    extracted_source["domain"] = "gokufashionoutfitstyle.com"

    record = builder(extracted_source, "verified source text", topic="legal")

    assert record.topic_authority_score == 0.5
    assert record.topic_authority_score < 0.6


@pytest.mark.parametrize("builder", [_source_record_from_extract, source_record])
def test_unknown_topic_uses_router_default_without_raising(builder, extracted_source) -> None:
    record = builder(extracted_source, "verified source text", topic="gov_tech")

    assert record.topic_authority_score == 0.5


def test_router_load_failure_falls_back_and_is_attempted_once(monkeypatch) -> None:
    attempts = 0

    def fail_to_load():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("broken router config")

    monkeypatch.setattr(router_module, "_DEFAULT_ROUTER", None)
    monkeypatch.setattr(router_module, "_DEFAULT_ROUTER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(router_module, "load_router", fail_to_load)

    assert router_module.source_authority_score("example.com", "legal") == 0.6
    assert router_module.source_authority_score("example.com", "legal") == 0.6
    assert attempts == 1


def test_real_secret_files_include_apify_key_aliases_and_skip_missing_file() -> None:
    values = apify_accounts._read_secret_files()
    assert any(name.startswith("APIFY_KEY_") for name in values)
    assert apify_accounts._read_secret_files((Path("/definitely/missing/apify.env"),)) == {}


def test_numbered_apify_key_alias_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        apify_accounts,
        "_read_secret_files",
        lambda: {"APIFY_KEY_12": "test-alias-token"},
    )
    for name in list(apify_accounts.os.environ):
        if name.startswith("APIFY_"):
            monkeypatch.delenv(name, raising=False)

    accounts = apify_accounts.load_accounts_from_env()

    assert [account.id for account in accounts] == ["apify_12"]


def test_numbered_apify_token_wins_over_key_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        apify_accounts,
        "_read_secret_files",
        lambda: {
            "APIFY_TOKEN_11": "preferred-token",
            "APIFY_KEY_11": "alias-token",
        },
    )
    for name in list(apify_accounts.os.environ):
        if name.startswith("APIFY_"):
            monkeypatch.delenv(name, raising=False)

    accounts = apify_accounts.load_accounts_from_env()

    assert len(accounts) == 1
    assert accounts[0].token == "preferred-token"
