from __future__ import annotations

import json

import pytest

from research_engine import extractor
from research_engine import fetch_proxy
from research_engine import wayback_fallback
from research_engine.politeness import DomainCooldown
from research_engine.tools import reader_ladder_check


def _good_payload() -> dict[str, str | None]:
    return extractor._payload("Example", "reader body " * 30)


@pytest.fixture(autouse=True)
def _isolated_reader_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_ENGINE_DATA_DIR", str(tmp_path))
    extractor._rung_availability.cache_clear()
    yield
    extractor._rung_availability.cache_clear()


def _telemetry(tmp_path):
    path = tmp_path / "research-reader-telemetry.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rung_without_precondition_is_skipped_not_called(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setattr(fetch_proxy, "env_file_values", lambda: {})
    monkeypatch.setattr(extractor, "_cloudflare_markdown_preflight", lambda _url: None)
    monkeypatch.setattr(extractor, "_trafilatura", lambda *_args: _good_payload())
    monkeypatch.setattr(extractor, "_jina", lambda _url: (_ for _ in ()).throw(AssertionError("must not be called")))

    extractor.extract_clean_text("https://example.com/a", seen_urls_path=tmp_path / "seen.txt")

    rows = _telemetry(tmp_path)
    assert any(row["method"] == "jina" and row["status"] == "skipped" and row["reason"] == "no JINA_API_KEY" for row in rows)
    assert not any(row["method"] == "jina" and row["status"] == "attempt" for row in rows)


def test_crawl4ai_skipped_when_script_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractor, "CRAWL4AI_SCRIPT", tmp_path / "missing.py")
    monkeypatch.setattr(extractor, "_cloudflare_markdown_preflight", lambda _url: None)
    monkeypatch.setattr(extractor, "_trafilatura", lambda *_args: {})
    monkeypatch.setattr(extractor, "_crawl4ai", lambda _url: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setattr(fetch_proxy, "env_file_values", lambda: {})
    monkeypatch.delenv("WAYBACK_ACCESS_KEY", raising=False)
    monkeypatch.delenv("WAYBACK_SECRET_KEY", raising=False)
    monkeypatch.setattr(wayback_fallback, "_ENV_FILE_VALUES", {})

    extractor.extract_clean_text("https://example.com/a", seen_urls_path=tmp_path / "seen.txt")

    assert any(row["method"] == "crawl4ai" and row["status"] == "skipped" for row in _telemetry(tmp_path))


def test_wayback_skipped_without_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WAYBACK_ACCESS_KEY", raising=False)
    monkeypatch.delenv("WAYBACK_SECRET_KEY", raising=False)
    monkeypatch.setattr(extractor, "_cloudflare_markdown_preflight", lambda _url: None)
    monkeypatch.setattr(extractor, "_trafilatura", lambda *_args: {})
    monkeypatch.setattr(wayback_fallback, "_ENV_FILE_VALUES", {})
    monkeypatch.setattr(wayback_fallback, "try_wayback", lambda _url: (_ for _ in ()).throw(AssertionError("must not be called")))

    extractor.extract_clean_text("https://example.com/a", seen_urls_path=tmp_path / "seen.txt")

    assert any(row["method"] == "wayback" and row["status"] == "skipped" and row["reason"] == "no WAYBACK keys" for row in _telemetry(tmp_path))


def test_origin_cooldown_stops_ladder_before_firecrawl(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("FIRECRAWL_API_KEY_1", "test-key")
    monkeypatch.setattr(extractor, "_cloudflare_markdown_preflight", lambda _url: None)
    monkeypatch.setattr(
        extractor,
        "_trafilatura",
        lambda *_args: (_ for _ in ()).throw(DomainCooldown("example.com", 60)),
    )
    monkeypatch.setattr(
        extractor,
        "_firecrawl",
        lambda _url: calls.append("firecrawl") or _good_payload(),
    )

    result = extractor.extract_clean_text("https://example.com/a", seen_urls_path=tmp_path / "seen.txt")

    assert result is None
    assert calls == []
    rows = _telemetry(tmp_path)
    cooldown = [row for row in rows if row["status"] == "domain_cooldown"]
    assert len(cooldown) == 1
    assert cooldown[0]["reason"].startswith("example.com ")
    assert any(
        row["method"] == "trafilatura" and row["status"] == "domain_cooldown"
        for row in rows
    )


def test_extract_clean_text_returns_none_for_gitingest_cooldown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractor, "_is_document_source", lambda *_args: False)
    monkeypatch.setattr(extractor, "_extract_pdf_or_document", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(extractor, "_is_github_repo_url", lambda _url: True)
    monkeypatch.setattr(
        extractor,
        "_gitingest",
        lambda *_args: (_ for _ in ()).throw(DomainCooldown("github.com", 60)),
    )

    result = extractor.extract_clean_text(
        "https://github.com/example/repo",
        seen_urls_path=tmp_path / "seen.txt",
    )

    assert result is None
    cooldown = [row for row in _telemetry(tmp_path) if row["status"] == "domain_cooldown"]
    assert len(cooldown) == 1
    assert cooldown[0]["reason"].startswith("github.com ")


def test_rung_order_is_by_measured_yield() -> None:
    assert [rung.name for rung in extractor.WEB_RUNGS] == ["cloudflare-markdown", "trafilatura", "firecrawl", "crawl4ai", "scrapling", "crawlee", "jina"]


def test_precondition_evaluated_once_per_process(tmp_path, monkeypatch) -> None:
    calls = 0

    def available() -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return False, "test unavailable"

    rung = next(rung for rung in extractor.WEB_RUNGS if rung.name == "jina")
    monkeypatch.setattr(
        extractor,
        "WEB_RUNGS",
        tuple(
            extractor.Rung(rung.name, rung.fn, available, rung.carries_fetch_meta)
            if item.name == "jina"
            else item
            for item in extractor.WEB_RUNGS
        ),
    )
    monkeypatch.setattr(extractor, "_cloudflare_markdown_preflight", lambda _url: None)
    monkeypatch.setattr(extractor, "_trafilatura", lambda *_args: _good_payload())

    extractor.extract_clean_text("https://example.com/a", seen_urls_path=tmp_path / "one.txt")
    extractor.extract_clean_text("https://example.com/b", seen_urls_path=tmp_path / "two.txt")

    assert calls == 1


def test_reader_ladder_check_prints_availability(capsys) -> None:
    reader_ladder_check.main([])
    output = capsys.readouterr().out
    for rung in extractor.WEB_RUNGS:
        assert f"{rung.name}: " in output
    assert "wayback: " in output
    assert "agent_browser (T3 only, gated)" in output
