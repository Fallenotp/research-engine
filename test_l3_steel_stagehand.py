from pathlib import Path
from unittest.mock import patch

from research_engine import extractor
from research_engine.schema import ExtractionMethod


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_steel_stagehand_preflight_blocked(monkeypatch) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_URL", "http://127.0.0.1:7799/extract")

    with patch.object(extractor.l3_guard, "preflight", return_value=(False, "L3_BLOCKED_LOW_RAM")), patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.requests,
        "post",
    ) as post_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        out = extractor._steel_stagehand("https://x")

    assert out is None
    post_mock.assert_not_called()
    record_failure_mock.assert_not_called()
    politeness_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="blocked",
        reason="L3_BLOCKED_LOW_RAM",
        rung="steel_stagehand",
    )


def test_steel_stagehand_returns_payload_and_records_success(monkeypatch) -> None:
    calls = []

    monkeypatch.setenv("WEBREAD_SIDECAR_URL", "http://127.0.0.1:7799/extract")
    monkeypatch.setenv("WEBREAD_L3_TIMEOUT_S", "45")

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(
            {
                "title": "T",
                "text": "hello world body " * 20,
            }
        )

    monkeypatch.setattr(extractor.requests, "post", fake_post)

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_stagehand(
            "https://example.com",
            instruction="return the page title",
        )

    assert out is not None
    assert out["title"] == "T"
    assert "hello world body" in (out.get("text") or "")
    assert calls[0][0] == "http://127.0.0.1:7799/extract"
    assert calls[0][1]["json"] == {
        "url": "https://example.com",
        "instruction": "return the page title",
    }
    assert calls[0][1]["timeout"] == 45
    record_success_mock.assert_called_once_with()
    record_failure_mock.assert_not_called()
    log_attempt_mock.assert_not_called()


def test_steel_stagehand_failure_records_failure_and_reaps_container(monkeypatch) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_URL", "http://127.0.0.1:7799/extract")

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.requests,
        "post",
        side_effect=RuntimeError("boom"),
    ), patch.object(
        extractor.subprocess,
        "run",
    ) as subprocess_run_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_stagehand("https://x")

    assert out is None
    record_failure_mock.assert_called_once_with()
    record_success_mock.assert_not_called()
    subprocess_run_mock.assert_called_once_with(
        ["docker", "kill", "webread-steel"],
        capture_output=True,
        timeout=10,
    )
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="failed",
        reason="boom",
        rung="steel_stagehand",
    )


def test_steel_stagehand_error_response_records_failure(monkeypatch) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_URL", "http://127.0.0.1:7799/extract")

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.requests,
        "post",
        return_value=_FakeResponse({"error": "x"}),
    ), patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_stagehand("https://x")

    assert out is None
    record_failure_mock.assert_called_once_with()
    record_success_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="failed",
        reason="sidecar error/invalid response",
        rung="steel_stagehand",
    )


def test_steel_stagehand_empty_text_records_failure(monkeypatch) -> None:
    monkeypatch.setenv("WEBREAD_SIDECAR_URL", "http://127.0.0.1:7799/extract")

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.requests,
        "post",
        return_value=_FakeResponse({"text": ""}),
    ), patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_stagehand("https://x")

    assert out is None
    record_failure_mock.assert_called_once_with()
    record_success_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="failed",
        reason="empty body",
        rung="steel_stagehand",
    )


def test_steel_scrape_preflight_blocked(monkeypatch) -> None:
    monkeypatch.setenv("STEEL_SCRAPE_URL", "http://127.0.0.1:3000/v1/scrape")

    with patch.object(extractor.l3_guard, "preflight", return_value=(False, "L3_BLOCKED_LOW_RAM")), patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.requests,
        "post",
    ) as post_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        out = extractor._steel_scrape("https://x")

    assert out is None
    post_mock.assert_not_called()
    record_failure_mock.assert_not_called()
    politeness_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="blocked",
        reason="L3_BLOCKED_LOW_RAM",
        rung="steel_scrape",
    )


def test_steel_scrape_returns_payload_and_records_success(monkeypatch) -> None:
    calls = []

    monkeypatch.setenv("STEEL_SCRAPE_URL", "http://127.0.0.1:3000/v1/scrape")
    monkeypatch.setenv("WEBREAD_L3_TIMEOUT_S", "45")

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(
            {
                "content": {"markdown": "# Hi\n\nbody text here", "html": "<p>ignored</p>"},
                "metadata": {"title": "T"},
            }
        )

    monkeypatch.setattr(extractor.requests, "post", fake_post)

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_scrape("https://example.com")

    assert out is not None
    assert out["title"] == "T"
    assert "body text here" in (out.get("text") or "")
    assert calls[0][0] == "http://127.0.0.1:3000/v1/scrape"
    assert calls[0][1]["json"] == {
        "url": "https://example.com",
        "format": ["markdown", "html"],
    }
    assert calls[0][1]["timeout"] == 45
    record_success_mock.assert_called_once_with()
    record_failure_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://example.com",
        decision="success",
        rung="steel_scrape",
    )


def test_steel_scrape_failure_records_failure_and_reaps_container(monkeypatch) -> None:
    monkeypatch.setenv("STEEL_SCRAPE_URL", "http://127.0.0.1:3000/v1/scrape")

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.requests,
        "post",
        side_effect=RuntimeError("boom"),
    ), patch.object(
        extractor.subprocess,
        "run",
    ) as subprocess_run_mock, patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_scrape("https://x")

    assert out is None
    record_failure_mock.assert_called_once_with()
    record_success_mock.assert_not_called()
    subprocess_run_mock.assert_called_once_with(
        ["docker", "kill", "webread-steel"],
        capture_output=True,
        timeout=10,
    )
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="failed",
        reason="boom",
        rung="steel_scrape",
    )


def test_steel_scrape_empty_content_records_failure(monkeypatch) -> None:
    monkeypatch.setenv("STEEL_SCRAPE_URL", "http://127.0.0.1:3000/v1/scrape")

    with patch.object(extractor.l3_guard, "preflight", return_value=(True, None)), patch.object(
        extractor.l3_guard,
        "record_failure",
    ) as record_failure_mock, patch.object(
        extractor.l3_guard,
        "record_success",
    ) as record_success_mock, patch.object(
        extractor.l3_guard,
        "log_attempt",
    ) as log_attempt_mock, patch.object(
        extractor.requests,
        "post",
        return_value=_FakeResponse({"content": {"markdown": "", "html": ""}}),
    ), patch.object(extractor, "_get_politeness") as politeness_mock:
        politeness = politeness_mock.return_value
        politeness.allowed.return_value = True
        out = extractor._steel_scrape("https://x")

    assert out is None
    record_failure_mock.assert_called_once_with()
    record_success_mock.assert_not_called()
    log_attempt_mock.assert_called_once_with(
        url="https://x",
        decision="failed",
        reason="empty scrape",
        rung="steel_scrape",
    )


def test_l3_not_called_when_scrapling_succeeds(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setenv("WEBREAD_ALLOW_L3", "1")
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
    ), patch.object(extractor, "_steel_scrape") as l3_mock, patch.object(
        extractor,
        "_firecrawl",
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.SCRAPLING.value
    l3_mock.assert_not_called()
    firecrawl_mock.assert_not_called()


def test_l3_called_when_earlier_rungs_fail_and_gate_enabled(monkeypatch, tmp_path) -> None:
    url = "https://protected.example/article"
    l3_payload = {
        "title": "Example Domain",
        "text": "steel stagehand recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    monkeypatch.setenv("WEBREAD_ALLOW_L3", "1")
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
        return_value={},
    ), patch.object(
        extractor,
        "_steel_scrape",
        return_value=l3_payload,
    ) as l3_mock, patch.object(
        extractor,
        "_firecrawl",
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.STEEL_STAGEHAND.value
    assert "steel stagehand recovered" in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    l3_mock.assert_called_once_with(url)
    firecrawl_mock.assert_not_called()


def test_l3_instruction_path_stays_on_stagehand(monkeypatch, tmp_path) -> None:
    url = "https://protected.example/article"
    l3_payload = {
        "title": "Example Domain",
        "text": "steel stagehand recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    monkeypatch.setenv("WEBREAD_ALLOW_L3", "1")
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
        return_value={},
    ), patch.object(
        extractor,
        "_steel_stagehand",
        return_value=l3_payload,
    ) as stagehand_mock, patch.object(
        extractor,
        "_steel_scrape",
    ) as scrape_mock, patch.object(
        extractor,
        "_firecrawl",
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(
            url,
            seen_urls_path=tmp_path / "seen.txt",
            instruction="return the page title",
        )

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.STEEL_STAGEHAND.value
    stagehand_mock.assert_called_once_with(url, instruction="return the page title")
    scrape_mock.assert_not_called()
    firecrawl_mock.assert_not_called()


def test_l3_skipped_when_gate_disabled(monkeypatch, tmp_path) -> None:
    url = "https://protected.example/article"
    firecrawl_payload = {
        "title": "T",
        "text": "firecrawl recovered this body text " * 8,
        "author": None,
        "published_date": None,
    }

    monkeypatch.delenv("WEBREAD_ALLOW_L3", raising=False)
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
        return_value={},
    ), patch.object(extractor, "_steel_scrape") as l3_mock, patch.object(
        extractor,
        "_firecrawl",
        return_value=firecrawl_payload,
    ) as firecrawl_mock:
        out = extractor.extract_clean_text(url, seen_urls_path=tmp_path / "seen.txt")

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.FIRECRAWL.value
    l3_mock.assert_not_called()
    firecrawl_mock.assert_called_once_with(url)
