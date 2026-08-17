import inspect
import json
import threading
from unittest.mock import create_autospec
from urllib.parse import urlencode
from urllib.request import urlopen

from research_engine import extractor
from research_engine import webread_service
from research_engine.schema import ExtractionMethod


class _FakeLock:
    def __init__(self, *, acquire_result: bool):
        self.acquire_result = acquire_result
        self._locked = False
        self.acquire_calls = []

    def acquire(self, timeout=None):
        self.acquire_calls.append(timeout)
        self._locked = self.acquire_result
        return self.acquire_result

    def release(self) -> None:
        if not self._locked:
            raise AssertionError("release called while unlocked")
        self._locked = False

    def locked(self) -> bool:
        return self._locked


class _FailAcquireLock:
    def acquire(self, timeout=None):
        raise AssertionError(f"lock acquire should not be called (timeout={timeout})")

    def release(self) -> None:
        raise AssertionError("lock release should not be called")

    def locked(self) -> bool:
        return False


def _autospec_extract(fake_impl):
    mock = create_autospec(webread_service.extract_clean_text, side_effect=fake_impl)
    assert inspect.signature(mock) == inspect.signature(webread_service.extract_clean_text)
    return mock


def test_webread_service_returns_text_and_then_cache_hit(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "example.txt"
    raw_text_path.write_text("Example Domain\n\nExample body text", encoding="utf-8")
    extract_calls = []

    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path / "cache"))

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        extract_calls.append((url, instruction))
        assert tier == extractor.SourceTier.T2
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "0" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "Example Domain",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    server = webread_service.make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        query = urlencode({"url": "https://example.com"})

        with urlopen(f"{base_url}/read?{query}", timeout=10) as response:
            assert response.status == 200
            first = json.loads(response.read().decode("utf-8"))

        assert first["text"] == "Example Domain\n\nExample body text"
        assert first["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
        assert first["receipt"]["layer"] == 2
        assert first["receipt"]["cache_hit"] is False

        with urlopen(f"{base_url}/read?{query}", timeout=10) as response:
            assert response.status == 200
            second = json.loads(response.read().decode("utf-8"))

        assert second["text"] == "Example Domain\n\nExample body text"
        assert second["receipt"]["cache_hit"] is True
        assert len(extract_calls) == 1
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_webread_service_retries_short_pages_without_enabling_l3(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "short.txt"
    raw_text_path.write_text("Example Domain", encoding="utf-8")
    extract_calls = []

    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path / "cache-retry"))

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        extract_calls.append((url, instruction))
        assert tier == extractor.SourceTier.T2
        if instruction is None:
            return None
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "1" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "Example Domain",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    server = webread_service.make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        query = urlencode({"url": "https://example.com"})

        with urlopen(f"{base_url}/read?{query}", timeout=10) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["text"] == "Example Domain"
        assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
        assert extract_calls == [
            ("https://example.com", None),
            ("https://example.com", "return the readable page text"),
        ]
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_webread_service_prefers_short_page_retry_over_firecrawl(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "short-firecrawl.txt"
    raw_text_path.write_text("Example Domain", encoding="utf-8")
    extract_calls = []

    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path / "cache-firecrawl"))

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        extract_calls.append((url, instruction))
        assert tier == extractor.SourceTier.T2
        if instruction is None:
            return {
                "url": url,
                "domain": "example.com",
                "title": "Wrong Result",
                "author": None,
                "published_date": None,
                "content_hash": "2" * 64,
                "extraction_method": ExtractionMethod.FIRECRAWL.value,
                "raw_text_path": str(raw_text_path),
                "char_count": len(raw_text_path.read_text(encoding="utf-8")),
                "char_text_preview": "Wrong Result",
            }
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "3" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "Example Domain",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    server = webread_service.make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        query = urlencode({"url": "https://example.com"})

        with urlopen(f"{base_url}/read?{query}", timeout=10) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["text"] == "Example Domain"
        assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
        assert extract_calls == [
            ("https://example.com", None),
            ("https://example.com", "return the readable page text"),
        ]
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_extract_returns_l3_busy_when_lock_times_out(monkeypatch, tmp_path) -> None:
    fake_lock = _FakeLock(acquire_result=False)
    cache_dir = tmp_path / "cache-lock-busy"
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)

    def fail_extract(*_args, **_kwargs):
        raise AssertionError("extract_clean_text should not be called when the lock times out")

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fail_extract))

    status_code, payload = webread_service._extract("https://example.com", 3, None)

    assert status_code == 503
    assert payload["text"] == ""
    assert payload["receipt"]["error"] == "L3_BUSY"
    assert fake_lock.acquire_calls == [webread_service.LOCK_WAIT_S]


def test_extract_releases_lock_after_success(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "normal.txt"
    raw_text_path.write_text("normal body text", encoding="utf-8")
    fake_lock = _FakeLock(acquire_result=True)
    cache_dir = tmp_path / "cache-normal"
    l3_lock_path = cache_dir / "l3.lock"
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(webread_service, "_L3_LOCK", l3_lock_path)

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert url == "https://example.com"
        assert instruction is None
        assert tier == extractor.SourceTier.T3
        assert l3_lock_path.exists() is True
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "4" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "normal body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, payload = webread_service._extract("https://example.com", 3, None)

    assert status_code == 200
    assert payload["text"] == "normal body text"
    assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
    assert webread_service._extract_lock.locked() is False
    assert l3_lock_path.exists() is False


def test_extract_non_l3_path_skips_lock_and_uses_t2_tier(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "non-l3-skip-lock.txt"
    raw_text_path.write_text("non l3 skip lock body text", encoding="utf-8")
    cache_dir = tmp_path / "cache-non-l3-skip-lock"
    extract_calls = []
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", _FailAcquireLock())
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        extract_calls.append((url, instruction, tier))
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "4a" * 32,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "non l3 skip lock body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    first_status_code, first_payload = webread_service._extract("https://example.com/1", 2, None)
    second_status_code, second_payload = webread_service._extract("https://example.com/2", 2, None)

    assert first_status_code == 200
    assert second_status_code == 200
    assert first_payload["receipt"]["error"] is None
    assert second_payload["receipt"]["error"] is None
    assert extract_calls == [
        ("https://example.com/1", None, extractor.SourceTier.T2),
        ("https://example.com/2", None, extractor.SourceTier.T2),
    ]


def test_extract_writes_last_request_timestamp(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "timestamp.txt"
    raw_text_path.write_text("timestamp body text", encoding="utf-8")
    cache_dir = tmp_path / "cache-timestamp"
    fake_lock = _FakeLock(acquire_result=True)
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert tier == extractor.SourceTier.T2
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "5" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "timestamp body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, _payload = webread_service._extract("https://example.com", 2, None)

    last_request_path = cache_dir / "last_request"
    assert status_code == 200
    assert last_request_path.exists()
    assert float(last_request_path.read_text(encoding="utf-8")) > 0.0


def test_extract_ask_respects_max_layer_cap(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "ask-layer-cap.txt"
    raw_text_path.write_text("ask layer cap body text", encoding="utf-8")
    cache_dir = tmp_path / "cache-ask-layer-cap"
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", _FailAcquireLock())
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(webread_service, "_L3_LOCK", cache_dir / "l3.lock")

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert instruction == "summarize the page"
        assert tier == extractor.SourceTier.T2
        assert webread_service._L3_LOCK.exists() is False
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "6" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "ask layer cap body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, payload = webread_service._extract("https://example.com", 2, "summarize the page")

    assert status_code == 200
    assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
    assert payload["receipt"]["layer"] == 2
    assert webread_service._L3_LOCK.exists() is False


def test_extract_ask_uses_t3_tier_and_l3_lock_when_max_layer_is_3(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "allow-l3.txt"
    raw_text_path.write_text("allow l3 body text", encoding="utf-8")
    cache_dir = tmp_path / "cache-allow-l3"
    fake_lock = _FakeLock(acquire_result=True)
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(webread_service, "_L3_LOCK", cache_dir / "l3.lock")

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert instruction == "summarize the page"
        assert tier == extractor.SourceTier.T3
        assert webread_service._L3_LOCK.exists() is True
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "7" * 64,
            "extraction_method": ExtractionMethod.AGENT_BROWSER.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "allow l3 body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, payload = webread_service._extract("https://example.com", 3, "summarize the page")

    assert status_code == 200
    assert payload["receipt"]["method"] == ExtractionMethod.AGENT_BROWSER.value
    assert payload["receipt"]["layer"] == 1
    assert webread_service._L3_LOCK.exists() is False
    assert fake_lock.acquire_calls == [webread_service.LOCK_WAIT_S]


def test_extract_non_l3_request_does_not_create_lock(monkeypatch, tmp_path) -> None:
    raw_text_path = tmp_path / "non-l3.txt"
    raw_text_path.write_text("non l3 body text", encoding="utf-8")
    cache_dir = tmp_path / "cache-non-l3"
    fake_lock = _FakeLock(acquire_result=True)
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(webread_service, "_L3_LOCK", cache_dir / "l3.lock")

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert instruction is None
        assert tier == extractor.SourceTier.T2
        assert webread_service._L3_LOCK.exists() is False
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "8" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "non l3 body text",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, payload = webread_service._extract("https://example.com", 2, None)

    assert status_code == 200
    assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
    assert webread_service._L3_LOCK.exists() is False


def test_extract_returns_502_when_result_file_is_unreadable(monkeypatch, tmp_path) -> None:
    missing_raw_text_path = tmp_path / "missing.txt"
    cache_dir = tmp_path / "cache-missing-raw-text"
    fake_lock = _FakeLock(acquire_result=True)
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(webread_service, "_extract_lock", fake_lock)
    monkeypatch.setattr(webread_service.l3_guard, "CACHE_DIR", cache_dir)

    def fake_extract(url: str, *, instruction=None, tier=None, **_kwargs):
        assert instruction is None
        assert tier == extractor.SourceTier.T2
        return {
            "url": url,
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "9" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(missing_raw_text_path),
            "char_count": 0,
            "char_text_preview": "",
        }

    monkeypatch.setattr(webread_service, "extract_clean_text", _autospec_extract(fake_extract))

    status_code, payload = webread_service._extract("https://example.com", 2, None)

    assert status_code == 502
    assert payload["receipt"]["error"] == "result file unreadable"


def test_webread_service_real_extractor_signature_path_does_not_raise_typeerror(
    monkeypatch,
    tmp_path,
) -> None:
    raw_text_path = tmp_path / "real-path.txt"
    raw_text_path.write_text("real extractor path body text", encoding="utf-8")
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path / "cache-real-path"))

    monkeypatch.setattr(extractor, "_extract_pdf_or_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(extractor, "_extract_github_repo", lambda *args, **kwargs: None)
    monkeypatch.setattr(extractor, "_extract_apify_route", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        extractor,
        "_extract_web_ladder",
        lambda *args, **kwargs: {
            "url": "https://example.com",
            "domain": "example.com",
            "title": "Example Domain",
            "author": None,
            "published_date": None,
            "content_hash": "a" * 64,
            "extraction_method": ExtractionMethod.CRAWL4AI.value,
            "raw_text_path": str(raw_text_path),
            "char_count": len(raw_text_path.read_text(encoding="utf-8")),
            "char_text_preview": "real extractor path body text",
        },
    )

    status_code, payload = webread_service._extract("https://example.com", 2, None)

    assert status_code == 200
    assert payload["text"] == "real extractor path body text"
    assert payload["receipt"]["method"] == ExtractionMethod.CRAWL4AI.value
