from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from research_engine.apify_accounts import (
    AccountPool,
    ApifyAccount,
    ApifyActorRoute,
    actor_route_for_url,
    load_accounts_from_env,
)

from . import paths
from research_engine.schema import ExtractionMethod, SourceTier
from research_engine.politeness import Politeness, respect_robots

logger = logging.getLogger("extractor")
CACHE_DIR = paths.package_path("cache")
READER_TELEMETRY_LOG = paths.telemetry_path("research-reader-telemetry.jsonl")
BLOCK_STATUSES = frozenset({401, 403, 407, 429, 451})
BLOCKED_LOG_PATH = paths.telemetry_path("research-blocked-sources.jsonl")
_BLOCK_EVENTS: list[dict] = []
CRAWL4AI_SCRIPT = paths.optional_path(paths.CRAWL4AI_SCRIPT_ENV) or paths.package_path(
    "scripts",
    "crawl4ai_fetch.py",
)
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
_LISTICLE_TITLE_RE = re.compile(
    r"(?:\b(?:best|top|cheapest|worst|greatest)\b[^.]{0,40}?\b(?:\d{1,3}|picks|list|roundup)\b)"
    r"|(?:\b\d{1,3}\s+(?:best|top|greatest)\b)"
    r"|(?:\b(?:best|top)\s+\d{1,3}\b)",
    re.IGNORECASE,
)
_LISTICLE_URL_RE = re.compile(
    r"(?:/(?:top|best)(?:[-/]|$))|(?:/(?:top|best)-\d{1,3}(?:-|/|$))|(?:-vs-)",
    re.IGNORECASE,
)
MARKITDOWN_SUFFIXES = {
    ".doc",
    ".docx",
    ".mp3",
    ".m4a",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".wav",
    ".xls",
    ".xlsx",
}
MARKITDOWN_MIME_TYPES = {
    "application/msword",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MARKITDOWN_MIME_PREFIXES = ("audio/", "image/")
KNOWN_METHODS = {method.value for method in ExtractionMethod}
CF_MARKDOWN_MIN_CHARS = 200
_GITHUB_SUB_RESOURCES = frozenset(
    {
        "actions",
        "blob",
        "commit",
        "commits",
        "discussions",
        "issues",
        "projects",
        "pull",
        "raw",
        "releases",
        "settings",
        "wiki",
    }
)
_GITINGEST_EXCLUDE_PATTERNS = {
    "*.lock",
    "*.min.js",
    "*.map",
    "node_modules/*",
    "dist/*",
    "build/*",
    "*.png",
    "*.jpg",
    "*.gif",
    "*.pdf",
    "*.zip",
}
_GITINGEST_TIMEOUT_S = 90
_GITINGEST_CHAR_CAP = 200_000
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
APIFY_CAMOUFOX_ACTOR = "apify/camoufox-scraper"
APIFY_CAMOUFOX_RUNS_PATH = "/v2/acts/apify~camoufox-scraper/runs"
APIFY_OUTPUT_RECORD_KEY = "OUTPUT"
APIFY_WAIT_FOR_FINISH_S = 20
APIFY_MAX_POLLS = 4
APIFY_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"})
APIFY_PAGE_FUNCTION = """
async function pageFunction(context) {
    const { request, page, pushData } = context;
    const title = await page.title();
    const text = await page.evaluate(() => document.body ? document.body.innerText : "");
    await pushData({
        url: request.url,
        title,
        text,
    });
}
""".strip()
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
FIRECRAWL_ADVANCE_STATUSES = {401, 403, 429}
_PROXY_BACKEND = None
_POLITENESS = None
_APIFY_ACCOUNT_POOL = None
_FIRECRAWL_KEY_INDEX = 0
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}(?:[T ][^ ]+)?)|(\d{4}-\d{2})|(\d{4})")
TRACKING_QUERY_PARAMS = {
    "_ga",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
PUBLISHER_HOSTS = (
    "sciencedirect.com",
    "researchgate.net",
    "mdpi.com",
    "springer.com",
    "wiley.com",
    "tandfonline.com",
    "sagepub.com",
    "nature.com",
    "cell.com",
    "jstor.org",
    "academic.oup.com",
)
PAPER_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>&]+", re.IGNORECASE)
PAPER_PII_RE = re.compile(r"S\d{15,17}", re.IGNORECASE)
__all__ = ["compact_search_results", "extract_clean_text"]


class _MissingPdfDependencyError(RuntimeError):
    pass


def _is_web_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _source_ref(url_or_path: str) -> tuple[str, Path | None]:
    if _is_web_url(url_or_path):
        return url_or_path, None
    if url_or_path.startswith("file://"):
        local_path = Path(unquote(urlparse(url_or_path).path)).resolve()
        return local_path.as_uri(), local_path
    local_path = Path(url_or_path).resolve()
    return local_path.as_uri(), local_path


def _domain(source_url: str) -> str:
    return urlparse(source_url).netloc.lower() or "local"


def _plausibly_paper_url(source_url: str) -> bool:
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").lower()
    return bool(PAPER_DOI_RE.search(source_url) or PAPER_PII_RE.search(parsed.path)) or any(
        host == publisher or host.endswith(f".{publisher}")
        for publisher in PUBLISHER_HOSTS
    )


def note_block(
    url: str,
    *,
    method: str,
    reason: str,
    status: int | None = None,
) -> None:
    event = {
        "url": url,
        "domain": _domain(url),
        "method": method,
        "reason": reason,
        "status": status,
        "listicle_flagged": _is_listicle("", url),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    _BLOCK_EVENTS.append(event)
    logger.warning(
        "blocked source url=%s method=%s reason=%s status=%s",
        url,
        method,
        reason,
        status,
    )
    try:
        BLOCKED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BLOCKED_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - best-effort side log
        logger.debug("failed to append blocked-source log: %s", exc)


def blocked_events() -> list[dict]:
    return list(_BLOCK_EVENTS)


def clear_blocked_events() -> None:
    _BLOCK_EVENTS.clear()


def _is_listicle(title: str, url: str) -> bool:
    if _LISTICLE_TITLE_RE.search(title):
        return True
    normalized_path = urlparse(url).path.lower().replace("_", "-")
    return bool(_LISTICLE_URL_RE.search(normalized_path))


def _proxy_config() -> dict:
    try:
        from research_engine.router import load_router

        router = load_router()
        return (router.config.get("proxy") or {}) if router.config else {}
    except Exception as exc:
        logger.debug("proxy config load failed: %s", exc)
        return {}


def _get_proxy_backend():
    global _PROXY_BACKEND
    if _PROXY_BACKEND is None:
        from research_engine.fetch_proxy import load_proxy_backend

        _PROXY_BACKEND = load_proxy_backend({"proxy": _proxy_config()})
    return _PROXY_BACKEND


def _get_politeness():
    global _POLITENESS
    if _POLITENESS is None:
        config = _proxy_config()
        _POLITENESS = Politeness(min_interval_s=float(config.get("min_interval_s", 2.0)))
    return _POLITENESS


def _get_apify_account_pool() -> AccountPool | None:
    global _APIFY_ACCOUNT_POOL
    if _APIFY_ACCOUNT_POOL is None:
        accounts = load_accounts_from_env()
        if not accounts:
            logger.warning("apify actor fetch unavailable: no APIFY accounts loaded")
            return None
        _APIFY_ACCOUNT_POOL = AccountPool(accounts)
    return _APIFY_ACCOUNT_POOL


def _stealthy_fetcher():
    from scrapling.fetchers import StealthyFetcher

    return StealthyFetcher


def _canonical_seen_url(source_url: str) -> str:
    """Canonicalize dedup URLs.

    Rules:
    - lowercase scheme and netloc
    - strip fragments
    - drop common tracking query params
    - strip trailing slashes from non-empty paths
    - preserve path case
    - sort remaining query params alphabetically
    """
    parsed = urlparse(source_url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(query_items, key=lambda item: (item[0], item[1])))
    path = parsed.path.rstrip("/") if parsed.path else ""
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            query,
            "",
        )
    )


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _norm_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.startswith("D:") and len(text) >= 10:
        year, month, day = text[2:6], text[6:8], text[8:10]
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{year}-{month}-{day}"
    match = DATE_RE.search(text)
    if not match:
        return None
    full, year_month, year = match.groups()
    return (full.replace(" ", "T").rstrip(".;,") if full else year_month or year)


def _meta(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find(
            "meta", attrs={"name": key}
        )
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return None


def _html_meta(html: str) -> dict[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    title = _meta(soup, "og:title", "twitter:title", "parsely-title", "citation_title")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    return {
        "title": title or "",
        "author": _meta(
            soup,
            "author",
            "article:author",
            "parsely-author",
            "citation_author",
        ),
        "published_date": _norm_date(
            _meta(
                soup,
                "article:published_time",
                "date",
                "pubdate",
                "citation_publication_date",
                "parsely-pub-date",
            )
        ),
    }


def _title_from_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        return line.lstrip("# ").strip()[:200]
    return ""


def _title_from_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:40]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.lower().startswith("title:"):
                return stripped.partition(":")[2].strip().strip("'\"")[:200]
    return _title_from_text(markdown)


def _payload(
    title: str, body: str, author: str | None = None, published_date: str | None = None
) -> dict[str, str | None]:
    pieces = [title.strip()]
    if author:
        pieces.append(f"Author: {author.strip()}")
    if published_date:
        pieces.append(f"Published: {published_date}")
    pieces.append(body)
    return {
        "title": title.strip(),
        "author": author or None,
        "published_date": _norm_date(published_date),
        "text": _clean("\n\n".join(piece for piece in pieces if piece)),
    }


def _ok(payload: dict[str, str | None], *, min_chars: int = 200) -> bool:
    return bool((payload.get("title") or "").strip()) and len(payload.get("text") or "") >= min_chars


def _append_telemetry_row(row: dict) -> None:
    try:
        READER_TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with READER_TELEMETRY_LOG.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logger.warning("reader telemetry append failed: %s", exc)


def _log(method: str, started_at: float, success: bool, char_count: int, error: str = "") -> None:
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    status = "success" if success else "fail"
    if error:
        logger.warning(
            "method=%s ms=%s status=%s char_count=%s error=%s",
            method,
            elapsed_ms,
            status,
            char_count,
            error,
        )
    else:
        logger.info("method=%s ms=%s status=%s char_count=%s", method, elapsed_ms, status, char_count)
    row = {
        "ts": time.time(),
        "method": method,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "char_count": char_count,
    }
    if error:
        row["error"] = error
    _append_telemetry_row(row)


def _attempt(method: str, fn, *, min_chars: int = 200) -> dict[str, str | None] | None:
    started_at = time.perf_counter()
    try:
        payload = fn()
    except _MissingPdfDependencyError:
        _log(method, started_at, False, 0, "MissingPdfDependencyError")
        return None
    except Exception as exc:  # pragma: no cover - hard boundary for caller contract
        _log(method, started_at, False, 0, f"{type(exc).__name__}: {exc}")
        return None
    char_count = len((payload or {}).get("text") or "")
    success = bool(payload) and _ok(payload, min_chars=min_chars)
    _log(method, started_at, success, char_count)
    return payload if success else None


def _attempt_pdf_rung(
    method: str,
    fn,
    *,
    min_chars: int = 200,
) -> tuple[dict[str, str | None] | None, _MissingPdfDependencyError | None]:
    started_at = time.perf_counter()
    try:
        payload = fn()
    except _MissingPdfDependencyError as exc:
        _log(method, started_at, False, 0, "MissingPdfDependencyError")
        return None, exc
    except Exception as exc:  # pragma: no cover - hard boundary for caller contract
        _log(method, started_at, False, 0, f"{type(exc).__name__}: {exc}")
        return None, None
    char_count = len((payload or {}).get("text") or "")
    success = bool(payload) and _ok(payload, min_chars=min_chars)
    _log(method, started_at, success, char_count)
    return (payload if success else None), None


def _get(
    url: str,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    request_headers = dict(REQUEST_HEADERS)
    if headers:
        request_headers.update(headers)
    response = requests.get(url, headers=request_headers, timeout=timeout)
    if response.status_code in BLOCK_STATUSES:
        note_block(url, method="http_get", reason="http_status", status=response.status_code)
    return response


def _apify_api_request(
    account: ApifyAccount,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict[str, str | int] | None = None,
    timeout: int = 60,
):
    headers = dict(REQUEST_HEADERS)
    headers["Authorization"] = f"Bearer {account.token}"
    response = requests.request(
        method,
        f"https://api.apify.com{path}",
        headers=headers,
        json=json_body,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return response.text


def _is_pdf(source_url: str, local_path: Path | None) -> bool:
    if local_path is not None:
        return local_path.suffix.lower() == ".pdf"
    if urlparse(source_url).path.lower().endswith(".pdf"):
        return True
    content_type = _head_content_type(source_url)
    return content_type == "application/pdf"


def _head_content_type(source_url: str) -> str | None:
    try:
        response = requests.head(
            source_url, headers=REQUEST_HEADERS, allow_redirects=True, timeout=10
        )
        content_type = response.headers.get("content-type", "")
        return content_type.split(";", 1)[0].strip().lower() or None
    except Exception as exc:
        logger.debug("content-type probe failed for %s: %s", source_url, exc)
        return None


def _is_markitdown_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type in MARKITDOWN_MIME_TYPES or any(
        content_type.startswith(prefix) for prefix in MARKITDOWN_MIME_PREFIXES
    )


def _is_markitdown_document(source_url: str, local_path: Path | None) -> bool:
    suffix = (
        local_path.suffix.lower()
        if local_path is not None
        else Path(urlparse(source_url).path).suffix.lower()
    )
    if suffix in MARKITDOWN_SUFFIXES:
        return True
    if local_path is not None:
        return False
    return _is_markitdown_content_type(_head_content_type(source_url))


def _fetch_html(source_url: str, local_path: Path | None) -> tuple[str, dict[str, str | None]]:
    if local_path is not None:
        html = local_path.read_text(encoding="utf-8", errors="ignore")
    else:
        response = _get(source_url)
        response.raise_for_status()
        html = response.text
    return html, _html_meta(html)


def _write_cache(cleaned_text: str) -> tuple[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    raw_text_path = CACHE_DIR / f"{content_hash}.txt"
    raw_text_path.write_text(cleaned_text, encoding="utf-8")
    return content_hash, str(raw_text_path)


def _tier_text(cleaned_text: str, tier: SourceTier | None) -> tuple[str, int]:
    if tier in (None, SourceTier.T1):
        return cleaned_text, len(cleaned_text)
    if tier == SourceTier.T2:
        if len(cleaned_text) <= 9000:
            return cleaned_text, len(cleaned_text)
        trimmed_chars = len(cleaned_text) - 9000
        trimmed_text = (
            f"{cleaned_text[:8000]}\n\n"
            f"[... trimmed {trimmed_chars} chars ...]\n\n"
            f"{cleaned_text[-1000:]}"
        )
        return trimmed_text, 9000
    trimmed_text = cleaned_text[:1500]
    return trimmed_text, min(len(cleaned_text), 1500)


def _claim_seen_url(seen_urls_path: Path | str, source_url: str) -> bool:
    if isinstance(seen_urls_path, str):
        from pathlib import Path as PathLib
        seen_urls_path = PathLib(seen_urls_path)
    canonical_url = _canonical_seen_url(source_url)
    seen_urls_path.parent.mkdir(parents=True, exist_ok=True)
    with seen_urls_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            if any(line.strip() == canonical_url for line in handle):
                return False
            # Mark URLs as seen before extraction to avoid the TOCTOU race; failed extractions
            # stay marked, which trades retries for deterministic duplicate suppression.
            handle.seek(0, 2)
            handle.write(f"{canonical_url}\n")
            handle.flush()
            return True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _record(
    source_url: str,
    method: str,
    payload: dict[str, str | None],
    *,
    tier: SourceTier | None = None,
    extra: dict[str, str] | None = None,
) -> dict:
    cleaned_text = str(payload["text"])
    content_hash, raw_text_path = _write_cache(cleaned_text)
    returned_text, returned_char_count = _tier_text(cleaned_text, tier)
    record = {
        "url": source_url,
        "domain": _domain(source_url),
        "title": str(payload["title"]).strip(),
        "author": payload.get("author"),
        "published_date": payload.get("published_date"),
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": content_hash,
        "extraction_method": method,
        "raw_text_path": raw_text_path,
        "char_count": returned_char_count,
        "char_text_preview": returned_text[:200],
        "listicle_flagged": _is_listicle(str(payload["title"]).strip(), source_url),
    }
    if extra:
        record.update(extra)
    return record


def _is_markdown_response(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip().lower() == "text/markdown"


def _cloudflare_markdown_preflight(
    source_url: str,
) -> tuple[dict[str, str | None], dict[str, str]] | None:
    started_at = time.perf_counter()
    try:
        response = _get(source_url, timeout=15, headers={"Accept": "text/markdown"})
    except Exception as exc:
        logger.debug("cloudflare-markdown preflight failed for %s: %s", source_url, exc)
        return None
    if response.status_code != 200 or not _is_markdown_response(response):
        return None
    raw_markdown = _clean(response.text)
    if not raw_markdown:
        return None
    try:
        cleaned_markdown = _clean(trafilatura.extract(response.text) or "")
    except Exception as exc:
        logger.debug("cloudflare-markdown cleanup failed for %s: %s", source_url, exc)
        cleaned_markdown = ""
    body = cleaned_markdown if len(cleaned_markdown) >= CF_MARKDOWN_MIN_CHARS else raw_markdown
    payload = _payload(_title_from_markdown(response.text), body)
    char_count = len(payload.get("text") or "")
    success = _ok(payload)
    _log(ExtractionMethod.CLOUDFLARE_MARKDOWN.value, started_at, success, char_count)
    if not success:
        return None
    extra: dict[str, str] = {}
    cf_tokens = response.headers.get("x-markdown-tokens")
    if cf_tokens:
        extra["cf_markdown_tokens"] = cf_tokens
    return payload, extra


def _pdf_docling(source_url: str, local_path: Path | None) -> dict[str, str | None]:
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("docling unavailable: %s", exc)
        return {}
    source = str(local_path) if local_path is not None else source_url
    result = DocumentConverter().convert(source)
    document = getattr(result, "document", None)
    if document is None:
        return {}
    export = getattr(document, "export_to_markdown", None) or getattr(
        document, "export_to_text", None
    )
    if export is None:
        return {}
    title = getattr(document, "name", "") or (
        local_path.stem if local_path is not None else Path(urlparse(source_url).path).stem
    )
    return _payload(title, str(export()).strip())


def _pdf_pymupdf(source_url: str, local_path: Path | None) -> dict[str, str | None]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - optional dependency
        raise _MissingPdfDependencyError(
            "PyMuPDF is required for PDF extraction. Install it with "
            "`pip install research-engine[pdf]`."
        ) from exc
    if local_path is not None:
        doc = fitz.open(local_path)
    else:
        response = _get(source_url, timeout=45)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
    try:
        metadata = doc.metadata or {}
        body = "\n\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    title = str(metadata.get("title") or "").strip() or (
        local_path.stem if local_path is not None else Path(urlparse(source_url).path).stem
    )
    author = str(metadata.get("author") or "").strip() or None
    date = str(metadata.get("creationDate") or metadata.get("modDate") or "")
    return _payload(title, body, author, date)


def _markitdown_convert(path_or_url: str) -> str | None:
    try:
        from markitdown import MarkItDown
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("markitdown unavailable: %s", exc)
        return None

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    session.headers.setdefault(
        "Accept",
        "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1",
    )
    try:
        result = MarkItDown(requests_session=session).convert(path_or_url)
    except Exception as exc:
        logger.info("markitdown convert failed for %s: %s", path_or_url, exc)
        return None
    finally:
        session.close()

    markdown = _clean(str(getattr(result, "markdown", "") or result))
    return markdown or None


def _markitdown_payload(source_url: str, local_path: Path | None) -> dict[str, str | None]:
    source = str(local_path) if local_path is not None else source_url
    markdown = _markitdown_convert(source)
    if not markdown:
        return {}
    path_title = (
        local_path.stem if local_path is not None else Path(urlparse(source_url).path).stem
    )
    title = path_title.strip() or _title_from_markdown(markdown) or _title_from_text(markdown)
    return _payload(title, markdown)


def _jina(source_url: str) -> dict[str, str | None]:
    from research_engine.fetch_proxy import env_value

    token = env_value("JINA_API_KEY")
    if token:
        response = _get(
            f"https://r.jina.ai/{source_url}",
            timeout=45,
            headers={"Authorization": f"Bearer {token}"},
        )
    else:
        response = _get(f"https://r.jina.ai/{source_url}", timeout=45)
    response.raise_for_status()
    title = ""
    author = None
    published_date = None
    for line in response.text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Title:"):
            title = stripped.partition(":")[2].strip()
        elif stripped.startswith("Author:"):
            author = stripped.partition(":")[2].strip() or None
        elif stripped.startswith(("Published Time:", "Published:")):
            published_date = stripped.partition(":")[2].strip()
    text = _clean(response.text)
    return _payload(title or _title_from_text(text), text, author, published_date)


def _firecrawl_proxy(source_url: str) -> dict[str, str | None]:
    from research_engine.dispatcher import build_api_lane_request
    from research_engine.router import load_router

    logger.info("firecrawl path=proxy_lane")
    router = load_router()
    lane_config = ((router.config or {}).get("lanes") or {}).get("firecrawl_direct") or {}
    request = build_api_lane_request("firecrawl_direct", lane_config, source_url)
    headers = dict(request.headers)
    if request.body is not None:
        headers.setdefault("Content-Type", "application/json")
    response = requests.request(
        request.method,
        request.url,
        headers=headers,
        data=request.body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return _firecrawl_payload_from_proxy(data, source_url)


def _firecrawl_payload_from_proxy(data: dict, source_url: str) -> dict[str, str | None]:
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {}
    selected = next(
        (item for item in results if isinstance(item, dict) and item.get("url") == source_url),
        next((item for item in results if isinstance(item, dict)), {}),
    )
    body = str(
        selected.get("markdown")
        or selected.get("text")
        or selected.get("content")
        or selected.get("description")
        or ""
    ).strip()
    if not body:
        return {}
    title = str(selected.get("title") or _title_from_text(body)).strip()
    return _payload(title, body)


def _firecrawl_payload_from_direct(data: dict) -> dict[str, str | None]:
    scrape_data = data.get("data") if isinstance(data, dict) else None
    if not isinstance(scrape_data, dict):
        return {}
    metadata = scrape_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    body = str(
        scrape_data.get("markdown")
        or scrape_data.get("text")
        or scrape_data.get("content")
        or scrape_data.get("html")
        or ""
    ).strip()
    if not body:
        return {}
    title = str(scrape_data.get("title") or metadata.get("title") or _title_from_text(body)).strip()
    author = str(metadata.get("author") or "").strip() or None
    published_date = str(
        metadata.get("publishedTime")
        or metadata.get("publishedDate")
        or metadata.get("date")
        or ""
    ).strip() or None
    return _payload(title, body, author, published_date)


def _next_firecrawl_key(keys: list[tuple[str, str]]) -> tuple[str, str]:
    global _FIRECRAWL_KEY_INDEX
    selected = keys[_FIRECRAWL_KEY_INDEX % len(keys)]
    _FIRECRAWL_KEY_INDEX += 1
    return selected


def _firecrawl(source_url: str) -> dict[str, str | None]:
    from research_engine.fetch_proxy import load_firecrawl_keys_from_env

    keys = load_firecrawl_keys_from_env()
    if not keys:
        return _firecrawl_proxy(source_url)

    last_error: requests.HTTPError | None = None
    for attempt in range(len(keys)):
        env_var, key = _next_firecrawl_key(keys)
        logger.info("firecrawl path=direct key=%s", env_var)
        response = requests.request(
            "POST",
            FIRECRAWL_SCRAPE_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                **REQUEST_HEADERS,
            },
            json={"url": source_url, "formats": ["markdown"]},
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", response.status_code)
            if status in FIRECRAWL_ADVANCE_STATUSES and attempt < len(keys) - 1:
                logger.info("firecrawl direct key failed with status=%s; advancing key", status)
                continue
            raise
        return _firecrawl_payload_from_direct(response.json())

    if last_error is not None:
        raise last_error
    return {}


def _trafilatura(source_url: str, local_path: Path | None) -> dict[str, str | None]:
    html, meta = _fetch_html(source_url, local_path)
    return _payload(
        str(meta.get("title") or "").strip(),
        trafilatura.extract(html) or "",
        meta.get("author"),
        meta.get("published_date"),
    )


def _crawl4ai(source_url: str) -> dict[str, str | None]:
    try:
        _, meta = _fetch_html(source_url, None)
    except Exception as exc:
        logger.debug("crawl4ai meta-probe failed for %s: %s", source_url, exc)
        meta = {}
    result = subprocess.run(
        [paths.require_executable(paths.PYTHON_BIN_ENV, "python3"), str(CRAWL4AI_SCRIPT), source_url],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    body = result.stdout.strip()
    title = str(meta.get("title") or _title_from_text(body)).strip()
    return _payload(title, body, meta.get("author"), meta.get("published_date"))


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


async def _crawlee_http_fetch(source_url: str, proxy_url: str | None) -> dict[str, str | None]:
    from crawlee.crawlers import HttpCrawler, HttpCrawlingContext

    result: dict[str, str | None] = {}
    crawler = HttpCrawler(
        max_request_retries=1,
        max_requests_per_crawl=1,
        proxy_configuration=_crawlee_proxy_configuration(proxy_url),
        use_session_pool=True,
    )

    @crawler.router.default_handler
    async def request_handler(context: HttpCrawlingContext) -> None:
        response = context.http_response
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            if context.session:
                context.session.mark_bad()
            return
        raw = await response.read()
        html = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        meta = _html_meta(html)
        body = BeautifulSoup(html, "html.parser").get_text("\n")
        result.update(
            _payload(
                str(meta.get("title") or _title_from_text(body)).strip(),
                body,
                meta.get("author"),
                meta.get("published_date"),
            )
        )

    await crawler.run([source_url])
    return result


def _crawlee_proxy_configuration(proxy_url: str | None):
    if not proxy_url:
        return None
    from crawlee.proxy_configuration import ProxyConfiguration

    return ProxyConfiguration(proxy_urls=[proxy_url])


def _crawlee_http(
    source_url: str,
    *,
    sticky: bool | None = None,
) -> dict[str, str | None]:
    config = _proxy_config()
    use_sticky = bool(config.get("sticky_default", False)) if sticky is None else sticky
    politeness = _get_politeness()
    if respect_robots() and not politeness.allowed(source_url):
        logger.info("robots.txt disallows %s; skipping crawlee", source_url)
        note_block(source_url, method="crawlee", reason="robots_disallow")
        return {}

    domain = _domain(source_url)
    backend = _get_proxy_backend()
    session = backend.acquire(domain=domain, sticky=use_sticky)
    politeness.wait(domain)
    released = False

    try:
        payload = _run_async(_crawlee_http_fetch(source_url, session.proxy_url))
        ok = _ok(payload)
        backend.release(session, ok=ok)
        released = True
        if not ok:
            return {}
        payload["fetch_meta"] = {
            "reader": ExtractionMethod.CRAWLEE.value,
            "proxy_label": session.label,
            "sticky": use_sticky,
            "ladder_depth": 6,
        }
        return payload
    except Exception as exc:
        logger.info("crawlee http failed for %s: %s", source_url, exc)
        if not released:
            # Session release is best-effort on the error path; preserve the original fetch
            # failure instead of surfacing a secondary backend cleanup problem here.
            backend.release(session, ok=False)
        return {}


def _scrapling_stealth(
    source_url: str,
    *,
    sticky: bool | None = None,
) -> dict[str, str | None]:
    config = _proxy_config()
    use_sticky = bool(config.get("sticky_default", False)) if sticky is None else sticky
    politeness = _get_politeness()
    if respect_robots() and not politeness.allowed(source_url):
        logger.info("robots.txt disallows %s; skipping scrapling", source_url)
        note_block(source_url, method="scrapling", reason="robots_disallow")
        return {}

    domain = _domain(source_url)
    backend = _get_proxy_backend()
    session = backend.acquire(domain=domain, sticky=use_sticky)
    politeness.wait(domain)
    released = False

    try:
        fetcher = _stealthy_fetcher()
        page = fetcher.fetch(
            source_url,
            headless=True,
            network_idle=True,
            proxy=session.proxy_url,
            solve_cloudflare=True,
            block_webrtc=True,
        )
        html = str(getattr(page, "html_content", "") or "")
        ok = bool(page) and int(getattr(page, "status", 0) or 0) < 400 and bool(html.strip())
        backend.release(session, ok=ok)
        released = True
        if not ok:
            return {}

        meta = _html_meta(html)
        body = BeautifulSoup(html, "html.parser").get_text("\n")
        payload = _payload(
            str(meta.get("title") or _title_from_text(body)).strip(),
            body,
            meta.get("author"),
            meta.get("published_date"),
        )
        payload["fetch_meta"] = {
            "reader": ExtractionMethod.SCRAPLING.value,
            "proxy_label": session.label,
            "sticky": use_sticky,
            "ladder_depth": 6,
        }
        return payload
    except Exception as exc:
        logger.info("scrapling stealth failed for %s: %s", source_url, exc)
        if not released:
            # Proxy cleanup is deliberate best-effort after a failed scrape. Keep the caller's
            # outcome tied to the scrape failure, not to whether release bookkeeping succeeds.
            backend.release(session, ok=False)
        return {}


def _agent_browser(source_url: str) -> dict[str, str | None] | None:
    politeness = _get_politeness()
    if respect_robots() and not politeness.allowed(source_url):
        logger.info("robots.txt disallows %s; skipping agent_browser", source_url)
        note_block(source_url, method="agent_browser", reason="robots_disallow")
        return None

    politeness.wait(_domain(source_url))
    timeout = int(os.environ.get("WEBREAD_L3_TIMEOUT_S", "120"))
    agent_browser_bin = paths.require_executable(
        paths.AGENT_BROWSER_BIN_ENV,
        "agent-browser",
    )
    session_seed = f"{source_url}:{os.getpid()}:{time.time_ns()}"
    session_hash = hashlib.sha256(session_seed.encode("utf-8")).hexdigest()[:16]
    session_name = f"reader-{os.getpid()}-{session_hash}"

    def run_command(*args: str, json_output: bool = True) -> subprocess.CompletedProcess[str]:
        command = [agent_browser_bin, "--session", session_name, *args]
        if json_output:
            command.append("--json")
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def read_envelope(*args: str) -> dict[str, object] | None:
        response = run_command(*args)
        try:
            envelope = json.loads(response.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(envelope, dict) or envelope.get("success") is not True:
            return None
        data = envelope.get("data")
        return data if isinstance(data, dict) else None

    try:
        open_data = read_envelope("open", source_url)
        if open_data is None:
            return None
        title_data = read_envelope("get", "title")
        if title_data is None:
            return None
        text_data = read_envelope("get", "text", "body")
        if text_data is None:
            return None
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.info("agent_browser failed for %s: %s", source_url, exc)
        return None
    finally:
        try:
            run_command("close", json_output=False)
        except FileNotFoundError:
            logger.info("agent_browser close skipped for %s: command not found", source_url)
        except Exception as exc:
            logger.warning("agent_browser close failed for %s: %s", source_url, exc)

    body = str(text_data.get("text") or "").strip()
    if not body:
        return None

    title = str(title_data.get("title") or open_data.get("title") or "").strip()
    if not title:
        title = _title_from_text(body)
    return _payload(title, body, author=None, published_date=None)


def _apify_actor_input(
    source_url: str,
    route: ApifyActorRoute | None = None,
) -> dict[str, object]:
    if route and route.platform == "instagram":
        return {"directUrls": [source_url], "resultsLimit": 20}
    if route and route.platform == "tiktok":
        return {"startUrls": [source_url], "maxItems": 20}
    return {
        "startUrls": [{"url": source_url}],
        "maxConcurrency": 1,
        "maxRequestRetries": 0,
        "maxRequestsPerCrawl": 1,
        "pageFunction": APIFY_PAGE_FUNCTION,
    }


def _apify_run_data(payload) -> dict[str, object]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def _apify_dataset_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    return []


def _apify_item_text(item: dict) -> str:
    for field in ("text", "markdown", "body", "content", "caption", "description", "html"):
        value = item.get(field)
        if not value:
            continue
        if field == "html":
            return BeautifulSoup(str(value), "html.parser").get_text("\n")
        return str(value)
    return ""


def _apify_output_payload(output) -> dict[str, str | None]:
    if isinstance(output, dict):
        title = str(output.get("title") or "").strip()
        author = str(output.get("author") or "").strip() or None
        published_date = str(output.get("published_date") or "").strip() or None
        text = ""
        for field in ("text", "markdown", "body", "content", "html"):
            value = output.get(field)
            if not value:
                continue
            if field == "html":
                text = BeautifulSoup(str(value), "html.parser").get_text("\n")
            else:
                text = str(value)
            break
        return _payload(title or _title_from_text(text), text, author, published_date)
    if isinstance(output, str):
        return _payload(_title_from_text(output), output)
    return {}


def _apify_actor_fetch(source_url: str) -> dict[str, str | None]:
    route = actor_route_for_url(source_url) or ApifyActorRoute(
        platform="web",
        actor_id=APIFY_CAMOUFOX_ACTOR,
        env_var="",
    )
    pool = _get_apify_account_pool()
    if pool is None:
        return {}
    account = pool.next_account()
    try:
        started = _apify_api_request(
            account,
            "POST",
            route.runs_path,
            json_body=_apify_actor_input(source_url, route),
            timeout=60,
        )
        run_data = _apify_run_data(started)
        run_id = str(run_data.get("id") or "").strip()
        if not run_id:
            return {}

        final_status = ""
        for _ in range(APIFY_MAX_POLLS):
            polled = _apify_api_request(
                account,
                "GET",
                f"/v2/actor-runs/{run_id}",
                params={"waitForFinish": APIFY_WAIT_FOR_FINISH_S},
                timeout=APIFY_WAIT_FOR_FINISH_S + 10,
            )
            run_data = _apify_run_data(polled)
            final_status = str(run_data.get("status") or "").upper()
            if final_status in APIFY_TERMINAL_STATES:
                break
        if final_status != "SUCCEEDED":
            return {}

        items = _apify_dataset_items(
            _apify_api_request(
                account,
                "GET",
                f"/v2/actor-runs/{run_id}/dataset/items",
                params={"format": "json", "clean": "true"},
                timeout=60,
            )
        )
        payload: dict[str, str | None] = {}
        if items:
            first = items[0]
            text = _apify_item_text(first)
            payload = _payload(
                str(first.get("title") or "").strip() or _title_from_text(text),
                text,
                first.get("author"),
                first.get("published_date"),
            )
        if not _ok(payload):
            try:
                output = _apify_api_request(
                    account,
                    "GET",
                    f"/v2/actor-runs/{run_id}/key-value-store/records/{APIFY_OUTPUT_RECORD_KEY}",
                    timeout=60,
                )
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                output = None
            payload = _apify_output_payload(output)
        if not _ok(payload):
            return {}
        payload["fetch_meta"] = {
            "reader": ExtractionMethod.APIFY.value,
            "actor": route.actor_id,
            "run_id": run_id,
            "platform": route.platform,
            "account_id": account.id,
            "ladder_depth": 1 if route.platform in {"instagram", "tiktok"} else 7,
        }
        return payload
    except Exception as exc:
        logger.info("apify fallback failed for %s: %s", source_url, exc)
        return {}


def _local_text(local_path: Path) -> dict[str, str | None]:
    return _payload(
        local_path.stem,
        local_path.read_text(encoding="utf-8", errors="ignore"),
    )


def _finalize_record(
    source_url: str,
    method: str,
    payload: dict[str, str | None],
    *,
    tier: SourceTier | None,
    extra: dict[str, str] | None = None,
) -> dict:
    return _record(source_url, method, payload, tier=tier, extra=extra)


def _extract_pdf_or_document(
    source_url: str,
    local_path: Path | None,
    *,
    prefer_pdf_engine: str,
    min_chars: int,
    tier: SourceTier | None,
) -> dict | None:
    if _is_pdf(source_url, local_path):
        missing_pdf_dependency: _MissingPdfDependencyError | None = None
        methods = [
            (ExtractionMethod.DOCLING.value, lambda: _pdf_docling(source_url, local_path)),
            (ExtractionMethod.PYMUPDF.value, lambda: _pdf_pymupdf(source_url, local_path)),
        ]
        if prefer_pdf_engine != ExtractionMethod.DOCLING.value:
            methods.reverse()
        for method, fn in methods:
            payload, missing_exc = _attempt_pdf_rung(method, fn, min_chars=min_chars)
            if missing_exc is not None:
                missing_pdf_dependency = missing_exc
            if payload:
                return _finalize_record(source_url, method, payload, tier=tier)
        payload = _attempt(
            ExtractionMethod.MARKITDOWN.value,
            lambda: _markitdown_payload(source_url, local_path),
            min_chars=min_chars,
        )
        if payload:
            return _finalize_record(
                source_url, ExtractionMethod.MARKITDOWN.value, payload, tier=tier
            )
        if missing_pdf_dependency is not None:
            raise RuntimeError(str(missing_pdf_dependency)) from missing_pdf_dependency
        return None

    if _is_markitdown_document(source_url, local_path):
        payload = _attempt(
            ExtractionMethod.MARKITDOWN.value,
            lambda: _markitdown_payload(source_url, local_path),
            min_chars=min_chars,
        )
        if payload:
            return _finalize_record(
                source_url, ExtractionMethod.MARKITDOWN.value, payload, tier=tier
            )
        return None

    if local_path is not None and local_path.suffix.lower() not in HTML_SUFFIXES:
        payload = _attempt(
            ExtractionMethod.UNSTRUCTURED.value,
            lambda: _local_text(local_path),
            min_chars=min_chars,
        )
        if payload:
            return _finalize_record(
                source_url, ExtractionMethod.UNSTRUCTURED.value, payload, tier=tier
            )
    return None


def _is_document_source(source_url: str, local_path: Path | None) -> bool:
    return (
        _is_pdf(source_url, local_path)
        or _is_markitdown_document(source_url, local_path)
        or (local_path is not None and local_path.suffix.lower() not in HTML_SUFFIXES)
    )


def _extract_github_repo(
    source_url: str,
    *,
    min_chars: int,
    tier: SourceTier | None,
) -> dict | None:
    if not (_is_web_url(source_url) and _is_github_repo_url(source_url)):
        return None
    github_parts = _github_repo_parts(source_url)
    branch = github_parts[2] if github_parts else None
    payload = _attempt(
        ExtractionMethod.GITINGEST.value,
        lambda: _gitingest(source_url, branch),
        min_chars=min_chars,
    )
    if not payload:
        return None
    return _finalize_record(source_url, ExtractionMethod.GITINGEST.value, payload, tier=tier)


def _extract_apify_route(
    source_url: str,
    *,
    min_chars: int,
    tier: SourceTier | None,
) -> dict | None:
    if not actor_route_for_url(source_url):
        return None
    payload = _attempt(
        ExtractionMethod.APIFY.value,
        lambda: _apify_actor_fetch(source_url),
        min_chars=min_chars,
    )
    if not payload:
        return None
    extra = {"fetch_meta": payload["fetch_meta"]} if payload.get("fetch_meta") else None
    return _finalize_record(source_url, ExtractionMethod.APIFY.value, payload, tier=tier, extra=extra)


def _web_ladder_rungs(source_url: str):
    return [
        (ExtractionMethod.TRAFILATURA.value, lambda: _trafilatura(source_url, None), False),
        (ExtractionMethod.CRAWL4AI.value, lambda: _crawl4ai(source_url), False),
        (ExtractionMethod.JINA.value, lambda: _jina(source_url), False),
        (ExtractionMethod.CRAWLEE.value, lambda: _crawlee_http(source_url), True),
        (ExtractionMethod.SCRAPLING.value, lambda: _scrapling_stealth(source_url), True),
        (ExtractionMethod.AGENT_BROWSER.value, lambda: _agent_browser(source_url), False),
        (ExtractionMethod.FIRECRAWL.value, lambda: _firecrawl(source_url), False),
    ]


def _extract_web_ladder(
    source_url: str,
    *,
    min_chars: int,
    tier: SourceTier | None,
) -> dict | None:
    for method, fn, carries_fetch_meta in _web_ladder_rungs(source_url):
        payload = _attempt(method, fn, min_chars=min_chars)
        if not payload:
            continue
        extra = {"fetch_meta": payload["fetch_meta"]} if carries_fetch_meta and payload.get("fetch_meta") else None
        return _finalize_record(source_url, method, payload, tier=tier, extra=extra)
    return None


def _extract_publisher_or_wayback(
    source_url: str,
    *,
    min_chars: int,
    tier: SourceTier | None,
) -> dict | None:
    if _plausibly_paper_url(source_url):
        from research_engine.publisher_fallback import try_publisher_fallback

        payload = _attempt(
            ExtractionMethod.PUBLISHER_OA.value,
            lambda: try_publisher_fallback(source_url),
            min_chars=min_chars,
        )
        if payload:
            return _finalize_record(
                source_url, ExtractionMethod.PUBLISHER_OA.value, payload, tier=tier
            )

    from research_engine.wayback_fallback import try_wayback

    payload = _attempt(
        ExtractionMethod.WAYBACK.value,
        lambda: try_wayback(source_url),
        min_chars=min_chars,
    )
    if payload:
        return _finalize_record(source_url, ExtractionMethod.WAYBACK.value, payload, tier=tier)
    return None


def _github_repo_parts(url: str) -> tuple[str, str, str | None] | None:
    """Return (owner, repo, branch) when *url* targets a GitHub repo root."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"gist.github.com"} or host.endswith(".gist.github.com"):
        return None
    if host == "raw.githubusercontent.com":
        return None
    if host not in {"github.com", "www.github.com"}:
        return None

    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [segment for segment in path.split("/") if segment]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    if len(parts) == 2:
        return owner, repo, None
    if parts[2] == "tree":
        branch = "/".join(parts[3:]) if len(parts) > 3 else None
        return owner, repo, branch or None
    if parts[2] in _GITHUB_SUB_RESOURCES:
        return None
    return None


def _is_github_repo_url(url: str) -> bool:
    return _github_repo_parts(url) is not None


def _gitingest(url: str, branch: str | None = None) -> dict[str, str | None] | None:
    parts = _github_repo_parts(url)
    if not parts:
        return None
    owner, repo, parsed_branch = parts
    effective_branch = branch if branch is not None else parsed_branch
    repo_url = f"https://github.com/{owner}/{repo}"
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        from gitingest import ingest

        future = executor.submit(
            ingest,
            repo_url,
            branch=effective_branch,
            exclude_patterns=_GITINGEST_EXCLUDE_PATTERNS,
        )
        summary, tree, content = future.result(timeout=_GITINGEST_TIMEOUT_S)
    except Exception as exc:
        logger.debug("gitingest failed for %s: %s", url, exc)
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    body = f"{summary}\n\n{tree}\n\n{content}"
    if len(body) > _GITINGEST_CHAR_CAP:
        body = body[:_GITINGEST_CHAR_CAP] + f"\n\n[truncated at {_GITINGEST_CHAR_CAP} chars]"
    title = f"{owner}/{repo} — GitHub repo (gitingest)"
    return _payload(title, body)


def compact_search_results(results: list[dict], *, max_results: int = 10) -> list[dict]:
    """Strip junk fields from SearXNG/proxy results, keep only what matters."""
    if max_results <= 0:
        return []
    compacted: list[dict] = []
    for result in results:
        url = (result.get("url") or result.get("link") or "").strip()
        title = (result.get("title") or "").strip()
        if not url or not title:
            continue
        snippet = (
            result.get("content")
            or result.get("snippet")
            or result.get("description")
            or ""
        )
        compacted_result = {
            "url": url,
            "title": title,
            "snippet": str(snippet).strip(),
        }
        for key in ("engine", "published_date", "score"):
            if result.get(key) is not None:
                compacted_result[key] = result[key]
        compacted.append(compacted_result)
        if len(compacted) >= max_results:
            break
    return compacted


def extract_clean_text(
    url_or_path: str,
    *,
    prefer_pdf_engine: str = "pymupdf",
    seen_urls_path: Path | None = None,
    tier: SourceTier | None = None,
    instruction: str | None = None,
) -> dict | None:
    source_url, local_path = _source_ref(url_or_path)
    min_chars = 1 if instruction else 200
    if seen_urls_path is not None and not _claim_seen_url(seen_urls_path, source_url):
        logger.info("skipping duplicate URL %s", source_url)
        return {
            "deduplicated": True,
            "url": source_url,
            "extraction_method": "skipped_duplicate",
        }

    handled_as_document = _is_document_source(source_url, local_path)
    extracted = _extract_pdf_or_document(
        source_url,
        local_path,
        prefer_pdf_engine=prefer_pdf_engine,
        min_chars=min_chars,
        tier=tier,
    )
    if extracted or handled_as_document:
        return extracted

    extracted = _extract_github_repo(source_url, min_chars=min_chars, tier=tier)
    if extracted:
        return extracted

    if _is_web_url(source_url):
        extracted = _extract_apify_route(source_url, min_chars=min_chars, tier=tier)
        if actor_route_for_url(source_url):
            return extracted
        if extracted:
            return extracted

        extracted = _extract_web_ladder(source_url, min_chars=min_chars, tier=tier)
        if extracted:
            return extracted
        extracted = _extract_publisher_or_wayback(source_url, min_chars=min_chars, tier=tier)
        if extracted:
            return extracted
        note_block(source_url, method="ladder", reason="all_rungs_failed")
        return None

    payload = _attempt(
        ExtractionMethod.TRAFILATURA.value,
        lambda: _trafilatura(source_url, local_path),
        min_chars=min_chars,
    )
    if payload:
        return _finalize_record(
            source_url, ExtractionMethod.TRAFILATURA.value, payload, tier=tier
        )
    return None


if __name__ == "__main__":
    from unittest.mock import patch

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    def fake_payload(char_count: int) -> dict[str, str | None]:
        return _payload("Example Domain", "A" * char_count, "Tester", "2026-05-09")

    failures = 0
    passes = 0

    def check(name: str, fn) -> None:
        global failures, passes
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passes += 1
            print(f"PASS {name}")

    def baseline_test() -> None:
        with patch(__name__ + "._is_pdf", return_value=False), patch(
            __name__ + "._jina", return_value=fake_payload(2400)
        ):
            result = extract_clean_text("https://example.com")
        assert result is not None
        assert result["char_count"] > 0
        assert result["extraction_method"] in KNOWN_METHODS
        assert Path(result["raw_text_path"]).exists()

    def tier_t2_test() -> None:
        with patch(__name__ + "._is_pdf", return_value=False), patch(
            __name__ + "._jina", return_value=fake_payload(14000)
        ):
            result = extract_clean_text("https://example.com", tier=SourceTier.T2)
        assert result is not None
        assert result["char_count"] <= 9000
        assert Path(result["raw_text_path"]).exists()
        assert len(Path(result["raw_text_path"]).read_text(encoding="utf-8")) > result["char_count"]

    def tier_t3_test() -> None:
        with patch(__name__ + "._is_pdf", return_value=False), patch(
            __name__ + "._jina", return_value=fake_payload(14000)
        ):
            result = extract_clean_text("https://example.com", tier=SourceTier.T3)
        assert result is not None
        assert result["char_count"] <= 1500

    def dedup_test() -> None:
        assert _canonical_seen_url("https://example.com/Page#section?utm_source=x") == _canonical_seen_url(
            "https://example.com/Page"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            seen_urls_path = Path(tmpdir) / "seen_urls.txt"
            with patch(__name__ + "._is_pdf", return_value=False), patch(
                __name__ + "._jina", return_value=fake_payload(2400)
            ):
                first = extract_clean_text(
                    "https://example.com",
                    seen_urls_path=seen_urls_path,
                )
                second = extract_clean_text(
                    "https://example.com",
                    seen_urls_path=seen_urls_path,
                )
            assert first is not None
            assert seen_urls_path.exists()
            assert second == {
                "deduplicated": True,
                "url": "https://example.com",
                "extraction_method": "skipped_duplicate",
            }

    def compact_results_test() -> None:
        noisy_results = [
            {
                "url": f"https://example.com/{idx}",
                "title": f"Title {idx}",
                "content": f"Snippet {idx}",
                "engine": "searxng",
                "published_date": "2026-05-09",
                "score": idx / 10,
                "description": "ignored",
                "link": "ignored",
                "category": "general",
                "thumbnail": "thumb",
                "favicon": "icon",
                "template": "default",
                "positions": [idx],
                "metadata": {"rank": idx},
                "junk": True,
            }
            for idx in range(12)
        ]
        compacted = compact_search_results(noisy_results)
        assert len(compacted) == 10
        assert all(len(item) <= 6 for item in compacted)
        assert all(set(item).issubset({"url", "title", "snippet", "engine", "published_date", "score"}) for item in compacted)

    check("baseline", baseline_test)
    check("tier_t2", tier_t2_test)
    check("tier_t3", tier_t3_test)
    check("dedup", dedup_test)
    check("compact_results", compact_results_test)
    summary = f"PASS {passes} FAIL {failures}"
    print(summary)
    if failures:
        raise SystemExit(1)
