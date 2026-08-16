from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from . import paths

logger = logging.getLogger("extractor")

__all__ = ["try_publisher_fallback"]

_TIMEOUT_S = 20
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>&]+", re.IGNORECASE)
_PII_RE = re.compile(r"S\d{15,17}", re.IGNORECASE)
_MDPI_RE = re.compile(r"^/([^/]+)/(\d+)/(\d+)/(\d+)(?:/|$)")
_RESEARCHGATE_RE = re.compile(r"/publication/\d+_(.+?)(?:/|$)", re.IGNORECASE)
_TRAILING_DOI_PUNCTUATION = ").,;:] }"
_IN_FALLBACK = threading.local()


def _headers() -> dict[str, str]:
    return {"User-Agent": paths.user_agent()}


def _mail_params(extra: dict[str, str]) -> dict[str, str]:
    params = dict(extra)
    email = paths.contact_email()
    if email:
        params["mailto"] = email
    return params


def _doi_from_url(source_url: str) -> str | None:
    match = _DOI_RE.search(unquote(source_url))
    if not match:
        return None
    return match.group(0).rstrip(_TRAILING_DOI_PUNCTUATION)


def _doi_from_pii(source_url: str) -> str | None:
    match = _PII_RE.search(urlparse(source_url).path)
    if not match:
        return None
    response = requests.get(
        "https://api.crossref.org/works",
        headers=_headers(),
        params=_mail_params({
            "filter": f"alternative-id:{match.group(0)}",
            "rows": 2,
        }),
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    if not items:
        return None
    doi = str(items[0].get("DOI") or "").strip()
    return doi or None


def _doi_from_mdpi(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if not (parsed.hostname or "").lower().endswith("mdpi.com"):
        return None
    match = _MDPI_RE.match(parsed.path)
    if not match:
        return None
    issn, volume, issue, page = match.groups()
    response = requests.get(
        "https://api.crossref.org/works",
        headers=_headers(),
        params=_mail_params({
            "query.bibliographic": f"{issn} {volume} {issue} {page}",
            "rows": 2,
        }),
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    for item in response.json().get("message", {}).get("items", []):
        if (
            issn.upper() in {str(value).upper() for value in item.get("ISSN", [])}
            and str(item.get("volume") or "") == volume
            and str(item.get("issue") or "") == issue
            and str(item.get("page") or "") == page
        ):
            doi = str(item.get("DOI") or "").strip()
            return doi or None
    return None


def _doi_from_researchgate(source_url: str) -> str | None:
    match = _RESEARCHGATE_RE.search(unquote(urlparse(source_url).path))
    if not match:
        return None
    title = match.group(1).replace("_", " ").strip()
    response = requests.get(
        "https://api.openalex.org/works",
        headers=_headers(),
        params=_mail_params({"search": title, "per-page": 2}),
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None
    doi_url = str(results[0].get("doi") or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi_url, flags=re.IGNORECASE)
    return doi or None


def _resolve_doi(source_url: str) -> str | None:
    return (
        _doi_from_url(source_url)
        or _doi_from_mdpi(source_url)
        or _doi_from_pii(source_url)
        or _doi_from_researchgate(source_url)
    )


def _open_access_url(doi: str) -> str | None:
    response = requests.get(
        f"https://api.openalex.org/works/doi:{doi}",
        headers=_headers(),
        params=_mail_params({}),
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    work = response.json()
    if not work.get("open_access", {}).get("is_oa"):
        return None
    location = work.get("best_oa_location") or {}
    return location.get("pdf_url") or location.get("landing_page_url") or None


def _extract_payload(oa_url: str) -> dict[str, str | None] | None:
    from research_engine.extractor import extract_clean_text

    extracted = extract_clean_text(oa_url)
    if not extracted:
        return None
    raw_text_path = str(extracted.get("raw_text_path") or "").strip()
    if not raw_text_path:
        return None
    text = Path(raw_text_path).read_text(encoding="utf-8")
    if not text.strip():
        return None
    return {
        "title": str(extracted.get("title") or "").strip(),
        "text": text,
        "author": extracted.get("author"),
        "published_date": extracted.get("published_date"),
    }


def try_publisher_fallback(source_url: str) -> dict[str, str | None] | None:
    """Resolve source_url to a DOI, find an open-access copy, and extract it.
    Returns an extractor payload dict, or None.
    """
    if getattr(_IN_FALLBACK, "active", False):
        return None
    _IN_FALLBACK.active = True
    try:
        doi = _resolve_doi(source_url)
        oa_url = _open_access_url(doi) if doi else None
        logger.info(
            "publisher fallback source_url=%s doi=%s oa_found=%s",
            source_url,
            doi,
            bool(oa_url),
        )
        return _extract_payload(oa_url) if oa_url else None
    except Exception as exc:  # pragma: no cover - hard boundary for caller contract
        logger.info("publisher fallback failed for %s: %s", source_url, exc)
        return None
    finally:
        _IN_FALLBACK.active = False
