from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote

import requests
import trafilatura

from . import paths

logger = logging.getLogger("extractor")
_WARNED_MISSING_KEYS = False


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extractor_helpers():
    extractor_module = sys.modules.get("research_engine.extractor")
    main_module = sys.modules.get("__main__")
    if extractor_module is None and (
        getattr(getattr(main_module, "__spec__", None), "name", None)
        == "research_engine.extractor"
    ):
        extractor_module = main_module
    if extractor_module is None:
        from research_engine import extractor as extractor_module

    return extractor_module._html_meta, extractor_module._payload


if not os.getenv("WAYBACK_ACCESS_KEY") or not os.getenv("WAYBACK_SECRET_KEY"):
    configured_env = paths.env_file()
    if configured_env is not None:
        _load_env_file(configured_env)


def try_wayback(source_url: str) -> dict[str, str | None] | None:
    """Final fallback: look up source_url in the Wayback Machine and extract the snapshot."""
    global _WARNED_MISSING_KEYS

    try:
        access = (os.getenv("WAYBACK_ACCESS_KEY") or "").strip()
        secret = (os.getenv("WAYBACK_SECRET_KEY") or "").strip()
        if not access or not secret:
            if not _WARNED_MISSING_KEYS:
                logger.warning("Wayback fallback disabled: missing API keys")
                _WARNED_MISSING_KEYS = True
            return None

        headers = {
            "Authorization": f"LOW {access}:{secret}",
            "User-Agent": paths.user_agent(),
        }
        cdx_url = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url={quote(source_url, safe='')}"
            "&limit=1&output=json&filter=statuscode:200&from=2020&sort=reverse"
        )
        cdx_response = requests.get(cdx_url, headers=headers, timeout=8)
        cdx_response.raise_for_status()
        rows = cdx_response.json()
        if not isinstance(rows, list) or len(rows) < 2:
            return None

        snapshot = dict(zip(rows[0], rows[1], strict=False))
        timestamp = str(snapshot.get("timestamp") or "").strip()
        original = str(snapshot.get("original") or "").strip()
        if not timestamp or not original:
            return None

        snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{original}"
        response = requests.get(
            snapshot_url, headers=headers, allow_redirects=True, timeout=30
        )
        response.raise_for_status()
        html = response.text
        text = trafilatura.extract(html) or ""
        if len(text.strip()) < 200:
            return None

        html_meta, payload = _extractor_helpers()
        meta = html_meta(html)
        return payload(
            str(meta.get("title") or "").strip(),
            text,
            meta.get("author"),
            meta.get("published_date"),
        )
    except Exception as exc:  # pragma: no cover - hard boundary for caller contract
        logger.warning("Wayback fallback failed for %s: %s", source_url, exc)
        return None
