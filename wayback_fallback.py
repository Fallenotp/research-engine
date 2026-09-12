from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote

import requests
import trafilatura

from . import paths
from .politeness import DomainCooldown

logger = logging.getLogger("extractor")
_WARNED_MISSING_KEYS = False


def _load_env_file_values(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = value
    return values


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

    return extractor_module._html_meta, extractor_module._payload, extractor_module.fetch_gate


_ENV_FILE_VALUES = _load_env_file_values(paths.env_file())


def keys_available() -> tuple[bool, str]:
    """Return whether the Wayback fallback can authenticate without a request."""
    access = (os.getenv("WAYBACK_ACCESS_KEY") or _ENV_FILE_VALUES.get("WAYBACK_ACCESS_KEY") or "").strip()
    secret = (os.getenv("WAYBACK_SECRET_KEY") or _ENV_FILE_VALUES.get("WAYBACK_SECRET_KEY") or "").strip()
    return (True, "") if access and secret else (False, "no WAYBACK keys")


def try_wayback(source_url: str) -> dict[str, str | None] | None:
    """Final fallback: look up source_url in the Wayback Machine and extract the snapshot."""
    global _WARNED_MISSING_KEYS

    try:
        available, _ = keys_available()
        if not available:
            if not _WARNED_MISSING_KEYS:
                logger.warning("Wayback fallback disabled: missing API keys")
                _WARNED_MISSING_KEYS = True
            return None
        access = (os.getenv("WAYBACK_ACCESS_KEY") or _ENV_FILE_VALUES.get("WAYBACK_ACCESS_KEY") or "").strip()
        secret = (os.getenv("WAYBACK_SECRET_KEY") or _ENV_FILE_VALUES.get("WAYBACK_SECRET_KEY") or "").strip()

        headers = {
            "Authorization": f"LOW {access}:{secret}",
            "User-Agent": paths.user_agent(),
        }
        cdx_url = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url={quote(source_url, safe='')}"
            "&limit=1&output=json&filter=statuscode:200&from=2020&sort=reverse"
        )
        _html_meta, _payload, fetch_gate = _extractor_helpers()
        fetch_gate(cdx_url, group_url=source_url)
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
        fetch_gate(snapshot_url, group_url=source_url)
        response = requests.get(
            snapshot_url, headers=headers, allow_redirects=True, timeout=30
        )
        response.raise_for_status()
        html = response.text
        text = trafilatura.extract(html) or ""
        if len(text.strip()) < 200:
            return None

        html_meta, payload, _fetch_gate = _extractor_helpers()
        meta = html_meta(html)
        return payload(
            str(meta.get("title") or "").strip(),
            text,
            meta.get("author"),
            meta.get("published_date"),
        )
    except DomainCooldown:
        raise
    except Exception as exc:  # pragma: no cover - hard boundary for caller contract
        logger.warning("Wayback fallback failed for %s: %s", source_url, exc)
        return None
