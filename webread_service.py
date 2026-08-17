from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_engine import l3_guard, paths
from research_engine.extractor import extract_clean_text
from research_engine.schema import ExtractionMethod
from research_engine.schema import SourceTier

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 6 * 60 * 60
CACHE_MAX_ENTRIES = 500
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8077
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "webread"
LOCK_WAIT_S = float(os.environ.get("WEBREAD_LOCK_WAIT_S", "125"))
L2_METHODS = {
    ExtractionMethod.APIFY.value,
    ExtractionMethod.CRAWL4AI.value,
    ExtractionMethod.CRAWLEE.value,
    ExtractionMethod.FIRECRAWL.value,
    ExtractionMethod.SCRAPLING.value,
}

_state_lock = threading.Lock()
_extract_lock = threading.Lock()
_served_requests = 0
_L3_LOCK = l3_guard.CACHE_DIR / "l3.lock"


def _cache_dir() -> Path:
    return Path(os.environ.get("WEBREAD_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _cache_dir() / f"{digest}.json"


def _layer_for_method(method: str) -> int:
    if method in L2_METHODS:
        return 2
    return 1


def _read_cache(url: str) -> dict | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
        path.unlink(missing_ok=True)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    path.touch()
    return payload if isinstance(payload, dict) else None


def _prune_cache() -> None:
    cache_dir = _cache_dir()
    if not cache_dir.exists():
        return
    cache_files = sorted(
        cache_dir.glob("*.json"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )
    for stale_path in cache_files[CACHE_MAX_ENTRIES:]:
        stale_path.unlink(missing_ok=True)


def _write_cache(url: str, text: str, receipt: dict) -> None:
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "text": text,
        "receipt": receipt,
        "cached_at": time.time(),
    }
    _cache_path(url).write_text(json.dumps(payload), encoding="utf-8")
    _prune_cache()


def _read_full_text(record: dict) -> str:
    raw_text_path = Path(str(record["raw_text_path"]))
    return raw_text_path.read_text(encoding="utf-8", errors="ignore")


def _touch_last_request() -> None:
    last_request_path = l3_guard.CACHE_DIR / "last_request"
    last_request_path.parent.mkdir(parents=True, exist_ok=True)
    last_request_path.write_text(str(time.time()), encoding="utf-8")


def _next_cold_start() -> bool:
    global _served_requests
    with _state_lock:
        cold_start = _served_requests == 0
        _served_requests += 1
    return cold_start


def _extract(url: str, max_layer: int, ask: str | None) -> tuple[int, dict]:
    started_at = time.perf_counter()
    try:
        _touch_last_request()
    except Exception as exc:
        paths.safe_log(
            logger,
            logging.WARNING,
            "last-request timestamp could not be touched; extraction continues: %s",
            exc,
        )
    cold_start = _next_cold_start()
    if not ask:
        cached = _read_cache(url)
        if cached is not None:
            receipt = dict(cached.get("receipt") or {})
            receipt.update(
                {
                    "url": url,
                    "cache_hit": True,
                    "cold_start": cold_start,
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    "chars": len(str(cached.get("text") or "")),
                    "error": None,
                }
            )
            return 200, {"text": str(cached.get("text") or ""), "receipt": receipt}

    l3_requested = max_layer >= 3
    requested_tier = (
        SourceTier.T3 if max_layer >= 3 else SourceTier.T2 if max_layer == 2 else SourceTier.T1
    )

    def run_extract() -> dict | None:
        record = extract_clean_text(url, instruction=ask, tier=requested_tier)
        if not ask and (
            record is None
            or str(record.get("extraction_method") or "") == ExtractionMethod.FIRECRAWL.value
        ):
            # Keep L3 gated off, but allow short titled pages like example.com to
            # succeed on the existing ladder by relaxing the char threshold only.
            retry_record = extract_clean_text(
                url,
                instruction="return the readable page text",
                tier=requested_tier,
            )
            if retry_record is not None and (
                record is None
                or str(retry_record.get("extraction_method") or "")
                != ExtractionMethod.FIRECRAWL.value
            ):
                record = retry_record
        return record

    if l3_requested:
        acquired = _extract_lock.acquire(timeout=LOCK_WAIT_S)
        if not acquired:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return 503, {
                "text": "",
                "receipt": {
                    "url": url,
                    "method": "",
                    "layer": max(1, min(max_layer, 3)),
                    "cold_start": cold_start,
                    "cache_hit": False,
                    "elapsed_ms": elapsed_ms,
                    "cost_usd": 0.0,
                    "chars": 0,
                    "error": "L3_BUSY",
                },
            }
        try:
            try:
                _L3_LOCK.parent.mkdir(parents=True, exist_ok=True)
                _L3_LOCK.write_text(str(time.time()), encoding="utf-8")
            except Exception as exc:
                paths.safe_log(
                    logger,
                    logging.WARNING,
                    "L3 cross-process lock file could not be written; "
                    "in-process lock remains held: %s",
                    exc,
                )
            record = run_extract()
        finally:
            try:
                _L3_LOCK.unlink(missing_ok=True)
            except Exception as exc:
                paths.safe_log(
                    logger,
                    logging.WARNING,
                    "L3 lock file could not be removed; stale lock may remain: %s",
                    exc,
                )
            finally:
                _extract_lock.release()
    else:
        record = run_extract()

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    if not record or not record.get("raw_text_path"):
        return 502, {
            "text": "",
            "receipt": {
                "url": url,
                "method": "",
                "layer": max(1, min(max_layer, 3)),
                "cold_start": cold_start,
                "cache_hit": False,
                "elapsed_ms": elapsed_ms,
                "cost_usd": 0.0,
                "chars": 0,
                "error": "extract_clean_text returned no record",
            },
        }

    try:
        text = _read_full_text(record)
    except OSError:
        return 502, {
            "text": "",
            "receipt": {
                "url": url,
                "method": "",
                "layer": max(1, min(max_layer, 3)),
                "cold_start": cold_start,
                "cache_hit": False,
                "elapsed_ms": elapsed_ms,
                "cost_usd": 0.0,
                "chars": 0,
                "error": "result file unreadable",
            },
        }
    method = str(record.get("extraction_method") or "")
    receipt = {
        "url": url,
        "method": method,
        "layer": _layer_for_method(method),
        "cold_start": cold_start,
        "cache_hit": False,
        "elapsed_ms": elapsed_ms,
        "cost_usd": 0.0,
        "chars": len(text),
        "error": None,
    }
    response_body = {"text": text, "receipt": receipt}
    if not ask:
        _write_cache(url, text, receipt)
    return 200, response_body


class WebreadHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebreadHandler(BaseHTTPRequestHandler):
    server_version = "webread/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if parsed.path != "/read":
            self._send_json(404, {"error": "not found"})
            return

        params = parse_qs(parsed.query)
        url = (params.get("url") or [""])[0].strip()
        ask = (params.get("ask") or [""])[0].strip() or None
        max_layer_raw = (params.get("max_layer") or ["2"])[0].strip()
        if not url:
            self._send_json(400, {"error": "missing url"})
            return

        try:
            max_layer = max(1, min(int(max_layer_raw), 3))
        except ValueError:
            self._send_json(400, {"error": "invalid max_layer"})
            return

        status_code, payload = _extract(url, max_layer, ask)
        self._send_json(status_code, payload)


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> WebreadHTTPServer:
    return WebreadHTTPServer((host, port), WebreadHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("WEBREAD_SERVICE_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WEBREAD_SERVICE_PORT", str(DEFAULT_PORT))),
    )
    args = parser.parse_args(argv)

    server = make_server(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
