from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from research_engine import extractor
from research_engine.extractor import (
    _cloudflare_markdown_preflight,
    _crawl4ai,
    _crawlee_http,
    _jina,
    _scrapling_stealth,
    _trafilatura,
)
from research_engine.fetch_proxy import NoProxyBackend
from research_engine.politeness import Politeness


OUT = Path("/tmp/extractor_bench.json")
TIMEOUT_S = 45
QUERIES = ("machine learning 2025", "climate news today", "startup funding 2025")
RUN_METHODS = (
    "jina",
    "trafilatura",
    "crawl4ai",
    "crawlee_http",
    "scrapling_stealth",
    "cloudflare-markdown",
)
SKIPPED_METHODS = ("firecrawl", "apify")


def domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def collect_urls() -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for query in QUERIES:
        try:
            response = requests.get(
                "http://localhost:8888/search",
                params={"format": "json", "q": query},
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
        except Exception:
            results = []
        for item in results:
            url = str(item.get("url") or "")
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return spread_domains(urls[:25])


def spread_domains(urls: list[str]) -> list[str]:
    pending = list(urls)
    ordered: list[str] = []
    while pending:
        pick = 0
        if len(ordered) >= 2 and domain(ordered[-1]) == domain(ordered[-2]):
            last_domain = domain(ordered[-1])
            for idx, url in enumerate(pending):
                if domain(url) != last_domain:
                    pick = idx
                    break
        ordered.append(pending.pop(pick))
    return ordered


def call_method(method: str, url: str) -> dict:
    extractor._PROXY_BACKEND = NoProxyBackend()
    extractor._POLITENESS = Politeness(0.0)
    started = time.perf_counter()
    try:
        if method == "jina":
            payload = _jina(url)
        elif method == "trafilatura":
            payload = _trafilatura(url, None)
        elif method == "crawl4ai":
            payload = _crawl4ai(url)
        elif method == "crawlee_http":
            payload = _crawlee_http(url, sticky=False)
        elif method == "scrapling_stealth":
            payload = _scrapling_stealth(url, sticky=False)
        elif method == "cloudflare-markdown":
            result = _cloudflare_markdown_preflight(url)
            payload = result[0] if result else {}
        else:
            raise ValueError(f"unknown method: {method}")
        text = str((payload or {}).get("text") or "")
        return {
            "success": len(text.strip()) > 200,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "char_count": len(text),
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "char_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def child(method: str, url: str, queue) -> None:
    queue.put(call_method(method, url))


def timed_call(method: str, url: str) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    started = time.perf_counter()
    proc = ctx.Process(target=child, args=(method, url, queue))
    proc.start()
    proc.join(TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "success": False,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "char_count": 0,
            "error": "timeout",
        }
    if not queue.empty():
        return queue.get()
    return {
        "success": False,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "char_count": 0,
        "error": f"worker exited {proc.exitcode}",
    }


def summarize(results: list[dict]) -> dict[str, dict]:
    scorecard: dict[str, dict] = {}
    for method in RUN_METHODS:
        rows = [row for row in results if row["method"] == method and not row.get("skipped")]
        attempts = len(rows)
        successes = [row for row in rows if row["success"]]
        success_pct = (len(successes) / attempts * 100) if attempts else 0.0
        elapsed = [row["elapsed_ms"] for row in rows]
        chars = [row["char_count"] for row in rows]
        median_elapsed = statistics.median(elapsed) if elapsed else 0
        median_chars = statistics.median(chars) if chars else 0
        if success_pct >= 50:
            verdict = "KEEP"
        elif success_pct >= 25 and median_elapsed > 15000:
            verdict = "SLOW-LOW-YIELD"
        elif success_pct < 25:
            verdict = "DROP-CANDIDATE"
        else:
            verdict = "LOW-YIELD"
        scorecard[method] = {
            "attempts": attempts,
            "success_pct": round(success_pct, 1),
            "median_elapsed_ms": int(median_elapsed),
            "median_char_count": int(median_chars),
            "verdict": verdict,
        }
    for method in SKIPPED_METHODS:
        scorecard[method] = {
            "attempts": 0,
            "success_pct": 0.0,
            "median_elapsed_ms": 0,
            "median_char_count": 0,
            "verdict": "skipped: metered",
        }
    return scorecard


def print_table(scorecard: dict[str, dict]) -> None:
    print("| method | attempts | success_pct | median_elapsed_ms | median_char_count | verdict |")
    print("|---|---:|---:|---:|---:|---|")
    for method, row in scorecard.items():
        print(
            f"| {method} | {row['attempts']} | {row['success_pct']:.1f}% | "
            f"{row['median_elapsed_ms']} | {row['median_char_count']} | {row['verdict']} |"
        )


def main() -> int:
    urls = collect_urls()
    results: list[dict] = []
    for url in urls:
        for method in RUN_METHODS:
            row = timed_call(method, url)
            row.update({"url": url, "domain": domain(url), "method": method})
            results.append(row)
        for method in SKIPPED_METHODS:
            results.append(
                {
                    "url": url,
                    "domain": domain(url),
                    "method": method,
                    "success": False,
                    "elapsed_ms": 0,
                    "char_count": 0,
                    "skipped": "metered",
                }
            )
    scorecard = summarize(results)
    OUT.write_text(
        json.dumps({"urls": urls, "results": results, "scorecard": scorecard}, indent=2),
        encoding="utf-8",
    )
    print_table(scorecard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
