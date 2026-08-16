import argparse
import datetime
import fcntl
from functools import lru_cache
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from . import paths

try:
    from research_engine import query_quality
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import query_quality


CALL_LOG = str(paths.telemetry_path("research-call-log.jsonl"))
SEARXNG_URL = "http://localhost:8888/search"
PROXY_URL = "http://localhost:18791/search"
PROVIDER_ALIASES = {
    "exa_direct": "exa",
    "linkup_direct": "linkup",
    "tavily_direct": "tavily",
    "youcom_direct": "youcom",
}
LANE_CONFIG_ALIASES = {alias: lane for lane, alias in PROVIDER_ALIASES.items()}
# Belt-and-braces guard against a config edit making a paid lane look free.
# exa_direct is intentionally excluded because it is genuinely free and may retry.
NON_RETRYABLE_PAID_LANES = frozenset(
    {
        "paid_proxy",
        "linkup_direct",
        "tavily_direct",
        "youcom_direct",
        "firecrawl_direct",
    }
)
_QUALITY_VERDICT_RANK = {
    query_quality.ResultQualityVerdict.ERROR: 0,
    query_quality.ResultQualityVerdict.EMPTY: 1,
    query_quality.ResultQualityVerdict.THIN: 2,
    query_quality.ResultQualityVerdict.GOOD: 3,
}
_NEAR_TIE_RESULT_COUNT_DELTA = 1


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AttemptOutcome:
    payload: dict[str, Any]
    error: str | None
    result_count: int | None
    quality: query_quality.ResultQuality | None


@dataclass(frozen=True)
class _RetryMetadata:
    transforms: tuple[str, ...]
    prior_result_count: int | None
    prior_verdict: str


def _utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _agent_name(agent):
    return (
        agent
        or os.environ.get("RESEARCH_AGENT")
        or os.environ.get("CLAUDE_AGENT")
        or "unknown"
    )


def _result_count(payload):
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if isinstance(results, list):
        return len(results)
    return None


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = str(provider).strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


@lru_cache(maxsize=1)
def _load_default_router():
    try:
        from research_engine import router as router_module
    except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
        import router as router_module

    return router_module.load_router()


def _lane_config_name(lane: str) -> str:
    normalized = str(lane or "").strip().lower()
    return LANE_CONFIG_ALIASES.get(normalized, normalized)


def _lane_cost_per_call_usd(lane: str) -> float | None:
    lane_name = _lane_config_name(lane)
    try:
        lane_config = _load_default_router().lane_endpoint(lane_name)
    except Exception as exc:
        logger.warning("lane cost lookup failed for %s: %s", lane_name, exc)
        return None

    cost = lane_config.get("cost_per_call_usd")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        logger.warning(
            "lane cost lookup returned non-numeric cost for %s: %r",
            lane_name,
            cost,
        )
        return None


def _load_payload(
    request: str | urllib.request.Request,
    *,
    response_format: str = "json",
) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    if response_format == "atom":
        return {"results": _parse_atom_results(body)}
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("response was not a JSON object")
    return payload


def _load_json(request: str | urllib.request.Request) -> dict[str, Any]:
    return _load_payload(request)


def _parse_atom_results(body: str) -> list[dict[str, str]]:
    root = ET.fromstring(body)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", namespace):
        title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
        link = entry.find("atom:link", namespace)
        url = str(link.attrib.get("href", "")).strip() if link is not None else ""
        author = (
            entry.findtext("atom:author/atom:name", default="", namespaces=namespace) or ""
        ).strip()
        if author.startswith("/u/"):
            author = author[3:]
        if not title or not url:
            continue
        results.append({"title": title, "url": url, "author": author})
    return results


def _append_call(row):
    try:
        parent = os.path.dirname(CALL_LOG)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(CALL_LOG, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logger.warning("failed to append research call-log row: %s", exc)


def _record_call(
    *,
    started: float,
    lane: str,
    protocol,
    topic,
    agent,
    error: str | None,
    result_count: int | None,
    quality: query_quality.ResultQuality | None = None,
    retry_metadata: _RetryMetadata | None = None,
) -> None:
    duration_ms = int((time.time() - started) * 1000)
    _append_call(
        _build_call_row(
            protocol=protocol,
            topic=topic,
            lane=lane,
            error=error,
            result_count=result_count,
            duration_ms=duration_ms,
            agent=agent,
            quality=quality,
            retry_metadata=retry_metadata,
        )
    )


def _repair_query_text(query: str, *, lane: str) -> tuple[str, bool]:
    try:
        issues = query_quality.validate_query(query, lane=lane)
        if not issues:
            return query, False
        logger.warning("query issues for lane %s: %s", lane, _format_query_issues(issues))
        repaired_query, repairs = query_quality.repair_query(query)
        if repairs and repaired_query != query:
            logger.info("query repairs for lane %s: %s", lane, "; ".join(repairs))
            return repaired_query, False
    except Exception as exc:
        logger.warning(
            "query diagnostics failed for lane %s; using original query: %s",
            lane,
            exc,
        )
        return query, True
    return query, False


def _call_log_supports_extended_fields() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


def _build_call_row(
    *,
    protocol,
    topic,
    lane: str,
    error: str | None,
    result_count: int | None,
    duration_ms: int,
    agent,
    quality: query_quality.ResultQuality | None = None,
    retry_metadata: _RetryMetadata | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ts": _utc_iso(),
        "protocol": protocol,
        "topic": topic,
        "lane": lane,
        "ok": error is None,
        "duration_ms": duration_ms,
        "result_count": result_count,
        "error": error,
        "agent": _agent_name(agent),
    }
    supports_extended_fields = _call_log_supports_extended_fields()
    if quality is not None and supports_extended_fields:
        row.update(
            {
                "retrieval_verdict": quality.verdict.value,
                "retrieval_empty": quality.empty,
                "retrieval_unique_domain_count": quality.unique_domain_count,
                "retrieval_top_domain_share": quality.top_domain_share,
                "retrieval_has_error": quality.has_error,
            }
        )
    if supports_extended_fields and retry_metadata is not None:
        row.update(
            {
                "query_retry": True,
                "query_retry_transforms": list(retry_metadata.transforms),
                "query_retry_prior_result_count": retry_metadata.prior_result_count,
                "query_retry_prior_verdict": retry_metadata.prior_verdict,
            }
        )
    return row


def _format_query_issues(issues: query_quality.QueryIssues) -> str:
    parts: list[str] = []
    for issue in issues.issues:
        if issue.detail:
            parts.append(f"{issue.code.value}:{issue.detail}")
        else:
            parts.append(issue.code.value)
    return ", ".join(parts)


def _prepare_query(query: str, *, lane: str) -> tuple[str, bool]:
    return _repair_query_text(query, lane=lane)


def _decode_request_body(body: str | bytes | None) -> str | None:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        return body
    return None


def _extract_request_query(request) -> str | None:
    body_text = _decode_request_body(getattr(request, "body", None))

    query = _query_from_body(body_text)
    if query:
        return query
    return _query_from_url(str(getattr(request, "url", "")))


def _extract_request_query_from_parts(
    url: str,
    body: str | bytes | None,
) -> str | None:
    body_text = _decode_request_body(body)

    query = _query_from_body(body_text)
    if query:
        return query
    return _query_from_url(url)


def _query_from_body(body: str | None) -> str | None:
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in query_quality._QUERY_PARAM_NAMES:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _unwrap_query_candidate(value)

    parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
    for key in query_quality._QUERY_PARAM_NAMES:
        values = parsed.get(key)
        if values:
            candidate = values[0]
            if candidate and candidate.strip():
                return _unwrap_query_candidate(candidate)
    return None


def _query_from_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlsplit(url)
    query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key in query_quality._QUERY_PARAM_NAMES:
        values = query_params.get(key)
        if values:
            candidate = values[0]
            if candidate and candidate.strip():
                return _unwrap_query_candidate(candidate)

    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return None
    candidate = urllib.parse.unquote(path_parts[-1]).strip()
    if candidate and " " in candidate:
        return _unwrap_query_candidate(candidate)
    return None


def _unwrap_query_candidate(candidate: str) -> str:
    text = urllib.parse.unquote(str(candidate or "")).strip()
    if ":" not in text:
        return text
    prefix, remainder = text.split(":", 1)
    if (
        prefix.lower() not in query_quality.SEARCH_OPERATOR_NAMES
        and remainder.strip()
        and " " in remainder
    ):
        return remainder.strip()
    return text


def _prepare_api_request(
    request,
    *,
    lane: str,
) -> tuple[str, str | bytes | None, bool]:
    original_url = str(request.url)
    original_body = request.body
    query = _extract_request_query(request)
    if not query:
        return original_url, original_body, False

    repaired_query, diagnostics_failed = _repair_query_text(query, lane=lane)
    if diagnostics_failed or repaired_query == query:
        return original_url, original_body, diagnostics_failed

    repaired_url, repaired_body = _replace_request_query(
        original_url=original_url,
        original_body=original_body,
        original_query=query,
        repaired_query=repaired_query,
    )
    return repaired_url, repaired_body, False


def _replace_request_query(
    *,
    original_url: str,
    original_body: str | bytes | None,
    original_query: str,
    repaired_query: str,
) -> tuple[str, str | bytes | None]:
    if repaired_query == original_query:
        return original_url, original_body

    encoded_original = urllib.parse.quote(original_query, safe="")
    encoded_repaired = urllib.parse.quote(repaired_query, safe="")

    def _replace_text(text: str) -> str:
        replaced = text
        if encoded_original:
            replaced = replaced.replace(encoded_original, encoded_repaired)
        return replaced.replace(original_query, repaired_query)

    repaired_url = _replace_text(original_url)
    if original_body is None:
        return repaired_url, None
    if isinstance(original_body, bytes):
        repaired_body = _replace_text(original_body.decode("utf-8", errors="replace"))
        return repaired_url, repaired_body.encode("utf-8")
    return repaired_url, _replace_text(original_body)


def _score_payload_quality(
    payload: dict[str, Any] | None,
    *,
    lane: str,
) -> query_quality.ResultQuality | None:
    try:
        if not isinstance(payload, dict):
            return None
        return query_quality.score_result_quality(payload, lane)
    except Exception as exc:
        logger.warning("result-quality scoring failed for lane %s: %s", lane, exc)
        return None


def _run_logged_attempt(
    *,
    lane: str,
    protocol,
    topic,
    agent,
    execute,
    error_payload_factory,
    retry_metadata: _RetryMetadata | None = None,
) -> _AttemptOutcome:
    error = None
    result_count = None
    payload = None
    started = time.time()

    try:
        payload = execute()
        result_count = _result_count(payload)
    except Exception as exc:
        error = str(exc)
        payload = error_payload_factory(error)

    quality = _score_payload_quality(payload, lane=lane)
    _record_call(
        started=started,
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        error=error,
        result_count=result_count,
        quality=quality,
        retry_metadata=retry_metadata,
    )
    return _AttemptOutcome(
        payload=payload,
        error=error,
        result_count=result_count,
        quality=quality,
    )


def _should_retry_lane_call(
    lane: str,
    quality: query_quality.ResultQuality | None,
) -> bool:
    # Exactly one broadened retry is structural: callers execute one first
    # attempt, then _retry_with_broadened_query may issue one follow-up and
    # returns immediately. A numeric retry budget knob here would be misleading.
    if quality is None:
        return False
    if quality.verdict not in {
        query_quality.ResultQualityVerdict.EMPTY,
        query_quality.ResultQualityVerdict.THIN,
    }:
        return False
    if _lane_config_name(lane) in NON_RETRYABLE_PAID_LANES:
        return False
    cost_per_call = _lane_cost_per_call_usd(lane)
    return cost_per_call is not None and cost_per_call <= 0


def _pick_retry_winner(
    *,
    lane: str,
    first_payload: dict[str, Any],
    retry_payload: dict[str, Any],
) -> dict[str, Any]:
    """Keep the first payload unless the retry is clearly better.

    Verdict rank dominates (GOOD > THIN > EMPTY > ERROR). Only equal verdicts
    consult raw result counts; when those counts differ by at most one result,
    prefer the payload with more unique domains before count. Remaining ties use
    lower top-domain share, and an exact tie always keeps the first payload.
    """

    first_quality = query_quality.score_result_quality(first_payload, lane)
    retry_quality = query_quality.score_result_quality(retry_payload, lane)
    first_rank = _QUALITY_VERDICT_RANK[first_quality.verdict]
    retry_rank = _QUALITY_VERDICT_RANK[retry_quality.verdict]

    if retry_rank != first_rank:
        return retry_payload if retry_rank > first_rank else first_payload

    count_delta = retry_quality.result_count - first_quality.result_count
    if abs(count_delta) <= _NEAR_TIE_RESULT_COUNT_DELTA:
        if retry_quality.unique_domain_count != first_quality.unique_domain_count:
            return (
                retry_payload
                if retry_quality.unique_domain_count > first_quality.unique_domain_count
                else first_payload
            )
        if retry_quality.top_domain_share != first_quality.top_domain_share:
            return (
                retry_payload
                if retry_quality.top_domain_share < first_quality.top_domain_share
                else first_payload
            )

    if count_delta != 0:
        return retry_payload if count_delta > 0 else first_payload
    if retry_quality.unique_domain_count != first_quality.unique_domain_count:
        return (
            retry_payload
            if retry_quality.unique_domain_count > first_quality.unique_domain_count
            else first_payload
        )
    if retry_quality.top_domain_share != first_quality.top_domain_share:
        return (
            retry_payload
            if retry_quality.top_domain_share < first_quality.top_domain_share
            else first_payload
        )
    return first_payload


def _retry_with_broadened_query(
    *,
    lane: str,
    query: str | None,
    diagnostics_failed: bool,
    first_attempt: _AttemptOutcome,
    protocol,
    topic,
    agent,
    execute_retry,
    error_payload_factory,
) -> dict[str, Any]:
    if diagnostics_failed or not query or not _should_retry_lane_call(lane, first_attempt.quality):
        return first_attempt.payload
    assert first_attempt.quality is not None

    try:
        broadened = query_quality.broaden_query(query, first_attempt.quality.verdict)
    except Exception as exc:
        logger.warning(
            "query broadening failed for lane %s; returning original payload: %s",
            lane,
            exc,
        )
        return first_attempt.payload

    if broadened is None:
        return first_attempt.payload

    broadened_query, transforms = broadened
    logger.info(
        "retrying lane %s after %s with: %s",
        lane,
        first_attempt.quality.verdict.value,
        "; ".join(transforms),
    )
    retry_attempt = _run_logged_attempt(
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute=lambda: execute_retry(broadened_query),
        error_payload_factory=error_payload_factory,
        retry_metadata=_RetryMetadata(
            transforms=tuple(transforms),
            prior_result_count=first_attempt.result_count,
            prior_verdict=first_attempt.quality.verdict.value,
        ),
    )
    return _pick_retry_winner(
        lane=lane,
        first_payload=first_attempt.payload,
        retry_payload=retry_attempt.payload,
    )


def searxng(query, *, protocol=None, topic=None, agent=None) -> dict:
    lane = "searxng_general"
    effective_query, diagnostics_failed = _prepare_query(query, lane=lane)

    def _execute(attempt_query: str) -> dict[str, Any]:
        url = SEARXNG_URL + "?q=" + urllib.parse.quote(attempt_query, safe="")
        return _load_json(url + "&format=json")

    first_attempt = _run_logged_attempt(
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute=lambda: _execute(effective_query),
        error_payload_factory=lambda error: {"error": error},
    )
    return _retry_with_broadened_query(
        lane=lane,
        query=effective_query,
        diagnostics_failed=diagnostics_failed,
        first_attempt=first_attempt,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute_retry=_execute,
        error_payload_factory=lambda error: {"error": error},
    )


def proxy(
    query,
    *,
    provider=None,
    num_results=10,
    protocol=None,
    topic=None,
    agent=None,
) -> dict:
    normalized_provider = _normalize_provider(provider)
    lane = normalized_provider or "paid_proxy"
    effective_query, diagnostics_failed = _prepare_query(query, lane=lane)

    def _execute(attempt_query: str) -> dict[str, Any]:
        request_body = {"query": attempt_query, "numResults": num_results}
        if normalized_provider:
            request_body["provider"] = normalized_provider
        request = urllib.request.Request(
            PROXY_URL,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _load_json(request)

    first_attempt = _run_logged_attempt(
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute=lambda: _execute(effective_query),
        error_payload_factory=lambda error: {"error": error},
    )
    return _retry_with_broadened_query(
        lane=lane,
        query=effective_query,
        diagnostics_failed=diagnostics_failed,
        first_attempt=first_attempt,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute_retry=_execute,
        error_payload_factory=lambda error: {"error": error},
    )


def api_lane(
    lane: str,
    request,
    *,
    protocol=None,
    topic=None,
    agent=None,
) -> dict:
    request_url, request_body, diagnostics_failed = _prepare_api_request(request, lane=lane)
    effective_query = _extract_request_query_from_parts(request_url, request_body)

    def _execute_request(
        attempt_url: str,
        attempt_body: str | bytes | None,
    ) -> dict[str, Any]:
        body = attempt_body.encode("utf-8") if isinstance(attempt_body, str) else attempt_body
        url_request = urllib.request.Request(
            attempt_url,
            data=body,
            headers=request.headers,
            method=request.method,
        )
        payload = _load_payload(url_request, response_format=request.response_format)
        return _normalize_api_lane_payload(lane, payload)

    def _execute_query(attempt_query: str) -> dict[str, Any]:
        if not effective_query:
            return _execute_request(request_url, request_body)
        retry_url, retry_body = _replace_request_query(
            original_url=request_url,
            original_body=request_body,
            original_query=effective_query,
            repaired_query=attempt_query,
        )
        return _execute_request(retry_url, retry_body)

    first_attempt = _run_logged_attempt(
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute=lambda: _execute_request(request_url, request_body),
        error_payload_factory=lambda error: {"results": [], "error": error},
    )
    return _retry_with_broadened_query(
        lane=lane,
        query=effective_query,
        diagnostics_failed=diagnostics_failed,
        first_attempt=first_attempt,
        protocol=protocol,
        topic=topic,
        agent=agent,
        execute_retry=_execute_query,
        error_payload_factory=lambda error: {"results": [], "error": error},
    )


def _normalize_api_lane_payload(lane: str, payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if isinstance(results, list):
        payload["results"] = [_normalize_result_item(lane, item) for item in results]
        return payload

    if lane == "pubmed":
        search_result = payload.get("esearchresult")
        ids = search_result.get("idlist") if isinstance(search_result, dict) else None
        if isinstance(ids, list):
            payload["results"] = [
                {
                    "title": f"PubMed {pmid}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
                for pmid in ids
                if str(pmid).strip()
            ]
        return payload

    data = payload.get("data")
    if isinstance(data, list):
        payload["results"] = [_normalize_result_item(lane, item) for item in data]
        return payload

    hits = payload.get("hits")
    if isinstance(hits, list):
        payload["results"] = [_normalize_result_item(lane, item) for item in hits]
        return payload

    items = payload.get("items")
    if isinstance(items, list):
        payload["results"] = [_normalize_result_item(lane, item) for item in items]
    return payload


def _normalize_result_item(lane: str, item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"title": str(item), "url": ""}

    result = dict(item)
    if result.get("url"):
        return result

    url = (
        result.get("absolute_url")
        or result.get("html_url")
        or result.get("paper_url")
        or result.get("story_url")
    )
    if not url and lane == "semantic_scholar" and result.get("paperId"):
        url = f"https://www.semanticscholar.org/paper/{result['paperId']}"
    if not url and lane == "hn_algolia" and result.get("objectID"):
        url = f"https://news.ycombinator.com/item?id={result['objectID']}"
    if not url and lane == "courtlistener" and result.get("cluster"):
        url = result["cluster"]

    if url:
        result["url"] = str(url)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    parser.add_argument("--protocol")
    parser.add_argument("--topic")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("query")
    args = parser.parse_args()

    provider = _normalize_provider(args.provider)
    if not provider or provider == "searxng":
        payload = searxng(args.query, protocol=args.protocol, topic=args.topic)
    else:
        payload = proxy(
            args.query,
            provider=provider,
            num_results=args.num,
            protocol=args.protocol,
            topic=args.topic,
        )

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
