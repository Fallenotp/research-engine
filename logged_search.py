import argparse
import datetime
import fcntl
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


CALL_LOG = "/Users/cleo/lattice/data/agent_state/research-call-log.jsonl"
SEARXNG_URL = "http://localhost:8888/search"
PROXY_URL = "http://localhost:18791/search"


logger = logging.getLogger(__name__)


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
) -> None:
    duration_ms = int((time.time() - started) * 1000)
    _append_call(
        {
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
    )


def searxng(query, *, protocol=None, topic=None, agent=None) -> dict:
    error = None
    result_count = None
    payload = None
    started = time.time()

    try:
        url = SEARXNG_URL + "?q=" + urllib.parse.quote(query, safe="") + "&format=json"
        payload = _load_json(url)
        result_count = _result_count(payload)
    except Exception as exc:
        error = str(exc)
        payload = {"error": error}

    _record_call(
        started=started,
        lane="searxng_general",
        protocol=protocol,
        topic=topic,
        agent=agent,
        error=error,
        result_count=result_count,
    )
    return payload


def proxy(
    query,
    *,
    provider=None,
    num_results=10,
    protocol=None,
    topic=None,
    agent=None,
) -> dict:
    error = None
    result_count = None
    payload = None
    started = time.time()

    request_body = {"query": query, "numResults": num_results}
    if provider:
        request_body["provider"] = provider

    try:
        request = urllib.request.Request(
            PROXY_URL,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = _load_json(request)
        result_count = _result_count(payload)
    except Exception as exc:
        error = str(exc)
        payload = {"error": error}

    _record_call(
        started=started,
        lane=provider or "paid_proxy",
        protocol=protocol,
        topic=topic,
        agent=agent,
        error=error,
        result_count=result_count,
    )
    return payload


def api_lane(
    lane: str,
    request,
    *,
    protocol=None,
    topic=None,
    agent=None,
) -> dict:
    error = None
    result_count = None
    payload = None
    started = time.time()

    try:
        body = request.body.encode("utf-8") if request.body is not None else None
        url_request = urllib.request.Request(
            request.url,
            data=body,
            headers=request.headers,
            method=request.method,
        )
        payload = _load_payload(url_request, response_format=request.response_format)
        payload = _normalize_api_lane_payload(lane, payload)
        result_count = _result_count(payload)
    except Exception as exc:
        error = str(exc)
        payload = {"results": [], "error": error}

    _record_call(
        started=started,
        lane=lane,
        protocol=protocol,
        topic=topic,
        agent=agent,
        error=error,
        result_count=result_count,
    )
    return payload


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

    if not args.provider or args.provider == "searxng":
        payload = searxng(args.query, protocol=args.protocol, topic=args.topic)
    else:
        payload = proxy(
            args.query,
            provider=args.provider,
            num_results=args.num,
            protocol=args.protocol,
            topic=args.topic,
        )

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
