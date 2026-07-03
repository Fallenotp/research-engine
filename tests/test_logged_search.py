import json
from types import SimpleNamespace

from research_engine import logged_search


ROW_KEYS = {
    "ts",
    "protocol",
    "topic",
    "lane",
    "ok",
    "duration_ms",
    "result_count",
    "error",
    "agent",
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _read_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_searxng_logs_successful_call(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse({"results": [{"url": "https://a"}, {"url": "https://b"}]}),
    )

    result = logged_search.searxng(
        "test query",
        protocol="/search",
        topic="topic-slug",
        agent="agent-c",
    )

    assert result == {"results": [{"url": "https://a"}, {"url": "https://b"}]}
    rows = _read_rows(call_log)
    assert len(rows) == 1
    assert set(rows[0]) == ROW_KEYS
    assert rows[0]["lane"] == "searxng_general"
    assert rows[0]["ok"] is True
    assert rows[0]["result_count"] == 2
    assert rows[0]["protocol"] == "/search"
    assert rows[0]["topic"] == "topic-slug"
    assert rows[0]["agent"] == "agent-c"


def test_proxy_logs_failed_call(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _raise(_req, **_kwargs):
        raise OSError("proxy unavailable")

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _raise)

    result = logged_search.proxy(
        "test query",
        provider="tavily",
        protocol="/research",
        topic="topic-slug",
        agent="agent-c",
    )

    assert result == {"error": "proxy unavailable"}
    rows = _read_rows(call_log)
    assert len(rows) == 1
    assert set(rows[0]) == ROW_KEYS
    assert rows[0]["lane"] == "tavily"
    assert rows[0]["ok"] is False
    assert rows[0]["result_count"] is None
    assert rows[0]["error"] == "proxy unavailable"


def test_searxng_still_returns_when_log_write_fails(monkeypatch) -> None:
    monkeypatch.setattr(logged_search, "CALL_LOG", "/dev/null/research-call-log.jsonl")
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse({"results": [{"url": "https://a"}]}),
    )

    result = logged_search.searxng("test query", protocol="/search", topic="topic-slug")

    assert result == {"results": [{"url": "https://a"}]}


def test_api_lane_logs_and_normalizes_pubmed_results(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse(
            {"esearchresult": {"idlist": ["12345", "67890"]}}
        ),
    )

    result = logged_search.api_lane(
        "pubmed",
        SimpleNamespace(
            method="GET",
            url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            headers={},
            body=None,
            response_format="json",
        ),
        protocol="/search",
        topic="medicine",
        agent="agent-c",
    )

    assert result["results"] == [
        {"title": "PubMed 12345", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
        {"title": "PubMed 67890", "url": "https://pubmed.ncbi.nlm.nih.gov/67890/"},
    ]
    rows = _read_rows(call_log)
    assert rows[0]["lane"] == "pubmed"
    assert rows[0]["ok"] is True
    assert rows[0]["result_count"] == 2


def test_api_lane_normalizes_common_url_fields(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse(
            {"hits": [{"title": "HN item", "objectID": "42"}]}
        ),
    )

    result = logged_search.api_lane(
        "hn_algolia",
        SimpleNamespace(
            method="GET",
            url="https://hn.algolia.com/api/v1/search",
            headers={},
            body=None,
            response_format="json",
        ),
        protocol="/search",
        topic="social",
        agent="agent-c",
    )

    assert result["results"] == [
        {
            "title": "HN item",
            "objectID": "42",
            "url": "https://news.ycombinator.com/item?id=42",
        }
    ]
