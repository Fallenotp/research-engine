import json
import pytest
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlsplit

from research_engine import logged_search
from research_engine.query_quality import (
    ResultQualityVerdict,
    broaden_query,
    score_result_quality,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+")


@pytest.fixture(autouse=True)
def _stub_retryable_free_lane_costs(monkeypatch) -> None:
    monkeypatch.setattr(logged_search, "_lane_cost_per_call_usd", lambda _lane: 0.0)


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


def _payload_from_domains(domains: list[str]) -> dict[str, list[dict[str, str]]]:
    return {
        "results": [
            {"url": f"https://{domain}/{index}"}
            for index, domain in enumerate(domains, start=1)
        ]
    }


def _api_request(query: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="GET",
        url=f"https://api.example.test/search?q={quote(query, safe='')}",
        headers={},
        body=None,
        response_format="json",
    )


def _good_payload():
    return _payload_from_domains(
        ["a.example", "b.example", "c.example", "a.example"]
    )


def _thin_payload():
    return _payload_from_domains(["a.example", "b.example"])


def test_empty_verdict_on_free_lane_retries_once_with_broadened_query(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    queries = []
    responses = [FakeResponse({"results": []}), FakeResponse(_good_payload())]

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(req, **_kwargs):
        url = req if isinstance(req, str) else req.full_url
        queries.append(parse_qs(urlsplit(url).query)["q"][0])
        return responses.pop(0)

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng(
        "organic cotton t-shirt manufacturer UK site:example.com",
        protocol="/search",
        topic="apparel",
    )

    assert result == _good_payload()
    assert len(queries) == 2
    assert queries[0] == "organic cotton t-shirt manufacturer UK site:example.com"
    assert queries[1] == "organic cotton t-shirt manufacturer UK"
    assert queries[1] != queries[0]


def test_good_verdict_does_not_retry(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    queries = []

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(req, **_kwargs):
        url = req if isinstance(req, str) else req.full_url
        queries.append(parse_qs(urlsplit(url).query)["q"][0])
        return FakeResponse(_good_payload())

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng("organic cotton manufacturer", topic="apparel")

    assert result == _good_payload()
    assert len(queries) == 1


def test_error_verdict_does_not_retry(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _raise(_req, **_kwargs):
        calls.append("called")
        raise OSError("blocked")

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _raise)

    result = logged_search.searxng("organic cotton manufacturer", topic="apparel")

    assert result == {"error": "blocked"}
    assert len(calls) == 1


def test_empty_string_error_payload_does_not_retry(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []
    payload = {"results": [], "error": ""}

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(_req, **_kwargs):
        calls.append("called")
        return FakeResponse(payload)

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng(
        "organic cotton manufacturer site:example.com",
        topic="apparel",
    )

    assert result == payload
    assert (
        score_result_quality(result, "searxng_general").verdict
        is ResultQualityVerdict.ERROR
    )
    assert len(calls) == 1


def test_retry_winner_keeps_first_good_payload_over_thin_retry_with_more_results() -> None:
    first_payload = _payload_from_domains(
        [f"good-{index}.example" for index in range(1, 9)]
    )
    retry_payload = _payload_from_domains(["junk.example"] * 25)

    assert score_result_quality(first_payload, "searxng_general").verdict is ResultQualityVerdict.GOOD
    assert score_result_quality(retry_payload, "searxng_general").verdict is ResultQualityVerdict.THIN

    winner = logged_search._pick_retry_winner(
        lane="searxng_general",
        first_payload=first_payload,
        retry_payload=retry_payload,
    )

    assert winner is first_payload


def test_retry_winner_uses_more_results_when_verdicts_match() -> None:
    first_payload = _payload_from_domains(["a.example", "b.example", "c.example"])
    retry_payload = _payload_from_domains(
        ["a.example", "b.example", "c.example", "d.example", "e.example"]
    )

    assert score_result_quality(first_payload, "searxng_general").verdict is ResultQualityVerdict.GOOD
    assert score_result_quality(retry_payload, "searxng_general").verdict is ResultQualityVerdict.GOOD

    winner = logged_search._pick_retry_winner(
        lane="searxng_general",
        first_payload=first_payload,
        retry_payload=retry_payload,
    )

    assert winner is retry_payload


def test_retry_winner_prefers_more_domains_when_counts_are_near_tied() -> None:
    first_payload = _payload_from_domains(
        ["a.example", "a.example", "a.example", "a.example", "a.example", "b.example"]
    )
    retry_payload = _payload_from_domains(
        [
            "junk.example",
            "junk.example",
            "junk.example",
            "junk.example",
            "junk.example",
            "junk.example",
            "junk.example",
        ]
    )

    assert score_result_quality(first_payload, "searxng_general").verdict is ResultQualityVerdict.THIN
    assert score_result_quality(retry_payload, "searxng_general").verdict is ResultQualityVerdict.THIN

    winner = logged_search._pick_retry_winner(
        lane="searxng_general",
        first_payload=first_payload,
        retry_payload=retry_payload,
    )

    assert winner is first_payload


def test_retry_winner_keeps_first_payload_on_exact_tie() -> None:
    first_payload = _payload_from_domains(["a.example", "b.example", "c.example"])
    retry_payload = _payload_from_domains(["a.example", "b.example", "c.example"])

    winner = logged_search._pick_retry_winner(
        lane="searxng_general",
        first_payload=first_payload,
        retry_payload=retry_payload,
    )

    assert winner is first_payload


def test_errored_empty_payload_grades_error_and_does_not_retry(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _raise(_req, **_kwargs):
        calls.append("called")
        raise OSError("HTTP 429")

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _raise)

    result = logged_search.api_lane(
        "exa_direct",
        _api_request("organic cotton manufacturer site:example.com"),
        topic="apparel",
    )

    assert result == {"results": [], "error": "HTTP 429"}
    assert score_result_quality(result, "exa_direct").verdict is ResultQualityVerdict.ERROR
    assert len(calls) == 1


@pytest.mark.parametrize(
    "lane",
    [
        "paid_proxy",
        "linkup_direct",
        "tavily_direct",
        "youcom_direct",
        "firecrawl_direct",
    ],
)
def test_paid_lanes_never_retry_even_when_cost_looks_free(
    lane: str,
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(_req, **_kwargs):
        calls.append("called")
        return FakeResponse({"results": []})

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.api_lane(
        lane,
        _api_request("organic cotton manufacturer site:example.com"),
        topic="apparel",
    )

    assert result == {"results": []}
    assert len(calls) == 1


def test_exa_direct_retries_when_empty(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    queries = []
    responses = [FakeResponse({"results": []}), FakeResponse(_good_payload())]

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(req, **_kwargs):
        url = req if isinstance(req, str) else req.full_url
        queries.append(parse_qs(urlsplit(url).query)["q"][0])
        return responses.pop(0)

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.api_lane(
        "exa_direct",
        _api_request("organic cotton manufacturer site:example.com"),
        topic="apparel",
    )

    assert result == _good_payload()
    assert len(queries) == 2
    assert queries[0] == "organic cotton manufacturer site:example.com"
    assert queries[1] == "organic cotton manufacturer"


def test_two_empty_results_stop_after_two_calls_total(tmp_path, monkeypatch) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []
    responses = [FakeResponse({"results": []}), FakeResponse({"results": []})]

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(_req, **_kwargs):
        calls.append("called")
        return responses.pop(0)

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng(
        "organic cotton t-shirt manufacturer UK site:example.com",
        topic="apparel",
    )

    assert result == {"results": []}
    assert len(calls) == 2
    assert len(_read_rows(call_log)) == 2


def test_broaden_query_drops_site_operator_word_for_word() -> None:
    broadened = broaden_query(
        "organic cotton t-shirt manufacturer UK site:example.com",
        ResultQualityVerdict.EMPTY,
    )

    assert broadened == (
        "organic cotton t-shirt manufacturer UK",
        ["dropped restrictive operator: site:example.com"],
    )


def test_broaden_query_unquotes_phrase() -> None:
    broadened = broaden_query(
        '"organic cotton t-shirt manufacturer" UK small runs',
        ResultQualityVerdict.EMPTY,
    )

    assert broadened == (
        "organic cotton t-shirt manufacturer UK small runs",
        ['removed exact-phrase quotes: "organic cotton t-shirt manufacturer"'],
    )


def test_short_bare_query_returns_none_and_does_not_retry(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    calls = []

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(_req, **_kwargs):
        calls.append("called")
        return FakeResponse({"results": []})

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    assert broaden_query("organic cotton tshirts", ResultQualityVerdict.EMPTY) is None

    result = logged_search.searxng("organic cotton tshirts", topic="apparel")

    assert result == {"results": []}
    assert len(calls) == 1


def test_broadened_token_set_is_always_a_subset_of_original_tokens() -> None:
    queries = [
        "organic cotton t-shirt manufacturer UK site:example.com",
        '"organic cotton t-shirt manufacturer" UK small runs',
        "organic cotton manufacturer -reddit",
        "organic cotton t-shirt manufacturer UK small runs wholesale blanks supplier",
    ]

    for query in queries:
        broadened = broaden_query(query, ResultQualityVerdict.EMPTY)
        assert broadened is not None
        broadened_tokens = {token.lower() for token in WORD_RE.findall(broadened[0])}
        original_tokens = {token.lower() for token in WORD_RE.findall(query)}
        assert broadened_tokens <= original_tokens


def test_retry_returning_fewer_results_keeps_original_payload(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    queries = []
    first_payload = _thin_payload()
    second_payload = {"results": [{"url": "https://a.example/1"}]}
    responses = [FakeResponse(first_payload), FakeResponse(second_payload)]

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))

    def _capture(req, **_kwargs):
        url = req if isinstance(req, str) else req.full_url
        queries.append(parse_qs(urlsplit(url).query)["q"][0])
        return responses.pop(0)

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng(
        '"organic cotton manufacturer" UK small runs',
        topic="apparel",
    )

    assert result == first_payload
    assert len(queries) == 2
    assert queries[0] != queries[1]


def test_broaden_query_failure_returns_original_payload_and_records_telemetry(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(logged_search, "_call_log_supports_extended_fields", lambda: True)
    monkeypatch.setattr(
        logged_search.query_quality,
        "broaden_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse({"results": []}),
    )

    result = logged_search.searxng(
        "organic cotton t-shirt manufacturer UK site:example.com",
        topic="apparel",
    )

    assert result == {"results": []}
    rows = _read_rows(call_log)
    assert len(rows) == 1
    assert rows[0]["retrieval_verdict"] == "EMPTY"


def test_retry_telemetry_records_second_row_as_retry_with_transforms(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    responses = [FakeResponse({"results": []}), FakeResponse(_good_payload())]

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(logged_search, "_call_log_supports_extended_fields", lambda: True)
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: responses.pop(0),
    )

    logged_search.searxng(
        "organic cotton t-shirt manufacturer UK site:example.com",
        protocol="/search",
        topic="apparel",
    )

    rows = _read_rows(call_log)
    assert len(rows) == 2
    assert rows[0]["retrieval_verdict"] == "EMPTY"
    assert rows[1]["retrieval_verdict"] == "GOOD"
    assert rows[1]["query_retry"] is True
    assert rows[1]["query_retry_transforms"] == [
        "dropped restrictive operator: site:example.com"
    ]
    assert rows[1]["query_retry_prior_result_count"] == 0
    assert rows[1]["query_retry_prior_verdict"] == "EMPTY"
