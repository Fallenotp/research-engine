import json
import re
from urllib.parse import parse_qs, urlsplit

from research_engine import logged_search
from research_engine.query_quality import (
    QueryIssueCode,
    ResultQualityVerdict,
    repair_query,
    score_result_quality,
    validate_query,
)


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")


def _word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _is_subsequence(candidate: list[str], original: list[str]) -> bool:
    index = 0
    for token in candidate:
        while index < len(original) and original[index] != token:
            index += 1
        if index == len(original):
            return False
        index += 1
    return True


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


def test_regression_query_is_flagged_and_only_truncated_operator_is_repaired() -> None:
    query = "UK cut and sew organic cotton t-shirt manufacturer small run UK site:c"

    issues = validate_query(query)

    assert issues.has(QueryIssueCode.TRUNCATED_OPERATOR)
    assert "site:c" in issues.details_for(QueryIssueCode.TRUNCATED_OPERATOR)
    assert issues.has(QueryIssueCode.DUPLICATE_TERMS)
    assert "UK" in issues.details_for(QueryIssueCode.DUPLICATE_TERMS)

    repaired, repairs = repair_query(query)

    assert repaired == "UK cut and sew organic cotton t-shirt manufacturer small run UK"
    assert repairs == ["dropped truncated operator: site:c"]


def test_clean_query_has_no_issues_and_no_repairs() -> None:
    query = "organic cotton t-shirt manufacturer united kingdom"

    assert not validate_query(query)
    assert repair_query(query) == (query, [])


def test_unbalanced_quotes_are_detected_and_closed() -> None:
    query = '"organic cotton manufacturer'

    issues = validate_query(query)
    repaired, repairs = repair_query(query)

    assert issues.has(QueryIssueCode.UNBALANCED_QUOTES)
    assert repaired == '"organic cotton manufacturer"'
    assert repairs == ["closed unbalanced quote"]


def test_lane_specific_operators_are_flagged_only_for_api_lanes() -> None:
    query = "site:example.com transformer attention"

    api_issues = validate_query(query, lane="arxiv")
    web_issues = validate_query(query, lane="searxng_general")

    assert api_issues.has(QueryIssueCode.UNSUPPORTED_OPERATOR_FOR_LANE)
    assert "site:example.com" in api_issues.details_for(
        QueryIssueCode.UNSUPPORTED_OPERATOR_FOR_LANE
    )
    assert not web_issues.has(QueryIssueCode.UNSUPPORTED_OPERATOR_FOR_LANE)


def test_repair_query_collapses_only_obvious_duplicate_spill() -> None:
    query = "cotton cotton cotton manufacturer"

    repaired, repairs = repair_query(query)

    assert repaired == "cotton manufacturer"
    assert repairs == ["collapsed duplicated term: cotton"]


def test_score_result_quality_distinguishes_empty_error_good_and_thin() -> None:
    empty = score_result_quality({"results": []}, "searxng_general")
    errored = score_result_quality({"results": [], "error": "HTTP 429"}, "pubmed")
    errored_exception = score_result_quality(
        {"results": [], "exception": "timeout"},
        "pubmed",
    )
    errored_status = score_result_quality({"results": [], "status_code": 429}, "pubmed")
    good = score_result_quality(
        {
            "results": [
                {"url": "https://a.example/1"},
                {"url": "https://b.example/1"},
                {"url": "https://c.example/1"},
                {"url": "https://a.example/2"},
            ]
        },
        "searxng_general",
    )
    thin = score_result_quality(
        {
            "results": [
                {"url": "https://a.example/1"},
                {"url": "https://a.example/2"},
                {"url": "https://a.example/3"},
                {"url": "https://a.example/4"},
                {"url": "https://a.example/5"},
                {"url": "https://b.example/1"},
            ]
        },
        "searxng_general",
    )

    assert empty.empty is True
    assert empty.result_count == 0
    assert empty.verdict is ResultQualityVerdict.EMPTY

    assert errored.has_error is True
    assert errored.error == "HTTP 429"
    assert errored.verdict is ResultQualityVerdict.ERROR
    assert errored_exception.error == "timeout"
    assert errored_exception.verdict is ResultQualityVerdict.ERROR
    assert errored_status.error == "HTTP 429"
    assert errored_status.verdict is ResultQualityVerdict.ERROR

    assert good.result_count == 4
    assert good.unique_domain_count == 3
    assert good.top_domain_share == 0.5
    assert good.verdict is ResultQualityVerdict.GOOD

    assert thin.result_count == 6
    assert thin.unique_domain_count == 2
    assert thin.top_domain_share == 5 / 6
    assert thin.verdict is ResultQualityVerdict.THIN


def test_score_result_quality_treats_present_falsy_error_markers_as_errors() -> None:
    empty_string_error = score_result_quality(
        {"results": [], "error": ""},
        "searxng_general",
    )
    bare_timeout = score_result_quality(
        {"results": [], "exception": TimeoutError()},
        "searxng_general",
    )

    assert empty_string_error.error == "error"
    assert empty_string_error.verdict is ResultQualityVerdict.ERROR
    assert bare_timeout.error == "exception"
    assert bare_timeout.verdict is ResultQualityVerdict.ERROR


def test_score_result_quality_without_error_key_stays_good() -> None:
    payload = {
        "results": [
            {"url": f"https://{domain}/{index}"}
            for index, domain in enumerate(
                ["a.example", "b.example", "c.example", "d.example"] * 2,
                start=1,
            )
        ]
    }

    quality = score_result_quality(payload, "searxng_general")

    assert "error" not in payload
    assert quality.result_count == 8
    assert quality.verdict is ResultQualityVerdict.GOOD


def test_logged_search_fail_safe_uses_original_query_when_validation_raises(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"
    captured = {}
    original_query = (
        "UK cut and sew organic cotton t-shirt manufacturer small run UK site:c"
    )

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(
        logged_search.query_quality,
        "validate_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    def _capture(req, **_kwargs):
        captured["url"] = req if isinstance(req, str) else req.full_url
        return FakeResponse({"results": [{"url": "https://example.com/1"}]})

    monkeypatch.setattr(logged_search.urllib.request, "urlopen", _capture)

    result = logged_search.searxng(original_query, protocol="/search", topic="topic")

    assert result == {"results": [{"url": "https://example.com/1"}]}
    assert parse_qs(urlsplit(captured["url"]).query)["q"] == [original_query]
    rows = _read_rows(call_log)
    assert len(rows) == 1
    assert rows[0]["lane"] == "searxng_general"
    assert rows[0]["result_count"] == 1


def test_logged_search_records_retrieval_quality_fields_when_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    call_log = tmp_path / "agent_state" / "research-call-log.jsonl"

    monkeypatch.setattr(logged_search, "CALL_LOG", str(call_log))
    monkeypatch.setattr(logged_search, "_call_log_supports_extended_fields", lambda: True)
    monkeypatch.setattr(
        logged_search.urllib.request,
        "urlopen",
        lambda _req, **_kwargs: FakeResponse(
            {
                "results": [
                    {"url": "https://a.example/1"},
                    {"url": "https://b.example/1"},
                    {"url": "https://c.example/1"},
                    {"url": "https://a.example/2"},
                ]
            }
        ),
    )

    logged_search.searxng("organic cotton manufacturer", protocol="/search", topic="topic")

    row = _read_rows(call_log)[0]
    assert row["retrieval_verdict"] == "GOOD"
    assert row["retrieval_empty"] is False
    assert row["retrieval_unique_domain_count"] == 3
    assert row["retrieval_top_domain_share"] == 0.5
    assert row["retrieval_has_error"] is False


def test_repair_query_preserves_meaningful_word_order() -> None:
    queries = [
        "UK cut and sew organic cotton t-shirt manufacturer small run UK site:c",
        '"organic cotton manufacturer',
        "cotton cotton cotton manufacturer",
        "site:example.com organic cotton supplier",
        "  filetype:   ",
    ]

    for query in queries:
        repaired, _repairs = repair_query(query)
        repaired_tokens = _word_tokens(repaired)
        original_tokens = _word_tokens(query)
        assert _is_subsequence(repaired_tokens, original_tokens)
