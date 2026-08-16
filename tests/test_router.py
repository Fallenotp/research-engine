from __future__ import annotations

from research_engine import paths
from research_engine.router import load_router


def test_gov_tech_transfer_query_routes_to_nasa_lane() -> None:
    router = load_router()

    decision = router.route("Find NASA software patent license opportunities")

    assert decision.rule_name == "gov_tech_transfer"
    assert decision.lanes == [
        "nasa_techtransfer",
        "github_code",
        "searxng_general",
        "exa_direct",
    ]
    assert decision.worker_model == "haiku"
    assert decision.topic == "gov_tech"


def test_router_loads_nasa_lane_and_data_gov_auth_config() -> None:
    router = load_router()

    nasa = router.lane_endpoint("nasa_techtransfer")
    congress = router.lane_endpoint("congress_gov")
    govinfo = router.lane_endpoint("govinfo_gov")
    fda = router.lane_endpoint("fda_gov")

    assert nasa["type"] == "api"
    assert nasa["endpoint"] == "https://technology.nasa.gov/api/query/{stream}/{query}"
    assert nasa["auth"] == "none"

    for lane in (congress, govinfo):
        assert lane["auth"] == "env:DATA_GOV_API_KEY"
        assert "{DATA_GOV_API_KEY}" in lane["endpoint"]
    assert fda["auth"] == "env:FDA_API_KEY"
    assert "{FDA_API_KEY}" in fda["endpoint"]


def test_scout_lane_routes_through_agy_cli_not_retired_gemini_cli() -> None:
    router = load_router()

    scout_lane = router.lane_endpoint("gemini_pro_scout")
    scout_config = router.scout_config()

    assert scout_lane["command"] == (paths.executable(paths.AGY_BIN_ENV, "agy-cli-1", "agy-cli-2", "agy") or "agy")
    assert scout_lane["command"].rsplit("/", 1)[-1] != "gemini"
    assert scout_lane.get("home") is None
    assert scout_config["provider"] == "agy_cli"
    assert scout_config["cli_home"] == scout_lane["command"]


def test_paid_lane_notes_mark_firecrawl_and_paid_proxy_as_paid_fallbacks() -> None:
    router = load_router()

    paid_proxy = router.lane_endpoint("paid_proxy")
    exa = router.lane_endpoint("exa_direct")
    firecrawl = router.lane_endpoint("firecrawl_direct")

    assert "paid" in paid_proxy["notes"].lower()
    assert "free" in exa["notes"].lower()
    assert "must not be reported as a paid burn" in exa["notes"].lower()
    assert "free keyless mcp.exa.ai" in exa["notes"].lower()
    assert "paid firecrawl fallback" in firecrawl["notes"].lower()
    assert "not a free lane" in firecrawl["notes"].lower()


def test_bluesky_lane_uses_buzz_http_search_not_dead_jetstream() -> None:
    router = load_router()

    bluesky = router.lane_endpoint("bluesky_jetstream")

    assert bluesky["type"] == "cli"
    assert bluesky["command"]
    assert "--search=bluesky" in bluesky["args"]
    assert "jetstream" not in bluesky.get("endpoint", "").lower()
    assert "working http bluesky search" in bluesky["notes"].lower()


def test_router_does_not_match_keywords_inside_larger_words() -> None:
    router = load_router()

    regressions = {
        "what did the newspaper say about the tariffs": "academic_paper",
        "is the billionaire tax working": "legislation",
        "best debugging tools for python": "error_diagnosis",
    }

    for question, bad_rule in regressions.items():
        assert router.route(question).rule_name != bad_rule


def test_router_matches_words_plurals_phrases_and_catch_all() -> None:
    router = load_router()

    expected_routes = {
        "supreme court ruling on EPA": "federal_court_ruling",
        "what do the papers say": "academic_paper",
        "how to use python asyncio": "code_pattern",
        "10-k filing for Apple": "sec_filing",
        "random question no keywords": "general_web",
    }

    for question, expected_rule in expected_routes.items():
        assert router.route(question).rule_name == expected_rule


def test_error_diagnosis_beats_weaker_code_pattern_match() -> None:
    decision = load_router().route("error in my code, how to fix this exception")

    assert decision.rule_name == "error_diagnosis"
    assert "stack_exchange" in decision.lanes


def test_reddit_congress_question_merges_both_rules_lanes() -> None:
    decision = load_router().route("reddit reaction to the new congress bill")

    assert "reddit_rss" in decision.lanes
    assert "congress_gov" in decision.lanes
    assert "legislation" in decision.contributing_rules
    assert "social_sentiment" in decision.contributing_rules


def test_breaking_sec_question_records_both_contributing_rules() -> None:
    decision = load_router().route("breaking news today on the SEC filing")

    assert "breaking_news" in decision.contributing_rules
    assert "sec_filing" in decision.contributing_rules


def test_free_general_lanes_are_appended_when_missing() -> None:
    decision = load_router().route("reddit community reaction")

    assert "searxng_general" in decision.lanes
    assert "exa_direct" in decision.lanes


def test_paid_lane_is_not_auto_appended() -> None:
    decision = load_router().route("error exception")

    assert "paid_proxy" not in decision.lanes


def test_merged_lanes_are_capped_at_ten() -> None:
    decision = load_router().route(
        "breaking reddit reaction today to congress bill sec filing error exception"
    )

    assert len(decision.lanes) <= 10


def test_catch_all_still_routes_unmatched_question() -> None:
    decision = load_router().route("random question with no keywords at all")

    assert decision.rule_name == "general_web"


def test_real_keyword_rule_always_beats_general_web() -> None:
    decision = load_router().route("reddit")

    assert decision.rule_name == "social_sentiment"
    assert decision.contributing_rules[0] == "social_sentiment"
    assert decision.contributing_rules == ["social_sentiment"]
