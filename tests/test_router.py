from __future__ import annotations

from research_engine.router import load_router


def test_gov_tech_transfer_query_routes_to_nasa_lane() -> None:
    router = load_router()

    decision = router.route("Find NASA software patent license opportunities")

    assert decision.rule_name == "gov_tech_transfer"
    assert decision.lanes == [
        "nasa_techtransfer",
        "github_code",
        "searxng_general",
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

    assert scout_lane["command"] == "agy-cli-1"
    assert scout_lane.get("home") is None
    assert scout_config["provider"] == "agy_cli"
    assert scout_config["cli_home"] == "agy-cli-1"


def test_paid_lane_notes_mark_exa_and_firecrawl_as_paid_fallbacks() -> None:
    router = load_router()

    paid_proxy = router.lane_endpoint("paid_proxy")
    exa = router.lane_endpoint("exa_direct")
    firecrawl = router.lane_endpoint("firecrawl_direct")

    assert "paid" in paid_proxy["notes"].lower()
    assert "paid exa backup" in exa["notes"].lower()
    assert "free keyless mcp.exa.ai" in exa["notes"].lower()
    assert "paid firecrawl fallback" in firecrawl["notes"].lower()
    assert "not a free lane" in firecrawl["notes"].lower()


def test_bluesky_lane_uses_buzz_http_search_not_dead_jetstream() -> None:
    router = load_router()

    bluesky = router.lane_endpoint("bluesky_jetstream")

    assert bluesky["type"] == "cli"
    assert bluesky["command"] == "/Users/cleo/buzz/buzz.py"
    assert "--search=bluesky" in bluesky["args"]
    assert "jetstream" not in bluesky.get("endpoint", "").lower()
    assert "working http bluesky search" in bluesky["notes"].lower()
