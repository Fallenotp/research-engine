from unittest.mock import patch

from research_engine.politeness import Politeness


def test_robots_blocks_disallowed_path() -> None:
    politeness = Politeness(min_interval_s=0.0)
    robots = "User-agent: *\nDisallow: /private"

    with patch.object(politeness, "_fetch_robots", return_value=robots):
        assert politeness.allowed("https://site.com/public") is True
        assert politeness.allowed("https://site.com/private/x") is False


def test_rate_limit_spaces_requests() -> None:
    politeness = Politeness(min_interval_s=0.05)
    state = {"now": 100.0, "slept": 0.0}

    with patch.object(politeness, "_now", side_effect=lambda: state["now"]), patch.object(
        politeness,
        "_sleep",
        side_effect=lambda seconds: state.__setitem__("slept", seconds),
    ):
        politeness.wait("site.com")
        assert state["slept"] == 0.0
        politeness.wait("site.com")
        assert state["slept"] > 0.0
