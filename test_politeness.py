from unittest.mock import patch

import pytest

from research_engine.politeness import DomainCooldown, Politeness, host_group


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


def test_wait_uses_host_group() -> None:
    politeness = Politeness(min_interval_s=2.0)
    state = {"now": 100.0, "slept": 0.0}

    with patch.object(politeness, "_now", side_effect=lambda: state["now"]), patch.object(
        politeness, "_sleep", side_effect=lambda seconds: state.__setitem__("slept", seconds)
    ):
        politeness.wait("old.reddit.com")
        politeness.wait("www.reddit.com")

    assert state["slept"] == 2.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("user:pass@www.example.com:443", "example.com"),
        ("[2001:db8::1]:443", "2001:db8::1"),
        ("www.old.reddit.com", "reddit.com"),
        ("xn--bcher-kva.example", "xn--bcher-kva.example"),
        ("bücher.example", "xn--bcher-kva.example"),
        ("192.0.2.1:8080", "192.0.2.1"),
    ],
)
def test_host_group_normalizes_all_host_forms(value: str, expected: str) -> None:
    assert host_group(value) == expected


def test_robots_fetch_respects_existing_cooldown() -> None:
    politeness = Politeness(min_interval_s=0.0)
    politeness.note_block("example.com", 429, "30")

    with pytest.raises(DomainCooldown):
        politeness.allowed("https://www.example.com/private")


def test_past_retry_after_uses_default_cooldown() -> None:
    politeness = Politeness(min_interval_s=0.0)
    politeness.note_block("example.com", 429, "Wed, 21 Oct 2015 07:28:00 GMT")

    assert politeness.in_cooldown("example.com") is not None
