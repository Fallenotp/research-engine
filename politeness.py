"""Shared rate limiting and robots.txt checks."""

from __future__ import annotations

import logging
import os
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

from . import paths

ROBOTS_TTL_S = 3600
RESPECT_ROBOTS_ENV = "RESEARCH_RESPECT_ROBOTS"
ROBOTS_UNREADABLE_STATUSES = frozenset({401, 403, 407, 429, 451, 500, 502, 503, 504})

logger = logging.getLogger("politeness")


def respect_robots() -> bool:
    """Default False: match the crawl4ai/firecrawl rungs, which never check robots.
    Set RESEARCH_RESPECT_ROBOTS=1 to make the four checking rungs honour it again."""
    return os.environ.get(RESPECT_ROBOTS_ENV, "") == "1"


class Politeness:
    def __init__(self, min_interval_s: float = 2.0):
        self._min_interval = min_interval_s
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = {}
        # Unreadable robots.txt still fails open but is now recorded.
        self.robots_unreadable: dict[str, int] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _note_unreadable(self, domain: str, status: int, detail: str) -> None:
        self.robots_unreadable[domain] = status
        logger.warning(
            "robots.txt unreadable for %s (status=%s, %s); treating as no rules",
            domain,
            status,
            detail,
        )

    def _fetch_robots(self, domain: str) -> str:
        try:
            response = requests.get(
                f"https://{domain}/robots.txt",
                headers={"User-Agent": paths.user_agent()},
                timeout=15,
            )
        except requests.RequestException as exc:
            self._note_unreadable(domain, 0, f"{type(exc).__name__}: {exc}")
            return ""
        if response.status_code == 200:
            self.robots_unreadable.pop(domain, None)
            return response.text
        if response.status_code in ROBOTS_UNREADABLE_STATUSES:
            self._note_unreadable(domain, response.status_code, "refused")
        return ""

    def allowed(self, url: str) -> bool:
        domain = urlparse(url).netloc
        cached = self._robots.get(domain)
        now = self._now()

        if cached is None or now - cached[0] > ROBOTS_TTL_S:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(self._fetch_robots(domain).splitlines())
            cached = (now, parser)
            self._robots[domain] = cached

        return cached[1].can_fetch(paths.user_agent(), url)

    def wait(self, domain: str) -> None:
        last_hit = self._last_hit.get(domain)
        now = self._now()

        if last_hit is not None:
            elapsed = now - last_hit
            if elapsed < self._min_interval:
                self._sleep(self._min_interval - elapsed)

        self._last_hit[domain] = self._now()
