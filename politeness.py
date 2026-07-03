"""Shared rate limiting and robots.txt checks."""

from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urlparse

import requests


USER_AGENT = "IanResearch/1.0 (123icpe@gmail.com)"
ROBOTS_TTL_S = 3600


class Politeness:
    def __init__(self, min_interval_s: float = 2.0):
        self._min_interval = min_interval_s
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _fetch_robots(self, domain: str) -> str:
        response = requests.get(
            f"https://{domain}/robots.txt",
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        return response.text if response.status_code == 200 else ""

    def allowed(self, url: str) -> bool:
        domain = urlparse(url).netloc
        cached = self._robots.get(domain)
        now = self._now()

        if cached is None or now - cached[0] > ROBOTS_TTL_S:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(self._fetch_robots(domain).splitlines())
            cached = (now, parser)
            self._robots[domain] = cached

        return cached[1].can_fetch(USER_AGENT, url)

    def wait(self, domain: str) -> None:
        last_hit = self._last_hit.get(domain)
        now = self._now()

        if last_hit is not None:
            elapsed = now - last_hit
            if elapsed < self._min_interval:
                self._sleep(self._min_interval - elapsed)

        self._last_hit[domain] = self._now()
