"""Shared rate limiting and robots.txt checks."""

from __future__ import annotations

import logging
import os
import time
import urllib.robotparser
from collections import OrderedDict
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from . import paths

ROBOTS_TTL_S = 3600
RESPECT_ROBOTS_ENV = "RESEARCH_RESPECT_ROBOTS"
ROBOTS_UNREADABLE_STATUSES = frozenset({401, 403, 407, 429, 451, 500, 502, 503, 504})
MAX_STATE_ENTRIES = 4096

logger = logging.getLogger("politeness")


def host_group(domain: str) -> str:
    """Return the shared rate-limit key for a host and its common aliases."""
    parsed = urlparse(domain if "://" in domain else f"//{domain}")
    host = (parsed.hostname or domain).rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    while True:
        for prefix in ("www.", "old.", "new.", "m.", "amp."):
            if host.startswith(prefix):
                host = host[len(prefix) :]
                break
        else:
            return host


class DomainCooldown(Exception):
    def __init__(self, group: str, seconds_remaining: float):
        self.group = group
        self.seconds_remaining = seconds_remaining
        super().__init__(f"{group} in cooldown for {seconds_remaining:.1f}s")


def respect_robots() -> bool:
    """Default False: match the crawl4ai/firecrawl rungs, which never check robots.
    Set RESEARCH_RESPECT_ROBOTS=1 to make the four checking rungs honour it again."""
    return os.environ.get(RESPECT_ROBOTS_ENV, "") == "1"


class Politeness:
    def __init__(self, min_interval_s: float = 2.0):
        self._min_interval = min_interval_s
        self._last_hit: OrderedDict[str, float] = OrderedDict()
        self._robots: OrderedDict[str, tuple[float, urllib.robotparser.RobotFileParser]] = OrderedDict()
        self.cooldown_until: OrderedDict[str, float] = OrderedDict()
        # Unreadable robots.txt still fails open but is now recorded.
        self.robots_unreadable: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _cap(mapping: OrderedDict) -> None:
        while len(mapping) > MAX_STATE_ENTRIES:
            mapping.popitem(last=False)

    def _prune(self) -> None:
        now = self._now()
        for group, until in list(self.cooldown_until.items()):
            if until <= now:
                self.cooldown_until.pop(group, None)
        for domain, (cached_at, _parser) in list(self._robots.items()):
            if now - cached_at > ROBOTS_TTL_S:
                self._robots.pop(domain, None)

    def fetch_gate(self, url: str, *, group_url: str | None = None) -> None:
        group = host_group(group_url or url)
        remaining = self.in_cooldown(group)
        if remaining is not None:
            raise DomainCooldown(group, remaining)
        self.wait(group)

    def _now(self) -> float:
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _note_unreadable(self, domain: str, status: int, detail: str) -> None:
        self.robots_unreadable[domain] = status
        self.robots_unreadable.move_to_end(domain)
        self._cap(self.robots_unreadable)
        logger.warning(
            "robots.txt unreadable for %s (status=%s, %s); treating as no rules",
            domain,
            status,
            detail,
        )

    def _fetch_robots(self, domain: str) -> str:
        self.fetch_gate(f"https://{domain}/robots.txt")
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
        domain = urlparse(url).hostname or urlparse(url).netloc
        cached = self._robots.get(domain)
        if cached is not None:
            self._robots.move_to_end(domain)
        now = self._now()

        if cached is None or now - cached[0] > ROBOTS_TTL_S:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(self._fetch_robots(domain).splitlines())
            cached = (now, parser)
            self._robots[domain] = cached
            self._cap(self._robots)

        return cached[1].can_fetch(paths.user_agent(), url)

    def wait(self, domain: str) -> None:
        self._prune()
        domain = host_group(domain)
        last_hit = self._last_hit.get(domain)
        now = self._now()

        if last_hit is not None:
            elapsed = now - last_hit
            if elapsed < self._min_interval:
                self._sleep(self._min_interval - elapsed)

        self._last_hit[domain] = self._now()
        self._last_hit.move_to_end(domain)
        self._cap(self._last_hit)

    def note_block(self, domain: str, status: int, retry_after: str | None = None) -> None:
        del status  # retained for callers and future status-specific policy.
        seconds = 60.0
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, IndexError, OverflowError):
                    seconds = 60.0
        if seconds <= 0:
            seconds = 60.0
        self._prune()
        group = host_group(domain)
        self.cooldown_until[group] = self._now() + seconds
        self.cooldown_until.move_to_end(group)
        self._cap(self.cooldown_until)

    def in_cooldown(self, domain: str) -> float | None:
        group = host_group(domain)
        until = self.cooldown_until.get(group)
        if until is None:
            return None
        remaining = until - self._now()
        if remaining > 0:
            self.cooldown_until.move_to_end(group)
            return remaining
        self.cooldown_until.pop(group, None)
        return None
