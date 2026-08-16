#!/usr/bin/env python3
"""Run a one-off, read-only health check of every configured research lane."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import requests
import yaml

from research_engine import paths

CONFIG_PATH = paths.package_path("router_config.yaml")
REPORT_PATH = paths.package_path("tools", "lane_health_report.json")
ENV_PATHS = (
    *(path for path in (paths.env_file(),) if path is not None),
)
PAID_LANES = {
    "paid_proxy",
    "linkup_direct",
    "tavily_direct",
    "youcom_direct",
    "firecrawl_direct",
}
QUERY = "climate change"
ENCODED_QUERY = quote(QUERY, safe="")
TIMEOUT_SECONDS = 20
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
UPPER_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


def parse_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without changing the process environment."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values.setdefault(name, value)
    return values


def load_env_map() -> dict[str, str]:
    """Load first-file-wins dotenv values, then overlay the real environment."""
    values: dict[str, str] = {}
    for path in ENV_PATHS:
        for name, value in parse_env_file(path).items():
            values.setdefault(name, value)
    values.update(os.environ)
    return values


def mask_secrets(text: str, env_map: dict[str, str]) -> str:
    """Remove known secret values from diagnostic text."""
    masked = text
    for value in sorted(set(env_map.values()), key=len, reverse=True):
        if len(value) >= 6 and value in masked:
            masked = masked.replace(value, "***")
    return re.sub(
        r"(?i)(api[_-]?key|token|authorization)=([^&\s]+)",
        r"\1=***",
        masked,
    )


def required_env_names(lane: dict[str, Any]) -> set[str]:
    """Find environment names required by templates, auth, and headers."""
    names: set[str] = set()
    for field in ("endpoint", "body_template"):
        names.update(UPPER_PLACEHOLDER_RE.findall(str(lane.get(field, ""))))

    auth = str(lane.get("auth", ""))
    if auth.startswith("env:"):
        names.add(auth[4:])
    for value in (lane.get("headers") or {}).values():
        names.update(re.findall(r"env:([A-Z][A-Z0-9_]*)", str(value)))
    return names


def render_template(template: str, env_map: dict[str, str]) -> str:
    """Render the fixed probe values and uppercase environment placeholders."""
    values = {
        "query": ENCODED_QUERY,
        "stream": "patent",
        "q_clause": ENCODED_QUERY,
        # The config's dod_oss URL also needs this routing placeholder.
        "resource": "repositories",
    }
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{name}}}", value)
    for name in UPPER_PLACEHOLDER_RE.findall(rendered):
        if name in env_map:
            rendered = rendered.replace(f"{{{name}}}", env_map[name])
    return rendered


def resolve_headers(lane: dict[str, Any], env_map: dict[str, str]) -> dict[str, str]:
    """Resolve env references in headers and enforce the probe User-Agent."""
    headers = {"User-Agent": paths.user_agent()}
    for name, raw_value in (lane.get("headers") or {}).items():
        value = str(raw_value)
        for env_name in re.findall(r"env:([A-Z][A-Z0-9_]*)", value):
            value = value.replace(f"env:{env_name}", env_map[env_name])
        headers[str(name)] = value
    headers["User-Agent"] = paths.user_agent()
    return headers


def result_count(body: bytes) -> int | None:
    """Return a recognized result count, or None when the shape is unknown."""
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None

    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        esearch = payload.get("esearchresult")
        if isinstance(esearch, dict) and isinstance(esearch.get("idlist"), list):
            return len(esearch["idlist"])
        for key in ("results", "items", "data", "hits", "papers"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                nested_hits = value.get("hits")
                if isinstance(nested_hits, list):
                    return len(nested_hits)
    if re.search(r"<(?:(?:\w+):)?feed\b", text, re.IGNORECASE):
        return len(re.findall(r"<(?:(?:\w+):)?entry\b", text, re.IGNORECASE))
    return None


def load_searxng_engines() -> tuple[dict[str, bool] | None, str | None]:
    """Read the live SearXNG engine registry once."""
    try:
        response = requests.get(
            "http://localhost:8888/config",
            headers={"User-Agent": paths.user_agent()},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        engines = response.json().get("engines")
        if not isinstance(engines, list):
            return None, "SearXNG config has no engine list"
        return {
            str(engine["name"]): bool(engine.get("enabled"))
            for engine in engines
            if isinstance(engine, dict) and "name" in engine
        }, None
    except requests.RequestException as exc:
        return None, f"SearXNG config {type(exc).__name__}"
    except (ValueError, TypeError) as exc:
        return None, f"SearXNG config {type(exc).__name__}"


def searxng_detail(
    endpoint: str, engines: dict[str, bool] | None, config_error: str | None
) -> str | None:
    """Describe unknown and disabled engines requested by one SearXNG lane."""
    if "localhost:8888" not in endpoint or "engines=" not in endpoint:
        return None
    if engines is None:
        return config_error or "SearXNG config unavailable"
    requested = parse_qs(urlparse(endpoint).query).get("engines", [""])[0].split(",")
    unknown = [name for name in requested if name not in engines]
    disabled = [name for name in requested if name in engines and not engines[name]]
    return f"UNKNOWN={','.join(unknown) or '-'} DISABLED={','.join(disabled) or '-'}"


def check_api_lane(
    lane: dict[str, Any],
    env_map: dict[str, str],
    searx_engines: dict[str, bool] | None,
    searx_error: str | None,
) -> tuple[str, str]:
    """Send one free API probe and classify its response."""
    missing = sorted(
        name for name in required_env_names(lane) if not env_map.get(name)
    )
    if missing:
        return "FAIL", f"missing key {missing[0]}"

    endpoint = render_template(str(lane.get("endpoint", "")), env_map)
    body_template = lane.get("body_template")
    body = (
        render_template(str(body_template), env_map).encode("utf-8")
        if body_template is not None
        else None
    )
    unresolved = sorted(
        set(PLACEHOLDER_RE.findall(endpoint))
        | set(PLACEHOLDER_RE.findall(body.decode() if body else ""))
    )
    if unresolved:
        return "FAIL", f"unresolved placeholder {unresolved[0]}"

    headers = resolve_headers(lane, env_map)
    if body is not None:
        headers["Content-Type"] = "application/json"
    searx = searxng_detail(endpoint, searx_engines, searx_error)
    try:
        response = requests.request(
            method=str(lane.get("method", "GET")).upper(),
            url=endpoint,
            headers=headers,
            data=body,
            timeout=TIMEOUT_SECONDS,
        )
        count = result_count(response.content)
        count_text = "unknown" if count is None else str(count)
        detail = f"HTTP {response.status_code}; results={count_text}"
        if searx:
            detail = f"{detail}; {searx}"
        if not 200 <= response.status_code < 300:
            return "FAIL", detail
        if count == 0:
            return "EMPTY", detail
        return "OK", detail
    except requests.RequestException as exc:
        detail = f"{type(exc).__name__}"
        if searx:
            detail = f"{detail}; {searx}"
        return "FAIL", mask_secrets(detail, env_map)


def glob_root(pattern: str) -> Path:
    """Return the non-wildcard root of a glob pattern."""
    path = Path(pattern).expanduser()
    parts: list[str] = []
    for part in path.parts:
        if any(marker in part for marker in ("*", "?", "[")):
            break
        parts.append(part)
    return Path(*parts)


def check_lane(
    name: str,
    lane: dict[str, Any],
    env_map: dict[str, str],
    searx_engines: dict[str, bool] | None,
    searx_error: str | None,
) -> dict[str, str]:
    """Classify one lane without executing CLIs or paid providers."""
    lane_type = str(lane.get("type", "unknown"))
    if name in PAID_LANES:
        status, detail = "SKIP", "paid lane"
    elif lane_type == "api":
        status, detail = check_api_lane(
            lane, env_map, searx_engines, searx_error
        )
    elif lane_type == "cli":
        command = Path(str(lane.get("command", ""))).expanduser()
        status = "OK" if command.is_file() else "FAIL"
        detail = f"command {'exists' if command.is_file() else 'missing'}: {command}"
    elif lane_type == "local":
        raw_path = lane.get("db_path")
        path = Path(str(raw_path)).expanduser() if raw_path else glob_root(str(lane.get("glob", "")))
        status = "OK" if path.exists() else "CHECK"
        detail = f"path {'exists' if path.exists() else 'missing'}: {path}"
    elif lane_type == "mcp":
        status, detail = "SKIP", f"mcp server {lane.get('mcp_server', 'unknown')}"
    else:
        status, detail = "CHECK", f"unsupported type {lane_type}"
    return {"lane": name, "type": lane_type, "status": status, "detail": detail}


def print_report(rows: list[dict[str, str]]) -> None:
    """Print the complete table, status counts, and non-OK problem rows."""
    headers = ("LANE", "TYPE", "STATUS", "DETAIL")
    widths = [len(header) for header in headers]
    for row in rows:
        values = (row["lane"], row["type"], row["status"], row["detail"])
        widths = [max(width, len(value)) for width, value in zip(widths, values)]
    line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        values = (row["lane"], row["type"], row["status"], row["detail"])
        print("  ".join(value.ljust(width) for value, width in zip(values, widths)))

    counts = Counter(row["status"] for row in rows)
    summary = " ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"\nSUMMARY total={len(rows)} {summary}")
    print("\nPROBLEMS")
    for row in rows:
        if row["status"] != "OK":
            print(f"- {row['lane']} [{row['status']}]: {row['detail']}")


def main() -> int:
    """Run all checks, print the report, and write its JSON equivalent."""
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    lanes = config.get("lanes") or {}
    env_map = load_env_map()
    searx_engines, searx_error = load_searxng_engines()
    rows = [
        check_lane(name, lane, env_map, searx_engines, searx_error)
        for name, lane in lanes.items()
    ]
    print_report(rows)
    counts = Counter(row["status"] for row in rows)
    report = {
        "query": QUERY,
        "total": len(rows),
        "counts": dict(sorted(counts.items())),
        "lanes": rows,
        "problems": [row for row in rows if row["status"] != "OK"],
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
