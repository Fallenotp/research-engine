#!/usr/bin/env python3
"""Read-only health check for every rung in the extractor reader ladder."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import shutil
import socket
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from research_engine import extractor
from research_engine import paths
from research_engine.fetch_proxy import FIRECRAWL_ENV_VARS
from research_engine.schema import ExtractionMethod


ROOT = paths.PROJECT_DIR
EXTRACTOR_PATH = paths.package_path("extractor.py")
ROUTER_CONFIG_PATH = paths.package_path("router_config.yaml")
REPORT_PATH = paths.package_path("tools", "reader_ladder_report.json")
TEST_URL = "https://example.org"
ENV_PATHS = tuple(path for path in (paths.env_file(),) if path is not None)
APIFY_ENV_NAMES = (
    "APIFY_ACCOUNTS",
    "APIFY_API_KEY",
)
APIFY_INDEXED_PREFIXES = (
    "APIFY_TOKEN_",
    "APIFY_KEY_",
)


@dataclass(frozen=True)
class RungSpec:
    method: str
    helper: str
    line: int
    guards: tuple[str, ...]


def parse_env_file(path: Path) -> dict[str, str]:
    """Read dotenv-style assignments without mutating os.environ."""
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
    """Load first-file-wins dotenv values, then overlay the live environment."""
    values: dict[str, str] = {}
    for path in ENV_PATHS:
        for name, value in parse_env_file(path).items():
            values.setdefault(name, value)
    values.update(os.environ)
    return values


def extractor_source_text(path: Path = EXTRACTOR_PATH) -> str:
    return path.read_text(encoding="utf-8")


def extraction_method_value(member_name: str) -> str:
    try:
        return ExtractionMethod[member_name].value
    except KeyError:
        return member_name.lower()


def _method_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Attribute) or node.attr != "value":
        return None
    parent = node.value
    if not isinstance(parent, ast.Attribute):
        return None
    if not isinstance(parent.value, ast.Name) or parent.value.id != "ExtractionMethod":
        return None
    return extraction_method_value(parent.attr)


def _helper_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Lambda):
        return _helper_name(node.body)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def derive_reader_rungs(source_text: str) -> list[RungSpec]:
    """Read extract_clean_text() from source instead of carrying a stale ladder list."""
    module = ast.parse(source_text)
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_clean_text"
        ),
        None,
    )
    if function is None:
        raise ValueError("extract_clean_text not found")

    rungs: list[RungSpec] = []

    def guard_text(node: ast.AST) -> str:
        return ast.get_source_segment(source_text, node) or ast.unparse(node)

    def add_attempt(call: ast.Call, guards: tuple[str, ...]) -> None:
        if not isinstance(call.func, ast.Name) or call.func.id != "_attempt":
            return
        if len(call.args) < 2:
            return
        method = _method_name(call.args[0])
        helper = _helper_name(call.args[1])
        if method and helper:
            rungs.append(
                RungSpec(
                    method=method,
                    helper=helper,
                    line=call.lineno,
                    guards=guards,
                )
            )

    def walk(statements: list[ast.stmt], guards: tuple[str, ...] = ()) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                next_guards = guards + (guard_text(statement.test),)
                walk(statement.body, next_guards)
                walk(statement.orelse, guards)
                continue
            if isinstance(statement, ast.For):
                walk(statement.body, guards)
                walk(statement.orelse, guards)
                continue
            if (
                isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "methods" for target in statement.targets)
                and isinstance(statement.value, ast.List)
            ):
                for entry in statement.value.elts:
                    if not isinstance(entry, ast.Tuple) or len(entry.elts) < 2:
                        continue
                    method = _method_name(entry.elts[0])
                    helper = _helper_name(entry.elts[1])
                    if method and helper:
                        rungs.append(
                            RungSpec(
                                method=method,
                                helper=helper,
                                line=entry.lineno,
                                guards=guards,
                            )
                        )
            value = getattr(statement, "value", None)
            if isinstance(value, ast.Call):
                add_attempt(value, guards)

    walk(function.body)
    return sorted(rungs, key=lambda rung: (rung.line, rung.method, rung.helper))


def load_router_config() -> dict[str, Any]:
    with ROUTER_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def import_status(*modules: str) -> tuple[str, str]:
    problems: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised by monkeypatch in tests
            problems.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if problems:
        return "IMPORT_FAILED", "; ".join(problems)
    return "OK", "import ok: " + ", ".join(modules)


def command_status(*commands: str) -> tuple[str, str]:
    missing: list[str] = []
    found: list[str] = []
    for command in commands:
        if not command:
            continue
        path = Path(command).expanduser()
        exists = path.exists() if path.is_absolute() else shutil.which(command) is not None
        (found if exists else missing).append(str(path if path.is_absolute() else command))
    if missing:
        return "NOT_INSTALLED", "missing command/path: " + ", ".join(missing)
    return "OK", "command exists: " + ", ".join(found)


def service_status(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    if not host or port is None:
        return "SERVICE_DOWN", f"invalid service URL: {url}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
    except OSError as exc:
        return "SERVICE_DOWN", f"{url} ({type(exc).__name__}: {exc})"
    finally:
        sock.close()
    if result != 0:
        return "SERVICE_DOWN", f"{url} (connect_ex={result})"
    return "OK", f"port open: {host}:{port}"


def firecrawl_status(
    env_map: dict[str, str],
    router_config: dict[str, Any],
) -> tuple[str, str]:
    present = [name for name in FIRECRAWL_ENV_VARS if env_map.get(name)]
    if present:
        return "OK", f"direct keys present: {len(present)} ({', '.join(present)})"
    firecrawl_lane = ((router_config.get("lanes") or {}).get("firecrawl_direct") or {})
    endpoint = str(firecrawl_lane.get("endpoint") or "").strip()
    status, detail = service_status(endpoint)
    return status, f"direct keys absent; proxy lane {detail}"


def apify_status(env_map: dict[str, str]) -> tuple[str, str]:
    loaded = 0
    account_blob = env_map.get("APIFY_ACCOUNTS", "").strip()
    if account_blob:
        loaded += sum(1 for chunk in account_blob.split(",") if chunk.strip())
    for idx in range(1, 13):
        if env_map.get(f"APIFY_TOKEN_{idx}") or env_map.get(f"APIFY_KEY_{idx}"):
            loaded += 1
    if env_map.get("APIFY_API_KEY"):
        loaded += 1
    if loaded == 0:
        indexed = ", ".join(f"{prefix}1..12" for prefix in APIFY_INDEXED_PREFIXES)
        detail = f"checked {', '.join(APIFY_ENV_NAMES)}, {indexed}"
        return "MISSING_KEY", detail
    return "OK", f"apify account credentials present: {loaded}"


def jina_status(env_map: dict[str, str]) -> tuple[str, str]:
    return (
        "OK",
        "JINA_API_KEY "
        + ("present (optional)" if env_map.get("JINA_API_KEY") else "absent (optional)"),
    )


def local_text_status() -> tuple[str, str]:
    return "OK", "local-file fallback; no external precondition"


def classify_rung(
    rung: RungSpec,
    env_map: dict[str, str],
    router_config: dict[str, Any],
) -> dict[str, Any]:
    helper = rung.helper
    if helper == "_pdf_docling":
        status, detail = import_status("docling.document_converter")
    elif helper == "_pdf_pymupdf":
        status, detail = import_status("fitz")
    elif helper == "_markitdown_payload":
        status, detail = import_status("markitdown")
    elif helper == "_local_text":
        status, detail = local_text_status()
    elif helper == "_gitingest":
        status, detail = import_status("gitingest")
    elif helper == "_apify_actor_fetch":
        status, detail = apify_status(env_map)
    elif helper == "_trafilatura":
        status, detail = import_status("trafilatura")
    elif helper == "_crawl4ai":
        status, detail = command_status(
            paths.require_executable(paths.PYTHON_BIN_ENV, "python3"),
            str(extractor.CRAWL4AI_SCRIPT),
        )
    elif helper == "_jina":
        status, detail = jina_status(env_map)
    elif helper == "_crawlee_http":
        status, detail = import_status("crawlee.crawlers", "crawlee.proxy_configuration")
    elif helper == "_scrapling_stealth":
        status, detail = import_status("scrapling.fetchers")
    elif helper == "_agent_browser":
        status, detail = command_status(extractor.AGENT_BROWSER_BIN)
    elif helper == "_firecrawl":
        status, detail = firecrawl_status(env_map, router_config)
    elif helper == "try_publisher_fallback":
        status, detail = import_status("research_engine.publisher_fallback")
    elif helper == "try_wayback":
        status, detail = import_status("research_engine.wayback_fallback")
    else:
        status, detail = "OK", f"no explicit precondition rule for helper {helper}"
    return {
        "rung": rung.method,
        "line": str(rung.line),
        "helper": helper,
        "status": status,
        "detail": detail,
        "guards": list(rung.guards),
    }


def read_blocked_log(path: Path = extractor.BLOCKED_LOG_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def summarize_blocked(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts = Counter(str(row.get("domain") or "unknown") for row in rows)
    listicle_field_rows = sum(1 for row in rows if "listicle_flagged" in row)
    listicle_count = sum(1 for row in rows if row.get("listicle_flagged") is True)
    return {
        "events": len(rows),
        "unique_domains": len(domain_counts),
        "listicle_flagged_count": listicle_count if listicle_field_rows else None,
        "listicle_field_rows": listicle_field_rows,
        "top_domains": [
            {"domain": domain, "count": count}
            for domain, count in domain_counts.most_common(5)
        ],
    }


def live_probe() -> dict[str, Any]:
    extractor.clear_blocked_events()
    result = extractor.extract_clean_text(TEST_URL)
    if result is None:
        return {
            "url": TEST_URL,
            "served_by": None,
            "char_count": 0,
            "blocked_events": extractor.blocked_events(),
        }
    return {
        "url": TEST_URL,
        "served_by": str(result.get("extraction_method") or ""),
        "char_count": int(result.get("char_count") or 0),
        "blocked_events": extractor.blocked_events(),
    }


def build_report(*, live: bool = False) -> dict[str, Any]:
    source_text = extractor_source_text()
    env_map = load_env_map()
    router_config = load_router_config()
    rows = [
        classify_rung(rung, env_map, router_config)
        for rung in derive_reader_rungs(source_text)
    ]
    counts = Counter(row["status"] for row in rows)
    blocked_rows = read_blocked_log()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "extractor_path": str(EXTRACTOR_PATH),
        "router_config_path": str(ROUTER_CONFIG_PATH),
        "test_url": TEST_URL,
        "total": len(rows),
        "counts": dict(sorted(counts.items())),
        "rungs": rows,
        "problems": [row for row in rows if row["status"] != "OK"],
        "blocked_summary": summarize_blocked(blocked_rows),
    }
    if live:
        report["live_probe"] = live_probe()
    return report


def print_report(report: dict[str, Any]) -> None:
    rows = report["rungs"]
    headers = ("RUNG", "LINE", "HELPER", "STATUS", "DETAIL")
    widths = [len(header) for header in headers]
    for row in rows:
        values = (row["rung"], row["line"], row["helper"], row["status"], row["detail"])
        widths = [max(width, len(value)) for width, value in zip(widths, values)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        values = (row["rung"], row["line"], row["helper"], row["status"], row["detail"])
        print("  ".join(value.ljust(width) for value, width in zip(values, widths)))

    counts = Counter(row["status"] for row in rows)
    summary = " ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"\nSUMMARY total={len(rows)} {summary}")

    blocked_summary = report["blocked_summary"]
    blocked_line = (
        f"events={blocked_summary['events']} "
        f"unique_domains={blocked_summary['unique_domains']} "
        f"listicle_flagged_count={blocked_summary['listicle_flagged_count']}"
    )
    print(f"BLOCKED {blocked_line}")
    for item in blocked_summary["top_domains"]:
        print(f"- blocked domain {item['domain']}: {item['count']}")

    if report.get("live_probe"):
        probe = report["live_probe"]
        print(
            f"LIVE url={probe['url']} served_by={probe['served_by']} "
            f"char_count={probe['char_count']}"
        )

    print("\nPROBLEMS")
    for row in report["problems"]:
        print(f"- {row['rung']}:{row['line']} [{row['status']}]: {row['detail']}")


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help=f"Fetch {TEST_URL} through the ladder once and report which rung served it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(live=args.live)
    print_report(report)
    write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
