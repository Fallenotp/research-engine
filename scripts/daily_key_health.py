#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import http.client
import json
import os
import ssl
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any

from research_engine import paths

ENV_FILE = paths.optional_path(paths.CONSEQUENCE_TRACKER_ENV_FILE_ENV) or paths.data_path(
    "missing-consequence-tracker.env"
)
OPENCLAW_ENV_FILE = paths.optional_path(paths.OPENCLAW_ENV_FILE_ENV) or paths.data_path(
    "missing-openclaw.env"
)
MISTRAL_ENV_FILE = paths.optional_path(paths.MISTRAL_KEYS_FILE_ENV) or paths.data_path(
    "missing-mistral-free-keys.env"
)
COUNTER_FILE = paths.optional_path(paths.CT_API_KEYS_STATE_ENV) or paths.data_path(
    "ct-api-keys.json"
)
STATE_FILE = paths.package_path("scripts", "key-health-state.json")
LOG_FILE = paths.package_path("scripts", "key-health-log.jsonl")
CURL_BIN = "/usr/bin/curl"
LAUNCHCTL_BIN = "/bin/launchctl"
PROXY_LABEL = "com.semantic-search-proxy"
DISCORD_CHANNEL_ID = "1495166380163596298"
REQUEST_TIMEOUT = 15
SLEEP_SECONDS = 0.5

KEYS = [
    ("linkup_1", "linkup", "LINKUP_API_KEY_1", "main", 0),
    ("exa_1", "exa", "EXA_API_KEY", "main", 1),
    ("linkup_2", "linkup", "LINKUP_API_KEY_2", "main", 2),
    ("exa_2", "exa", "EXA_API_KEY_2", "main", 3),
    ("exa_3", "exa", "EXA_API_KEY_3", "main", 4),
    ("linkup_3", "linkup", "LINKUP_API_KEY_3", "main", 5),
    ("exa_4", "exa", "EXA_API_KEY_4", "main", 6),
    ("linkup_4", "linkup", "LINKUP_API_KEY_4", "main", 7),
    ("linkup_5", "linkup", "LINKUP_API_KEY_FRESH", "main", 8),
    ("youcom_1", "youcom", "YOUCOM_API_KEY_1", "main", 9),
    ("youcom_2", "youcom", "YOUCOM_API_KEY_2", "main", 10),
    ("youcom_3", "youcom", "YOUCOM_API_KEY_3", "main", 11),
    ("youcom_4", "youcom", "YOUCOM_API_KEY_4", "main", 12),
    ("tavily_1", "tavily", "TAVILY_API_KEY", "tavily", 0),
    ("tavily_2", "tavily", "TAVILY_API_KEY_2", "tavily", 1),
    ("mistral_1", "mistral", "MISTRAL_FREE_KEY_1", "mistral", 0),
    ("mistral_2", "mistral", "MISTRAL_FREE_KEY_2", "mistral", 1),
    ("mistral_3", "mistral", "MISTRAL_FREE_KEY_3", "mistral", 2),
    ("mistral_4", "mistral", "MISTRAL_FREE_KEY_4", "mistral", 3),
    ("mistral_5", "mistral", "MISTRAL_FREE_KEY_5", "mistral", 4),
    ("mistral_6", "mistral", "MISTRAL_FREE_KEY_6", "mistral", 5),
    ("mistral_7", "mistral", "MISTRAL_FREE_KEY_7", "mistral", 6),
    ("mistral_8", "mistral", "MISTRAL_FREE_KEY_8", "mistral", 7),
    ("mistral_9", "mistral", "MISTRAL_FREE_KEY_9", "mistral", 8),
    ("mistral_10", "mistral", "MISTRAL_FREE_KEY_10", "mistral", 9),
    ("mistral_11", "mistral", "MISTRAL_FREE_KEY_11", "mistral", 10),
    ("mistral_12", "mistral", "MISTRAL_FREE_KEY_12", "mistral", 11),
    ("mistral_13", "mistral", "MISTRAL_FREE_KEY_13", "mistral", 12),
]

CTX = ssl.create_default_context()


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return env
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key] = value
    return env


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def compact_snippet(raw: bytes | str, limit: int = 200) -> str:
    text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    return " ".join(text.split())[:limit]


def classify_status(status: int | None, error: str | None) -> tuple[str, str]:
    if error:
        return "unknown", "UNKNOWN"
    if status is None:
        return "unknown", "UNKNOWN"
    if 200 <= status <= 299:
        return "alive", "ALIVE"
    if status in (401, 403):
        return "dead", "DEAD_AUTH"
    if status == 402:
        return "dead", "DEAD_CREDITS"
    if status == 429:
        return "alive", "RATE_LIMITED"
    if status >= 500:
        return "unknown", "PROVIDER_ERROR"
    return "unknown", "UNKNOWN"


def request(
    host: str,
    path: str,
    method: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> tuple[int | None, str, str | None]:
    conn = http.client.HTTPSConnection(host, timeout=REQUEST_TIMEOUT, context=CTX)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        snippet = compact_snippet(response.read(300))
        return response.status, snippet, None
    except Exception as exc:
        return None, compact_snippet(f"{type(exc).__name__}: {exc}"), str(exc)
    finally:
        conn.close()


def test_key(provider: str, api_key: str) -> tuple[int | None, str, str | None]:
    if provider == "exa":
        body = json.dumps({"query": "test", "numResults": 1}).encode("utf-8")
        return request(
            "api.exa.ai",
            "/search",
            "POST",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body,
        )
    if provider == "linkup":
        body = json.dumps(
            {"q": "test", "depth": "standard", "outputType": "searchResults"}
        ).encode("utf-8")
        return request(
            "api.linkup.so",
            "/v1/search",
            "POST",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body,
        )
    if provider == "youcom":
        query = urllib.parse.urlencode({"query": "test", "count": 1})
        return request(
            "ydc-index.io",
            f"/v1/search?{query}",
            "GET",
            {"X-API-Key": api_key},
        )
    if provider == "mistral":
        return request(
            "api.mistral.ai",
            "/v1/models",
            "GET",
            {"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
    body = json.dumps(
        {"api_key": api_key, "query": "test", "max_results": 1}
    ).encode("utf-8")
    return request(
        "api.tavily.com",
        "/search",
        "POST",
        {"Content-Type": "application/json"},
        body,
    )


def load_discord_token(primary_env: dict[str, str]) -> str:
    if primary_env.get("DISCORD_BOT_TOKEN"):
        return primary_env["DISCORD_BOT_TOKEN"]
    return load_env_file(OPENCLAW_ENV_FILE).get("DISCORD_BOT_TOKEN", "")


def revive_proxy_keys(results: list[dict[str, Any]]) -> tuple[list[str], bool]:
    state = read_json(COUNTER_FILE, {})
    if not isinstance(state, dict):
        return [], False

    dead_keys = {int(i) for i in state.get("dead_keys", [])}
    tavily_dead_keys = {int(i) for i in state.get("tavily_dead_keys", [])}
    revived_proxy_keys: list[str] = []

    for item in results:
        if item["state"] != "alive":
            continue
        if item["queue"] == "main" and item["queue_index"] in dead_keys:
            dead_keys.remove(item["queue_index"])
            revived_proxy_keys.append(item["name"])
        if item["queue"] == "tavily" and item["queue_index"] in tavily_dead_keys:
            tavily_dead_keys.remove(item["queue_index"])
            revived_proxy_keys.append(item["name"])

    if not revived_proxy_keys:
        return [], False

    state["dead_keys"] = sorted(dead_keys)
    state["tavily_dead_keys"] = sorted(tavily_dead_keys)
    atomic_write_json(COUNTER_FILE, state)

    restart = subprocess.run(
        [LAUNCHCTL_BIN, "kickstart", "-k", f"gui/{os.getuid()}/{PROXY_LABEL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return revived_proxy_keys, restart.returncode == 0


def send_discord_alert(
    token: str,
    newly_dead: list[dict[str, Any]],
    revived: list[dict[str, Any]],
    proxy_revived: list[str],
    summary: dict[str, int],
) -> tuple[bool, str]:
    if not token or (not newly_dead and not revived):
        return False, "skipped"

    lines = [
        "Daily key health report",
        (
            f"Alive: {summary['alive_count']} | Dead: {summary['dead_count']} | "
            f"Unknown: {summary['unknown_count']}"
        ),
    ]
    if newly_dead:
        lines.append("")
        lines.append("Newly dead:")
        lines.extend(
            f"- {item['name']} ({item['provider']}) status={item['status_code']} {item['verdict']}"
            for item in newly_dead
        )
    if revived:
        lines.append("")
        lines.append("Revived:")
        lines.extend(
            f"- {item['name']} ({item['provider']}) status={item['status_code']} {item['verdict']}"
            for item in revived
        )
    if proxy_revived:
        lines.append("")
        lines.append(f"Proxy dead-list cleanup: {', '.join(proxy_revived)}")
    content = "\n".join(lines)

    result = subprocess.run(
        [
            CURL_BIN,
            "-sS",
            "-X",
            "POST",
            f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
            "-H",
            f"Authorization: Bot {token}",
            "-F",
            f"content={content}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    ok = result.returncode == 0 and '"id"' in result.stdout
    detail = result.stdout.strip() or result.stderr.strip() or f"exit={result.returncode}"
    return ok, detail


def main() -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = load_env_file(ENV_FILE)
    env.update(load_env_file(MISTRAL_ENV_FILE))
    previous = read_json(STATE_FILE, {})
    previous_results = previous.get("results", {}) if isinstance(previous, dict) else {}

    results: list[dict[str, Any]] = []
    for index, (name, provider, env_var, queue, queue_index) in enumerate(KEYS):
        api_key = env.get(env_var, "")
        status, snippet, error = test_key(provider, api_key) if api_key else (None, "missing key", "missing key")
        state, verdict = classify_status(status, error)
        previous_state = previous_results.get(name, {}).get("state")
        delta = "no_change"
        if previous_state == "alive" and state == "dead":
            delta = "newly_dead"
        elif previous_state == "dead" and state == "alive":
            delta = "revived"
        results.append(
            {
                "name": name,
                "provider": provider,
                "env_var": env_var,
                "queue": queue,
                "queue_index": queue_index,
                "status_code": status,
                "state": state,
                "verdict": verdict,
                "response_snippet": snippet,
                "error": error,
                "tested_at": now,
                "previous_state": previous_state,
                "delta": delta,
            }
        )
        if index < len(KEYS) - 1:
            time.sleep(SLEEP_SECONDS)

    newly_dead = [item for item in results if item["delta"] == "newly_dead"]
    revived = [item for item in results if item["delta"] == "revived"]
    alive_count = sum(item["state"] == "alive" for item in results)
    dead_count = sum(item["state"] == "dead" for item in results)
    unknown_count = sum(item["state"] == "unknown" for item in results)

    proxy_revived, proxy_restart_ok = revive_proxy_keys(results)
    summary = {"alive_count": alive_count, "dead_count": dead_count, "unknown_count": unknown_count, "newly_dead_count": len(newly_dead), "revived_count": len(revived)}
    state_payload = {"timestamp": now, "summary": summary, "proxy_dead_list_revived": proxy_revived, "proxy_restart_ok": proxy_restart_ok, "results": {item["name"]: item for item in results}}
    atomic_write_json(STATE_FILE, state_payload)
    append_jsonl(LOG_FILE, {"timestamp": now, **summary})

    alert_ok = False
    alert_detail = "skipped"
    if newly_dead or revived:
        alert_ok, alert_detail = send_discord_alert(
            load_discord_token(env), newly_dead, revived, proxy_revived, summary
        )

    print(
        "PASS "
        f"alive={alive_count} dead={dead_count} unknown={unknown_count} "
        f"newly_dead={len(newly_dead)} revived={len(revived)} "
        f"proxy_revived={len(proxy_revived)} discord={'yes' if alert_ok else 'no'}"
    )
    if alert_detail != "skipped":
        print(f"discord_detail={alert_detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
