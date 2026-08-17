from __future__ import annotations

import logging
import os
import re
import shutil
import traceback
from collections.abc import Mapping
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
logger = logging.getLogger(__name__)

DATA_DIR_ENV = "RESEARCH_ENGINE_DATA_DIR"
ENV_FILE_ENV = "RESEARCH_ENGINE_ENV_FILE"
AGY_BIN_ENV = "RESEARCH_ENGINE_AGY_BIN"
GROK_BIN_ENV = "RESEARCH_ENGINE_GROK_BIN"
CODEX_BIN_ENV = "RESEARCH_ENGINE_CODEX_BIN"
AGENT_BROWSER_BIN_ENV = "RESEARCH_ENGINE_AGENT_BROWSER_BIN"
OLLAMA_BIN_ENV = "RESEARCH_ENGINE_OLLAMA_BIN"
PYTHON_BIN_ENV = "RESEARCH_ENGINE_PYTHON_BIN"
CLAUDE_BIN_ENV = "RESEARCH_ENGINE_CLAUDE_BIN"
CLAUDE_HOME_ENV = "RESEARCH_ENGINE_CLAUDE_HOME"
CRAWL4AI_SCRIPT_ENV = "RESEARCH_ENGINE_CRAWL4AI_SCRIPT"
RESEARCH_SESSIONS_DIR_ENV = "RESEARCH_ENGINE_RESEARCH_SESSIONS_DIR"
LANE_ENV_FILE_ENV = "RESEARCH_ENGINE_LANE_ENV_FILE"
MISTRAL_KEYS_FILE_ENV = "RESEARCH_ENGINE_MISTRAL_KEYS_FILE"
APIFY_PROXY_MODULE_ENV = "RESEARCH_ENGINE_APIFY_PROXY_MODULE"
NORDVPN_ENV_FILE_ENV = "RESEARCH_ENGINE_NORDVPN_ENV_FILE"
CONSEQUENCE_TRACKER_ENV_FILE_ENV = "RESEARCH_ENGINE_CONSEQUENCE_TRACKER_ENV_FILE"
OPENCLAW_ENV_FILE_ENV = "RESEARCH_ENGINE_OPENCLAW_ENV_FILE"
CT_API_KEYS_STATE_ENV = "RESEARCH_ENGINE_CT_API_KEYS_STATE_FILE"
NO_BLUFF_TELEMETRY_ENV = "RESEARCH_ENGINE_NO_BLUFF_TELEMETRY_PATH"
BUZZ_SCRIPT_ENV = "RESEARCH_ENGINE_BUZZ_SCRIPT"
SEMANTIC_PROXY_ENV = "RESEARCH_ENGINE_SEMANTIC_PROXY_PATH"
MEMORY_DB_ENV = "RESEARCH_ENGINE_MEMORY_DB"
CLAUDE_MEMORY_GLOB_ENV = "RESEARCH_ENGINE_CLAUDE_MEMORY_GLOB"
OBSIDIAN_GLOB_ENV = "RESEARCH_ENGINE_OBSIDIAN_GLOB"
CONTACT_EMAIL_ENV = "RESEARCH_ENGINE_CONTACT_EMAIL"
USER_AGENT_ENV = "RESEARCH_ENGINE_USER_AGENT"

_DATA_DIR_DEFAULT = Path.home() / ".research_engine"
_DEFAULT_USER_AGENT = "research-engine/1.0"
_MISSING_CONTACT_INFO_LOGGED = False

_URL_OR_ABSOLUTE_PATH_RE = re.compile(
    r"""
    (?P<double_url>
        "(?P<double_scheme>[A-Za-z][A-Za-z0-9+.-]*)://
        (?P<double_url_body>[^\s"]*)"
    )
    |
    (?P<single_url>
        '(?P<single_scheme>[A-Za-z][A-Za-z0-9+.-]*)://
        (?P<single_url_body>[^\s']*)'
    )
    |
    (?P<bare_url>
        (?P<bare_scheme>[A-Za-z][A-Za-z0-9+.-]*)://
        (?P<bare_url_body>[^\s'\"]*)
    )
    |
    (?<![\w:/])
    (?P<quote>['"])
    (?P<quoted_path>/(?:[^/'"\r\n]+/)+[^/'"\r\n]+)
    (?P=quote)
    |
    (?<![\w:/])
    (?P<bare_path>/(?:[^\s/'"]+/)+[^\s/'"]+)
    """,
    re.VERBOSE,
)


def redact_paths(text: str) -> str:
    """Replace absolute filesystem paths with '<path>/<basename>'.

    Logs from this package are public-facing. The basename is kept because it is what
    makes a failure diagnosable; the directories are what identify the machine and its
    user. file:// URLs keep the scheme and host: file://host/Users/me/x becomes
    file://host/<path>/x. URLs using every other scheme are returned byte-identical.

    Known limitations (recorded, not fixed): UNC paths such as
    //nas/Users/alice/secret and percent-encoded paths such as
    %2FUsers%2Falice%2Fsecret are not redacted. Empty segments such as
    /Users//alice//double//slash are not preserved. Windows drive paths in either
    C:/Users/alice/secret or C:\\Users\\alice\\secret form are not redacted because
    this package is POSIX-only. Colon-separated path lists such as
    PYTHONPATH=/a/b:/c/d collapse to a single redaction, losing detail while removing
    identity. Keeping the basename is deliberate, so /Users/alice alone becomes
    <path>/alice even though that basename is the username.
    """

    def replace(match: re.Match[str]) -> str:
        for prefix in ("double", "single", "bare"):
            scheme = match.group(f"{prefix}_scheme")
            if scheme is None:
                continue
            if scheme.lower() != "file":
                return match.group(0)

            body = match.group(f"{prefix}_url_body")
            host, separator, path = body.partition("/")
            if body.startswith("/"):
                host = ""
                path = body
            elif not separator:
                return match.group(0)
            basename = path.rsplit("/", 1)[-1]
            redacted = f"file://{host + '/' if host else ''}<path>/{basename}"
            quote = '"' if prefix == "double" else "'" if prefix == "single" else ""
            return f"{quote}{redacted}{quote}"

        path = match.group("quoted_path") or match.group("bare_path")
        redacted = f"<path>/{path.rsplit('/', 1)[-1]}"
        quote = match.group("quote") or ""
        return f"{quote}{redacted}{quote}"

    return _URL_OR_ABSOLUTE_PATH_RE.sub(replace, text)


def safe_error(exc: BaseException) -> str:
    return redact_paths(str(exc))


def safe_log(
    logger: "logging.Logger",
    level: int,
    msg: str,
    *args: object,
    exc_info: bool = False,
) -> None:
    """Log without ever propagating.

    A diagnostic must never change what the caller does. These call sites replaced
    `except Exception: pass`, which always continued; a raising handler or filter must
    not turn that into an abort.
    Exceptions are redacted automatically so callers never have to remember.
    """
    try:
        format_args: object = (
            args[0]
            if len(args) == 1 and isinstance(args[0], Mapping) and args[0]
            else args
        )
        rendered_msg = msg % format_args if args else msg
        safe_msg = redact_paths(rendered_msg)
        if exc_info:
            safe_msg = f"{safe_msg}\n{redact_paths(traceback.format_exc()).rstrip()}"
        logger.log(level, safe_msg)
    except Exception:  # noqa: BLE001 - a logging failure must never reach the caller
        pass


def package_path(*parts: str) -> Path:
    return PACKAGE_DIR.joinpath(*parts)


def data_dir() -> Path:
    raw = os.environ.get(DATA_DIR_ENV)
    return Path(raw).expanduser() if raw else _DATA_DIR_DEFAULT


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def env_file() -> Path | None:
    raw = os.environ.get(ENV_FILE_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def contact_email() -> str | None:
    raw = os.environ.get(CONTACT_EMAIL_ENV, "").strip()
    return raw or None


def user_agent() -> str:
    override = os.environ.get(USER_AGENT_ENV)
    if override is not None:
        return override

    email = contact_email()
    if email:
        return f"{_DEFAULT_USER_AGENT} (+mailto:{email})"

    global _MISSING_CONTACT_INFO_LOGGED
    if not _MISSING_CONTACT_INFO_LOGGED:
        logger.info(
            "Using default User-Agent without contact info. Set %s for better Crossref/Wayback rate limits.",
            CONTACT_EMAIL_ENV,
        )
        _MISSING_CONTACT_INFO_LOGGED = True
    return _DEFAULT_USER_AGENT


def optional_path(env_var: str) -> Path | None:
    raw = os.environ.get(env_var, "").strip()
    return Path(raw).expanduser() if raw else None


def executable(env_var: str, *names: str) -> str | None:
    override = os.environ.get(env_var, "").strip()
    if override:
        return override
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


def require_executable(env_var: str, *names: str) -> str:
    resolved = executable(env_var, *names)
    if resolved:
        return resolved
    choices = ", ".join(names)
    raise FileNotFoundError(
        f"Required executable not found. Set {env_var} or put one of [{choices}] on PATH."
    )


def telemetry_path(name: str) -> Path:
    return data_path(name)


def home_path(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


def glob_root(pattern: str) -> Path:
    path = Path(pattern).expanduser()
    parts: list[str] = []
    for part in path.parts:
        if any(marker in part for marker in ("*", "?", "[")):
            break
        parts.append(part)
    return Path(*parts)


def missing_config_message(path: Path, env_var: str, *, label: str) -> str:
    return f"{label} is not configured: missing {path}. Set {env_var}."
