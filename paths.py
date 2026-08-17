from __future__ import annotations

import logging
import os
import shutil
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
