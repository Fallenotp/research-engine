from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


_TEST_ROOT = Path(tempfile.mkdtemp(prefix="research-engine-test-"))
_TEST_DATA_DIR = _TEST_ROOT / "data"
_TEST_CACHE_DIR = _TEST_ROOT / "cache"
_PRODUCTION_DATA_DIR = Path.home() / ".research_engine"
_PRODUCTION_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _is_production_path(value: str | None, production_path: Path) -> bool:
    if not value:
        return True
    return Path(value).expanduser().resolve() == production_path.resolve()


def _set_test_paths() -> None:
    if _is_production_path(
        os.environ.get("RESEARCH_ENGINE_DATA_DIR"), _PRODUCTION_DATA_DIR
    ):
        os.environ["RESEARCH_ENGINE_DATA_DIR"] = str(_TEST_DATA_DIR)
    if _is_production_path(
        os.environ.get("RESEARCH_ENGINE_CACHE_DIR"), _PRODUCTION_CACHE_DIR
    ):
        os.environ["RESEARCH_ENGINE_CACHE_DIR"] = str(_TEST_CACHE_DIR)
    Path(os.environ["RESEARCH_ENGINE_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["RESEARCH_ENGINE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


_set_test_paths()


@pytest.fixture(autouse=True)
def restore_test_paths() -> None:
    _set_test_paths()
    # Import lazily so collection does not depend on extractor import order.
    from research_engine import extractor

    extractor.reset_process_state()
    yield
    _set_test_paths()
