from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_engine.schema import GeminiProRunKind, GeminiProRunRecord


def test_successful_gemini_pro_run_requires_model_id() -> None:
    with pytest.raises(ValidationError):
        GeminiProRunRecord(
            run_type=GeminiProRunKind.SCOUT,
            success=True,
        )


def test_failed_gemini_pro_run_requires_failure_reason() -> None:
    with pytest.raises(ValidationError):
        GeminiProRunRecord(
            run_type=GeminiProRunKind.PRO_SYNTHESIS_FALLBACK,
            success=False,
            model_id="gemini-3-flash",
        )


def test_successful_gemini_pro_run_accepts_model_id() -> None:
    record = GeminiProRunRecord(
        run_type=GeminiProRunKind.FINAL_SYNTHESIS,
        success=True,
        model_id="sonnet",
    )

    assert GeminiProRunKind.FINAL_SYNTHESIS.value == "final_synthesis"
    assert record.model_id == "sonnet"
