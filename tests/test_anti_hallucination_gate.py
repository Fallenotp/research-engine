from __future__ import annotations

from research_engine.anti_hallucination_gate import validate


def test_validate_keeps_clean_cited_input() -> None:
    text = "OpenAI announced the release on May 1. https://example.com/release"

    validated, dropped, flagged = validate(text)

    assert validated == text
    assert dropped == 0
    assert flagged == []


def test_validate_flags_hedging_without_source() -> None:
    text = "The feature likely shipped this week."

    validated, dropped, flagged = validate(text)

    assert validated == ""
    assert dropped == 1
    assert len(flagged) == 1
    assert "hedging_without_source" in flagged[0]
    assert "likely" in flagged[0]


def test_validate_known_false_negative_common_verb() -> None:
    """Pin a known false-negative case: a factual claim using a common verb
    not in VERBISH_RE (e.g. 'makes') with no digit bypasses the gate silently.
    This test asserts the CURRENT observed behavior (dropped == 0) so that any
    future change to _looks_like_claim() or VERBISH_RE that alters this outcome
    will cause a test failure and force an explicit decision."""
    text = "The team makes good things every day."

    validated, dropped, flagged = validate(text, source_required=True)

    assert dropped == 0, (
        "Known false-negative: 'makes' is not in VERBISH_RE so the gate "
        "treats this as a non-claim. If this assertion fails, VERBISH_RE was "
        "widened — verify the new behavior is intentional and update this test."
    )


def test_validate_partially_flags_mixed_input() -> None:
    text = (
        "The launch was confirmed today. https://example.com/launch. "
        "It might reach all users soon. "
        'The company said "rollout starts now."'
    )

    validated, dropped, flagged = validate(text)

    assert "confirmed today" in validated
    assert "rollout starts now" in validated
    assert "might reach all users" not in validated
    assert dropped == 1
    assert len(flagged) == 1
    assert "hedging_without_source" in flagged[0]
