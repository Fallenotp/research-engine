from __future__ import annotations

import re

import pytest

from research_engine.research_cli import (
    example_seeking_subquestions,
    generated_example_queries,
)


QUESTIONS = (
    "which UK companies make organic cotton t-shirts",
    "name three hospitals using ambient AI scribes",
    "list examples of councils that banned pavement parking",
)


@pytest.mark.parametrize("question", QUESTIONS)
def test_generated_queries_do_not_leak_hardcoded_topics(question: str) -> None:
    for query in generated_example_queries(question):
        lowered = query.lower()
        assert "uber" not in lowered
        assert "techcrunch" not in lowered
        assert "forbes" not in lowered


@pytest.mark.parametrize("question", QUESTIONS)
def test_every_generated_query_is_derived_from_question(question: str) -> None:
    question_words = set(re.findall(r"[a-z]{4,}", question.lower()))

    for query in generated_example_queries(question):
        query_words = set(re.findall(r"[a-z]{4,}", query.lower()))
        assert question_words & query_words


def test_generated_queries_preserve_quoted_phrase() -> None:
    queries = generated_example_queries(
        'which companies are called the "Netflix of fitness"'
    )

    assert any('"Netflix of fitness"' in query for query in queries)


def test_example_seeking_subquestions_honours_count_without_duplicates() -> None:
    question = "which UK companies make organic cotton t-shirts"

    queries = example_seeking_subquestions(question, 6, [])

    assert len(queries) == 6
    assert len(queries) == len(set(queries))


def test_caller_supplied_candidates_come_first() -> None:
    question = "which UK companies make organic cotton t-shirts"

    queries = example_seeking_subquestions(question, 6, ["CANDIDATE ONE"])

    assert queries[0] == "CANDIDATE ONE"
