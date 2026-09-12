from research_engine.windows import (
    Window,
    bm25_scores,
    rank_candidates,
    render_windows,
    select_windows,
    tokenize,
)


def test_tokenize_basic() -> None:
    assert tokenize("Hello, WORLD! A I 7 x2 don't") == ["hello", "world", "x2", "don"]


def test_bm25_prefers_matching_doc() -> None:
    scores = bm25_scores("solar energy", ["solar energy storage", "football results"])
    equal_scores = bm25_scores("solar", ["solar panels", "solar panels"])

    assert scores[0] > scores[1]
    assert equal_scores[0] == equal_scores[1]


def _result(number: int, title: str = "", snippet: str = "") -> dict[str, str]:
    return {"url": f"https://example.test/{number}", "title": title, "snippet": snippet}


def test_rank_candidates_moves_answering_result_from_position_9_to_top_3() -> None:
    results = [_result(number, "unrelated", "other topic") for number in range(10)]
    results[8] = _result(8, "Rare orchid conservation", "Rare orchid habitats")

    ranked = rank_candidates("rare orchid", results)

    assert ranked.index(next(item for item in ranked if item["url"].endswith("/8"))) < 3


def test_rank_candidates_is_stable_on_ties() -> None:
    results = [_result(number, f"title {number}") for number in range(4)]

    ranked = rank_candidates("missing term", results, exploration_slots=0)

    assert [item["url"] for item in ranked] == [item["url"] for item in results]


def test_rank_candidates_keeps_exploration_slot() -> None:
    results = [_result(0, "low overlap", "unrelated")]
    results.extend(_result(number, "climate policy", "climate policy results") for number in range(1, 8))

    ranked = rank_candidates("climate policy", results)

    assert next(index for index, item in enumerate(ranked) if item["url"].endswith("/0")) < 5


def test_rank_candidates_penalises_listicle() -> None:
    results = [
        _result(0, "Solar energy", "Solar energy basics"),
        {
            **_result(1, "Solar energy", "Solar energy basics"),
            "listicle_flagged": True,
        },
    ]

    ranked = rank_candidates("solar energy", results, exploration_slots=0)

    assert ranked[0]["url"].endswith("/0")


def test_rank_candidates_returns_every_input_once() -> None:
    results = [_result(number, "topic", "topic") for number in range(11)]

    ranked = rank_candidates("topic", results)

    assert len(ranked) == len(results)
    assert {item["url"] for item in ranked} == {item["url"] for item in results}


def _numbered_text(paragraph_count: int = 8, paragraph_chars: int = 1500) -> str:
    paragraphs = []
    for number in range(paragraph_count):
        prefix = f"paragraph-{number} "
        paragraphs.append(prefix + ("x" * (paragraph_chars - len(prefix))))
    return "\n\n".join(paragraphs)


def test_select_windows_offsets_are_exact() -> None:
    full_text = _numbered_text()

    windows = select_windows("paragraph", full_text, budget_chars=12_000)

    assert len(full_text) > 12_000
    assert all(
        full_text[window.char_offset : window.char_offset + window.char_length]
        == window.text
        for window in windows
    )


def test_select_windows_respects_budget() -> None:
    windows = select_windows("paragraph", _numbered_text(), budget_chars=3_100)

    assert sum(window.char_length for window in windows) <= 3_100


def test_select_windows_picks_matching_paragraph_not_the_start() -> None:
    paragraphs = [f"background paragraph {number}" for number in range(7)]
    paragraphs.append("The sought aurora mechanism is magnetic reconnection.")
    full_text = "\n\n".join(paragraphs)

    windows = select_windows(
        "aurora magnetic reconnection", full_text, budget_chars=100, window_chars=100
    )

    assert any("magnetic reconnection" in window.text for window in windows)
    assert all("background paragraph 0" not in window.text for window in windows)


def test_select_windows_returns_document_order() -> None:
    full_text = "first target.\n\nsecond target.\n\nthird target."

    windows = select_windows("target", full_text, budget_chars=100, window_chars=20)

    assert [window.char_offset for window in windows] == sorted(
        window.char_offset for window in windows
    )


def test_select_windows_long_paragraph_is_split() -> None:
    full_text = "z" * 5_000

    windows = select_windows("", full_text, budget_chars=5_000, window_chars=1_500)

    assert len(windows) == 4
    assert all(window.char_length <= 1_500 for window in windows)
    assert all(
        full_text[window.char_offset : window.char_offset + window.char_length]
        == window.text
        for window in windows
    )


def test_render_windows_joins_with_marker() -> None:
    windows = [Window("one", 0, 3, 1.0), Window("two", 5, 3, 0.5)]

    assert render_windows(windows) == "one\n[...]\ntwo"


def test_empty_inputs() -> None:
    assert rank_candidates("question", []) == []
    assert select_windows("question", "") == []
