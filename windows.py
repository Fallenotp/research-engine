"""Deterministic lexical ranking and document-window selection."""

from dataclasses import dataclass
import math
import re


DEFAULT_BATCH_SIZE = 5
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_PARAGRAPH_RE = re.compile(r"\S.*?(?=\n[ \t]*\n|\Z)", re.DOTALL)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")


@dataclass(frozen=True)
class Window:
    text: str
    char_offset: int
    char_length: int
    score: float


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens with at least two characters."""
    return [token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 2]


def bm25_scores(
    query: str, docs: list[str], *, k1: float = 1.2, b: float = 0.75
) -> list[float]:
    """Score documents against a query with standard BM25."""
    if not docs:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(docs)

    tokenized_docs = [tokenize(document) for document in docs]
    average_length = sum(len(document) for document in tokenized_docs) / len(docs)
    document_frequencies = {
        term: sum(term in set(document) for document in tokenized_docs)
        for term in set(query_terms)
    }
    scores: list[float] = []

    for document in tokenized_docs:
        counts = {term: document.count(term) for term in set(query_terms)}
        length_normalizer = 1 - b
        if average_length:
            length_normalizer += b * len(document) / average_length
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(docs) - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            score += inverse_frequency * frequency * (k1 + 1) / (
                frequency + k1 * length_normalizer
            )
        scores.append(score)
    return scores


def rank_candidates(
    query: str,
    results: list[dict],
    *,
    exploration_slots: int = 1,
    penalise_listicle: bool = True,
) -> list[dict]:
    """Rank results lexically, reserving slots for strong search-engine ranks."""
    if not results:
        return []

    documents = [f"{result.get('title', '')} {result.get('snippet', '')}" for result in results]
    scores = bm25_scores(query, documents)
    ranked = []
    for original_rank, (result, score) in enumerate(zip(results, scores)):
        adjusted_score = score
        if penalise_listicle and result.get("listicle_flagged") is True:
            adjusted_score *= 0.5
        ranked.append((original_rank, adjusted_score, result))
    ranked.sort(key=lambda item: -item[1])

    explored = _add_exploration_slots(ranked, exploration_slots)
    output: list[dict] = []
    for _, score, result in explored:
        copied_result = dict(result)
        copied_result["score"] = score
        output.append(copied_result)
    return output


def _add_exploration_slots(
    ranked: list[tuple[int, float, dict]], exploration_slots: int
) -> list[tuple[int, float, dict]]:
    """Place the best original ranks outside each batch's score slots at its end."""
    if exploration_slots <= 0:
        return ranked

    remaining = list(ranked)
    explored: list[tuple[int, float, dict]] = []
    while remaining:
        batch_size = min(DEFAULT_BATCH_SIZE, len(remaining))
        slot_count = min(exploration_slots, batch_size)
        score_count = batch_size - slot_count
        score_winners = remaining[:score_count]
        candidates = remaining[score_count:]
        explorers = sorted(candidates, key=lambda item: item[0])[:slot_count]
        explorer_ids = {id(item) for item in explorers}
        remaining = [
            item
            for item in remaining[score_count:]
            if id(item) not in explorer_ids
        ]
        explored.extend(score_winners)
        explored.extend(explorers)
    return explored


def select_windows(
    query: str,
    full_text: str,
    *,
    budget_chars: int = 4000,
    window_chars: int = 1500,
) -> list[Window]:
    """Select paragraph-aligned text windows within a strict character budget."""
    if not full_text or budget_chars <= 0:
        return []
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")

    spans = _window_spans(full_text, window_chars)
    texts = [full_text[start:end] for start, end in spans]
    scores = bm25_scores(query, texts)
    if tokenize(query):
        selection_order = sorted(range(len(spans)), key=lambda index: (-scores[index], index))
    else:
        selection_order = list(range(len(spans)))

    selected: list[Window] = []
    remaining_budget = budget_chars
    for index in selection_order:
        start, end = spans[index]
        length = end - start
        if length > remaining_budget:
            continue
        selected.append(Window(texts[index], start, length, scores[index]))
        remaining_budget -= length
    return sorted(selected, key=lambda window: window.char_offset)


def _window_spans(full_text: str, window_chars: int) -> list[tuple[int, int]]:
    """Create contiguous paragraph windows while retaining source-text offsets."""
    units: list[tuple[int, int]] = []
    for paragraph in _PARAGRAPH_RE.finditer(full_text):
        units.extend(_split_long_span(full_text, paragraph.start(), paragraph.end(), window_chars))

    windows: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end in units:
        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= window_chars:
            current_end = end
        else:
            windows.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))
    return windows


def _split_long_span(
    text: str, start: int, end: int, window_chars: int
) -> list[tuple[int, int]]:
    """Split one paragraph at sentence ends, falling back to exact hard splits."""
    if end - start <= window_chars:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > window_chars:
        limit = cursor + window_chars
        sentence_end = None
        for match in _SENTENCE_END_RE.finditer(text, cursor, limit):
            sentence_end = match.end()
        split_at = sentence_end or limit
        spans.append((cursor, split_at))
        cursor = split_at
    spans.append((cursor, end))
    return spans


def render_windows(windows: list[Window], *, separator: str = "\n[...]\n") -> str:
    """Render selected windows with an explicit omission marker."""
    return separator.join(window.text for window in windows)
