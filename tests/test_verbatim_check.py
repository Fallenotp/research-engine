from research_engine.verbatim_check import check_verbatim


def test_number_substring_is_not_support() -> None:
    result = check_verbatim("Revenue was 120.", ["Revenue was 1200."])

    assert result.unsupported_count == 1
    assert result.supported_count == 0


def test_thousands_separator_normalises() -> None:
    result = check_verbatim("Revenue was 1,200.", ["Revenue was 1200 units."])

    assert result.supported_count == 1


def test_currency_and_percent_normalise() -> None:
    result = check_verbatim("It rose 5% to $3.2 million.", ["rose 5 percent to $3.2m"])

    assert result.supported_count == 2


def test_quoted_span_keeps_substring_semantics() -> None:
    result = check_verbatim(
        'She said "we will appeal".', ["Statement: we will appeal the ruling."]
    )

    assert result.supported_count == 1


def test_empty_sources_is_not_applicable() -> None:
    result = check_verbatim("It rose 5%.", [])

    assert result.applicable is False
    assert result.pass_rate is None
    assert result.unsupported_count == 0


def test_no_tokens_is_not_applicable() -> None:
    result = check_verbatim("Hello world.", ["anything"])

    assert result.applicable is False
    assert result.pass_rate is None
