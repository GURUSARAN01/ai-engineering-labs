import pytest

from hnsw_benchmark import (
    recall_at_k,
)


def test_perfect_recall():
    exact = [
        (1, "a"),
        (2, "b"),
        (3, "c"),
    ]

    approximate = [
        (1, "a"),
        (2, "b"),
        (3, "c"),
    ]

    result = recall_at_k(
        exact,
        approximate,
    )

    assert result == pytest.approx(
        1.0
    )


def test_partial_recall():
    exact = [
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
    ]

    approximate = [
        (1, "a"),
        (2, "b"),
        (7, "x"),
        (8, "y"),
    ]

    result = recall_at_k(
        exact,
        approximate,
    )

    assert result == pytest.approx(
        0.5
    )


def test_empty_exact_results():
    assert recall_at_k(
        [],
        [],
    ) == 0.0