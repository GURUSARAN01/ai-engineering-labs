import pytest

from similarity import (
    cosine_similarity,
    dot_product,
    magnitude,
)


def test_dot_product():
    result = dot_product(
        [1, 2, 3],
        [4, 5, 6],
    )

    assert result == 32


def test_magnitude():
    result = magnitude(
        [3, 4]
    )

    assert result == 5


def test_identical_vectors_have_similarity_one():
    result = cosine_similarity(
        [1, 2, 3],
        [1, 2, 3],
    )

    assert result == pytest.approx(1.0)


def test_perpendicular_vectors_have_similarity_zero():
    result = cosine_similarity(
        [1, 0],
        [0, 1],
    )

    assert result == pytest.approx(0.0)


def test_different_dimensions_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1, 2],
            [1, 2, 3],
        )


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(
            [0, 0],
            [1, 1],
        )