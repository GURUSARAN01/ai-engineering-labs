import pytest

from chunking import chunk_text


def test_short_text_one_chunk():
    chunks = chunk_text(
        "one two three four",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1

    assert (
        chunks[0]["content"]
        == "one two three four"
    )


def test_chunk_overlap():
    chunks = chunk_text(
        (
            "one two three four "
            "five six seven eight"
        ),
        chunk_size=5,
        overlap=2,
    )

    assert (
        chunks[0]["content"]
        == "one two three four five"
    )

    assert (
        chunks[1]["content"]
        == "four five six seven eight"
    )


def test_invalid_overlap():
    with pytest.raises(
        ValueError
    ):
        chunk_text(
            "hello world",
            chunk_size=5,
            overlap=5,
        )