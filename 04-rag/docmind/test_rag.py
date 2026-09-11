from rag import (
    build_context,
    generate_answer,
)

from schemas import (
    RetrievedChunk,
)


def make_chunk():
    return RetrievedChunk(
        document_name=(
            "account_help.txt"
        ),
        chunk_index=0,
        content=(
            "Password reset links "
            "expire after 15 minutes."
        ),
        score=0.91,
    )


def test_build_context():
    context = build_context(
        [make_chunk()]
    )

    assert "[S1]" in context

    assert (
        "account_help.txt"
        in context
    )

    assert (
        "15 minutes"
        in context
    )


class FakeResponse:
    text = """
    {
        "answer":
        "The link expires after 15 minutes [S1].",
        "sources": ["S1"]
    }
    """


class FakeModels:

    def generate_content(
        self,
        **kwargs,
    ):
        return FakeResponse()


class FakeClient:

    def __init__(self):
        self.models = FakeModels()


def test_generate_answer():
    result = generate_answer(
        question=(
            "When does the reset "
            "link expire?"
        ),
        retrieved_chunks=[
            make_chunk()
        ],
        client=FakeClient(),
        model_name="fake-model",
    )

    assert "15 minutes" in (
        result.answer
    )

    assert result.sources == [
        "S1"
    ]