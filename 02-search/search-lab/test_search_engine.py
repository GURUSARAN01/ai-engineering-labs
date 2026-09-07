import numpy as np
import pytest

from schemas import Document
from search_engine import (
    InMemoryVectorSearch,
)


class FakeEmbeddingModel:
    def __init__(self):
        self.embeddings = {
            "reset password": [1.0, 0.0, 0.0],
            "forgot password": [0.95, 0.05, 0.0],
            "change password": [0.90, 0.10, 0.0],
            "python": [0.0, 1.0, 0.0],
            "football": [0.0, 0.0, 1.0],
        }

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        if isinstance(texts, str):
            vector = np.array(
                self.embeddings[texts],
                dtype=float,
            )

            return vector / np.linalg.norm(
                vector
            )

        vectors = np.array(
            [
                self.embeddings[text]
                for text in texts
            ],
            dtype=float,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        return vectors / norms

def create_engine():
        model = FakeEmbeddingModel()

        engine = InMemoryVectorSearch(
        model
        )

        documents = [
        Document(
            id="1",
            text="forgot password",
        ),
        Document(
            id="2",
            text="python",
        ),
        Document(
            id="3",
            text="change password",
        ),
        Document(
            id="4",
            text="football",
        ),
    ]

        engine.index_documents(
        documents
    )

        return engine


def test_search_returns_most_similar_document():
        engine = create_engine()

        results = engine.search(
        query="reset password",
        top_k=1,
    )

        assert len(results) == 1

        assert (
        results[0].document.text
        == "forgot password"
    )

def test_search_respects_top_k():
    engine = create_engine()

    results = engine.search(
        query="reset password",
        top_k=2,
    )

    assert len(results) == 2

def test_empty_query_is_rejected():
    engine = create_engine()

    with pytest.raises(ValueError):
        engine.search(
            query="   "
        )

def test_invalid_top_k_is_rejected():
    engine = create_engine()

    with pytest.raises(ValueError):
        engine.search(
            query="reset password",
            top_k=0,
        )

def test_search_before_indexing_is_rejected():
    engine = InMemoryVectorSearch(
        FakeEmbeddingModel()
    )

    with pytest.raises(ValueError):
        engine.search(
            query="reset password"
        )

def test_empty_documents_are_rejected():
    engine = InMemoryVectorSearch(
        FakeEmbeddingModel()
    )

    with pytest.raises(ValueError):
        engine.index_documents([])