import pytest
import numpy as np

from repository import search_documents


class FakeQueryResponse:
    def __init__(self):
        self.points = [
            FakePoint()
        ]


class FakePoint:
    score = 0.91

    payload = {
        "content": "Forgot password.",
        "category": "account",
    }


class FakeClient:
    def __init__(self):
        self.received_query = None
        self.received_limit = None

    def query_points(
        self,
        collection_name,
        query,
        limit,
        with_payload,
    ):
        self.received_query = query
        self.received_limit = limit

        return FakeQueryResponse()




def test_search_documents():
    client = FakeClient()

    query_embedding = np.array(
        [1.0, 0.0, 0.0]
    )

    results = search_documents(
        client=client,
        query_embedding=query_embedding,
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0].payload["category"]
        == "account"
    )

    assert client.received_limit == 1   

def test_invalid_top_k():
    client = FakeClient()

    with pytest.raises(ValueError):
        search_documents(
            client=client,
            query_embedding=np.array(
                [1.0, 0.0]
            ),
            top_k=0,
        )