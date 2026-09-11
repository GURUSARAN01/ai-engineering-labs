import numpy as np


def exact_top_k(
    query_embedding,
    document_embeddings,
    document_ids,
    top_k: int = 10,
):
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    scores = (
        document_embeddings
        @ query_embedding
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        int(document_ids[index])
        for index in indices
    ]


def recall_at_k(
    expected_ids,
    returned_ids,
):
    if not expected_ids:
        return 0.0

    expected = set(expected_ids)
    returned = set(returned_ids)

    matches = len(
        expected & returned
    )

    return (
        matches
        / len(expected)
    )


def summarize_latencies(
    latencies,
):
    return {
        "mean_ms": float(
            np.mean(latencies)
        ),
        "p50_ms": float(
            np.percentile(
                latencies,
                50,
            )
        ),
        "p95_ms": float(
            np.percentile(
                latencies,
                95,
            )
        ),
    }