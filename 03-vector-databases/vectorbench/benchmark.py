from time import perf_counter

import numpy as np

from metrics import (
    recall_at_k,
    summarize_latencies,
)


def evaluate_backend(
    name: str,
    backend,
    query_embeddings,
    ground_truths,
    top_k: int = 10,
    ef_search: int = 40,
    warmup_runs: int = 3,
    measured_runs: int = 10,
):
    recalls = []
    latencies = []

    for (
        query_embedding,
        expected_ids,
    ) in zip(
        query_embeddings,
        ground_truths,
    ):

        # Warm-up
        for _ in range(
            warmup_runs
        ):
            backend.search(
                query_embedding,
                top_k=top_k,
                ef_search=ef_search,
            )

        # Measured runs
        for _ in range(
            measured_runs
        ):
            start = perf_counter()

            returned_ids = backend.search(
                query_embedding,
                top_k=top_k,
                ef_search=ef_search,
            )

            latency_ms = (
                perf_counter()
                - start
            ) * 1000

            latencies.append(
                latency_ms
            )

            recalls.append(
                recall_at_k(
                    expected_ids,
                    returned_ids,
                )
            )

    latency_metrics = (
        summarize_latencies(
            latencies
        )
    )

    return {
        "backend": name,
        "recall": float(
            np.mean(recalls)
        ),
        "mean_ms": (
            latency_metrics[
                "mean_ms"
            ]
        ),
        "p50_ms": (
            latency_metrics[
                "p50_ms"
            ]
        ),
        "p95_ms": (
            latency_metrics[
                "p95_ms"
            ]
        ),
    }