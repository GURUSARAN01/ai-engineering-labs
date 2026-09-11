from benchmark import (
    evaluate_backend,
)

from dataset import (
    QUERIES,
    generate_documents,
)

from embeddings import (
    embed_texts,
    load_embedding_model,
)

from metrics import exact_top_k

from pgvector_backend import (
    PgVectorBackend,
)

from qdrant_backend import (
    QdrantBackend,
)

import time 


DOCUMENT_COUNT = 5000
TOP_K = 10
EF_SEARCH = 40


def display_result(result):
    print(
        f"\n{result['backend']}"
    )

    print("-" * 40)

    print(
        "Recall@10:",
        f"{result['recall']:.3f}",
    )

    print(
        "Mean latency:",
        f"{result['mean_ms']:.3f} ms",
    )

    print(
        "p50 latency:",
        f"{result['p50_ms']:.3f} ms",
    )

    print(
        "p95 latency:",
        f"{result['p95_ms']:.3f} ms",
    )


def main():

    print("\nVectorBench")
    print("=" * 60)

    print(
        f"Generating "
        f"{DOCUMENT_COUNT} documents..."
    )

    documents = generate_documents(
        DOCUMENT_COUNT
    )

    model = load_embedding_model()

    document_texts = [
        document["text"]
        for document in documents
    ]

    document_ids = [
        document["id"]
        for document in documents
    ]

    print(
        "\nEmbedding documents..."
    )

    document_embeddings = (
        embed_texts(
            document_texts,
            model,
        )
    )

    print(
        "\nEmbedding queries..."
    )

    query_embeddings = embed_texts(
        QUERIES,
        model,
    )

    print(
        "\nCalculating NumPy "
        "ground truth..."
    )

    ground_truths = [
        exact_top_k(
            query_embedding=(
                query_embedding
            ),
            document_embeddings=(
                document_embeddings
            ),
            document_ids=(
                document_ids
            ),
            top_k=TOP_K,
        )
        for query_embedding
        in query_embeddings
    ]

    print(
        "\nPreparing pgvector..."
    )

    pgvector = PgVectorBackend()

    pgvector.setup(
        documents,
        document_embeddings,
    )

    print(
        "Preparing Qdrant..."
    )

    qdrant = QdrantBackend()

    qdrant.setup(
        documents,
        document_embeddings,
    )

    print(
        "\nRunning benchmarks..."
    )

    pg_result = evaluate_backend(
        name="PGVECTOR HNSW",
        backend=pgvector,
        query_embeddings=(
            query_embeddings
        ),
        ground_truths=ground_truths,
        top_k=TOP_K,
        ef_search=EF_SEARCH,
    )

    qdrant_result = evaluate_backend(
        name="QDRANT HNSW",
        backend=qdrant,
        query_embeddings=(
            query_embeddings
        ),
        ground_truths=ground_truths,
        top_k=TOP_K,
        ef_search=EF_SEARCH,
    )

    print("\n")
    print("=" * 60)

    display_result(
        pg_result
    )

    display_result(
        qdrant_result
    )

    print("\n")
    print("=" * 60)

    print(
        f"Documents: "
        f"{DOCUMENT_COUNT}"
    )

    print(
        f"Queries: "
        f"{len(QUERIES)}"
    )

    print(
        "Embedding dimension: 384"
    )

    print(
        f"Top-k: {TOP_K}"
    )

    print(
        f"HNSW ef: {EF_SEARCH}"
    )

    pgvector.close()


if __name__ == "__main__":
    main()