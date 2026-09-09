from database import create_connection

from embeddings import (
    embed_text,
    embed_texts,
    load_embedding_model,
)

from repository import create_benchmark_table

from hnsw_benchmark import (
    create_hnsw_index,
    exact_search,
    explain_hnsw_search,
    generate_documents,
    hnsw_search,
    insert_benchmark_documents,
    recall_at_k,
    timed_search,
)


def main():
    connection = create_connection()
    model = load_embedding_model()

    create_benchmark_table(connection)

    documents = generate_documents(
        count=5000
    )

    texts = [
        content
        for content, _ in documents
    ]

    print(
        f"Preparing {len(documents)} documents..."
    )

    embeddings = embed_texts(
        texts,
        model,
    )

    insert_benchmark_documents(
        connection,
        documents,
        embeddings,
    )

    print("Creating HNSW index...")

    create_hnsw_index(
        connection
    )

    query = (
        "I forgot my password "
        "and cannot access my account."
    )

    query_embedding = embed_text(
        query,
        model,
    )

    exact_results, exact_ms = timed_search(
        exact_search,
        connection,
        query_embedding,
        10,
    )

    print(
        f"\nExact search: "
        f"{exact_ms:.2f} ms"
    )

    for ef_search in [10, 40, 100]:
        results, latency_ms = timed_search(
            hnsw_search,
            connection,
            query_embedding,
            10,
            ef_search,
        )

        recall = recall_at_k(
            exact_results,
            results,
        )

        print(
            f"\nef_search={ef_search}"
        )

        print(
            f"Latency: "
            f"{latency_ms:.2f} ms"
        )

        print(
            f"Recall@10: "
            f"{recall:.2f}"
        )

    explain_hnsw_search(
        connection,
        query_embedding,
    )

    connection.close()


if __name__ == "__main__":
    main()