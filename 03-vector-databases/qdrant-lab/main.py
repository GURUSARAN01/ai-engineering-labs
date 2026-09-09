from database import create_client

from embeddings import (
    embed_text,
    embed_texts,
    load_embedding_model,
)

from repository import (
    create_collection,
    search_documents,
    upsert_documents,
    search_by_category,
)


DOCUMENTS = [
    (
        "I forgot my password "
        "and cannot log in.",
        "account",
    ),
    (
        "Python is a programming language.",
        "programming",
    ),
    (
        "You can change your password "
        "from account settings.",
        "account",
    ),
    (
        "The football match begins tonight.",
        "sports",
    ),
]


def display_results(
    query,
    results,
):
    print("\nQdrant Search")
    print("-" * 60)

    print("Query:")
    print(query)

    print("\nResults:")

    for rank, point in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{point.score:.4f} | "
            f"{point.payload['content']}"
        )

        print(
            "   category="
            f"{point.payload['category']}"
        )

    print("-" * 60)


def main():
    client = create_client()
    model = load_embedding_model()

    create_collection(client)

    print(
        "Collection exists:",
        client.collection_exists(
            collection_name="documents"
        ),
    )

    texts = [
        content
        for content, _ in DOCUMENTS
    ]

    embeddings = embed_texts(
        texts,
        model,
    )

    upsert_documents(
        client=client,
        documents=DOCUMENTS,
        embeddings=embeddings,
    )

    query = input(
        "Enter search query: "
    )

    query_embedding = embed_text(
        query,
        model,
    )

    results = search_by_category(
    client=client,
    query_embedding=query_embedding,
    category="account",
    top_k=3,
)

    display_results(
        query,
        results,
    )
    

if __name__ == "__main__":
    main()