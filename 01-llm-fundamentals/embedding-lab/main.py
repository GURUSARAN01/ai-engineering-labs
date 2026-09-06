from embeddings import embed_text, embed_texts, load_model
from similarity import cosine_similarity

def main():
    model = load_model()
    query = "How can I reset my password?"
    documents = [
        "I forgot my password and cannot log in.",
        "Python is a programming language.",
        "You can change your password from account settings.",
        "The football match begins tonight.",
    ]

    query_embedding = embed_text(query, model)

    document_embeddings = embed_texts(documents, model)

    print("\nEmbeddingLab")
    print("-" * 60)

    print("Query:")
    print(query)

    print("\nEmbedding dimension:")
    print(len(query_embedding))

    print("\nSimilarities:")

    results = []
    for document, embedding in zip(
        documents,
        document_embeddings,
    ):
        score = cosine_similarity(
            query_embedding,
            embedding,
        )
        results.append(
        {
            "document": document,
            "score": score,
        }
        )
        print(
            f"{score:.4f} | {document}"
        )
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print("\nRanking:")

    for rank, result in enumerate(
    results,
    start=1,
    ):
     print(
        f"{rank}. "
        f"{result['score']:.4f} | "
        f"{result['document']}"
    )
    print("-" * 60)


if __name__ == "__main__":
    main()