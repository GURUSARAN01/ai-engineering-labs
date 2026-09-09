from database import create_connection
from embeddings import (
    embed_text,
    load_embedding_model,
)
from repository import (
    create_documents_table,
    insert_document,
    search_documents,
    search_documents_by_category,
)

DOCUMENTS = [
    (
        "I forgot my password and "
        "cannot log in.",
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

def main():
    connection = create_connection()
    model = load_embedding_model()

    create_documents_table(connection)

    for content, category in DOCUMENTS:
        embedding = embed_text(content, model)
        insert_document(connection=connection, category=category, content=content, embedding=embedding)
    query = input("Enter search query: ")

    query_embedding = embed_text(query, model,)

    results = search_documents(
    connection=connection,
    query_embedding=query_embedding,
    top_k=3,
    min_score=0.3,

    results = search_documents_by_category(
    connection=connection,
    query_embedding=query_embedding,
    category="account",
    top_k=3,
)
)
    
    print("\npgvector Search")
    print("-"*60)

    for rank, row in enumerate(results, start=1):
        document_id, content, category, score = row

        print(
            f"{rank}. "
            f"{score:.4f} | "
            f"{content}"
        )

        print(
            f"    id={document_id}, "
            f"category={category}"
        )

        connection.close()

if __name__ == "__main__":
    main()