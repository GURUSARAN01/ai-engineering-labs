from schemas import Document
from search_engine import (InMemoryVectorSearch, load_model)


def display_results(query, results):
    print("\nSearch Lab: ")

    print("-" *50)

    print("Query:")
    print(query)

    print("\nResults: ")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result.score:.4f} | "
            f"{result.document.text}"
        )

        if result.document.metadata:
            print("  Metadata:", result.document.metadata,)
    print("-"*50)

def main():
    model = load_model()

    search_engine = InMemoryVectorSearch(model)

    documents = [
        Document(
            id="doc-1",
            text=(
                "I forgot my password "
                "and cannot log in."
            ),
            metadata={
                "category": "account"
            },
        ),
        Document(
            id="doc-2",
            text=(
                "Python is a "
                "programming language."
            ),
            metadata={
                "category": "programming"
            },
        ),
        Document(
            id="doc-3",
            text=(
                "You can change your password "
                "from account settings."
            ),
            metadata={
                "category": "account"
            },
        ),
        Document(
            id="doc-4",
            text=(
                "The football match "
                "begins tonight."
            ),
            metadata={
                "category": "sports"
            },
        ),
    ]

    print("Indexing documents.....")

    search_engine.index_documents(documents=documents)

    query = input("Enter search query: ")

    results = search_engine.search(query=query, top_k=3)

    display_results(query, results)


if __name__ == "__main__":
    main()

