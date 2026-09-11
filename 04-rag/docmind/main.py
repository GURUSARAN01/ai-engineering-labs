import argparse

from config import (
    get_gemini_api_key,
    get_gemini_model,
)

from database import create_connection

from embeddings import (
    embed_text,
    load_embedding_model,
)

from ingestion import index_documents

from rag import (
    create_gemini_client,
    generate_answer,
)

from repository import (
    create_chunks_table,
    search_chunks,
)


DATA_DIRECTORY = "data"


def run_index():
    connection = create_connection()

    model = load_embedding_model()

    create_chunks_table(
        connection
    )

    print("\nIndexing documents...\n")

    index_documents(
        connection=connection,
        model=model,
        data_directory=DATA_DIRECTORY,
    )

    connection.close()


def run_question(
    question,
):
    connection = create_connection()

    model = load_embedding_model()

    create_chunks_table(
        connection
    )

    query_embedding = embed_text(
        question,
        model,
    )

    retrieved_chunks = search_chunks(
        connection=connection,
        query_embedding=query_embedding,
        top_k=4,
    )

    print("\nRetrieved Context")
    print("=" * 60)

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        print(
            f"\n[S{index}] "
            f"{chunk.document_name} "
            f"(chunk "
            f"{chunk.chunk_index})"
        )

        print(
            f"Similarity: "
            f"{chunk.score:.4f}"
        )

        print(
            chunk.content
        )

    client = create_gemini_client(
        get_gemini_api_key()
    )

    answer = generate_answer(
        question=question,
        retrieved_chunks=(
            retrieved_chunks
        ),
        client=client,
        model_name=(
            get_gemini_model()
        ),
    )

    print("\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        answer.answer
    )

    print(
        "\nSources:",
        ", ".join(
            answer.sources
        ),
    )

    connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="DocMind RAG"
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    subparsers.add_parser(
        "index",
        help="Index documents",
    )

    ask_parser = (
        subparsers.add_parser(
            "ask",
            help=(
                "Ask a question "
                "about documents"
            ),
        )
    )

    ask_parser.add_argument(
        "question",
        type=str,
    )

    args = parser.parse_args()

    if args.command == "index":
        run_index()

    elif args.command == "ask":
        run_question(
            args.question
        )


if __name__ == "__main__":
    main()