from pathlib import Path

from chunking import chunk_text
from embeddings import embed_texts
from repository import (
    replace_document_chunks,
)


def load_text_documents(
    data_directory: str,
):
    directory = Path(
        data_directory
    )

    if not directory.exists():
        raise ValueError(
            f"Directory does not exist: "
            f"{data_directory}"
        )

    files = sorted(
        directory.glob("*.txt")
    )

    if not files:
        raise ValueError(
            "No .txt documents found."
        )

    documents = []

    for file_path in files:

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            (
                file_path.name,
                text,
            )
        )

    return documents


def index_documents(
    connection,
    model,
    data_directory: str,
):
    documents = load_text_documents(
        data_directory
    )

    total_chunks = 0

    for (
        document_name,
        text,
    ) in documents:

        chunks = chunk_text(
            text=text,
            chunk_size=120,
            overlap=30,
        )

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = embed_texts(
            texts,
            model,
        )

        replace_document_chunks(
            connection=connection,
            document_name=document_name,
            chunks=chunks,
            embeddings=embeddings,
        )

        total_chunks += len(
            chunks
        )

        print(
            f"Indexed "
            f"{document_name}: "
            f"{len(chunks)} chunks"
        )

    print(
        f"\nTotal chunks indexed: "
        f"{total_chunks}"
    )