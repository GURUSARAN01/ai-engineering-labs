from pgvector import Vector

from schemas import RetrievedChunk

def create_chunks_table(connection,):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        rag_chunks(
            id BIGSERIAL PRIMARY KEY,
            document_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding Vector(384) NOT NULL,
            UNIQUE(document_name, chunk_index)
            )"""
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        rag_chunks_embedding_hnsw_idx
        ON rag_chunks
        USING hnsw( embedding vector_cosine_ops)
        """
    )

    connection.commit()

def replace_document_chunks(
    connection,
    document_name,
    chunks,
    embeddings,
):
    if len(chunks) != len(
        embeddings
    ):
        raise ValueError(
            "Chunks and embeddings "
            "must have equal lengths."
        )

    with connection.transaction():

        connection.execute(
            """
            DELETE FROM rag_chunks
            WHERE document_name = %s
            """,
            (document_name,),
        )

        rows = [
            (
                document_name,
                chunk["chunk_index"],
                chunk["content"],
                Vector(embedding),
            )
            for chunk, embedding
            in zip(
                chunks,
                embeddings,
            )
        ]

        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO rag_chunks (
                    document_name,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )


def search_chunks(
    connection,
    query_embedding,
    top_k: int = 4,
):
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    rows = connection.execute(
        """
        SELECT
            document_name,
            chunk_index,
            content,
            1 - (
                embedding <=> %s
            ) AS similarity
        FROM rag_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (
            Vector(query_embedding),
            Vector(query_embedding),
            top_k,
        ),
    ).fetchall()

    return [
        RetrievedChunk(
            document_name=row[0],
            chunk_index=row[1],
            content=row[2],
            score=float(row[3]),
        )
        for row in rows
    ]