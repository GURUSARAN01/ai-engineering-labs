from pgvector import Vector

EMBEDDING_DIMENSION = 384

def create_documents_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT UNIQUE NOT NULL,
            category TEXT,
            embedding VECTOR(384) NOT NULL
        )
        """
    )

    connection.commit()

def insert_document(
    connection,
    content: str,
    category: str,
    embedding,
):
    connection.execute(
        """
        INSERT INTO documents (
            content,
            category,
            embedding
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (content)
        DO NOTHING
        """,
        (
            content,
            category,
            Vector(embedding),
        ),
    )

    connection.commit()


def search_documents_by_category(
    connection,
    query_embedding,
    category: str,
    top_k: int = 3,
):
    rows = connection.execute(
        """
        SELECT
            id,
            content,
            category,
            1 - (embedding <=> %s) AS similarity
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (
            Vector(query_embedding),
            category,
            Vector(query_embedding),
            top_k,
        ),
    ).fetchall()

    return rows


def search_documents(
    connection,
    query_embedding,
    top_k: int = 3,
    min_score: float | None = None,
):
    rows = connection.execute(
        """
        SELECT
            id,
            content,
            category,
            1 - (
                embedding <=> %s
            ) AS similarity
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (
            Vector(query_embedding),
            Vector(query_embedding),
            top_k,
        ),
    ).fetchall()

    if min_score is not None:
        rows = [
            row
            for row in rows
            if row[3] >= min_score
        ]

    return rows


def create_benchmark_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL
        )
        """
    )

    connection.commit()