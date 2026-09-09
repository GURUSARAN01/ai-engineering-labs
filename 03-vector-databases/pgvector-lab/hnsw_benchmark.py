from time import perf_counter

from pgvector import Vector

from database import create_connection
from embeddings import (
    embed_text,
    embed_texts,
    load_embedding_model,
)
from repository import create_benchmark_table

TEMPLATES = {
    "account": [
        "The user forgot their password.",
        "The account cannot be accessed.",
        "Login authentication has failed.",
        "The password needs to be reset.",
        "The user cannot sign into their profile.",
    ],
    "programming": [
        "Python code processes a list of values.",
        "The application contains a software bug.",
        "The developer created a new API.",
        "The program uses a database connection.",
        "The code runs inside a Python environment.",
    ],
    "sports": [
        "The football team won the match.",
        "The player scored during the game.",
        "The coach prepared the team for competition.",
        "Fans watched the championship match.",
        "The athlete trained before the tournament.",
    ],
    "science": [
        "Researchers conducted a scientific experiment.",
        "Scientists studied a new material.",
        "The research team analyzed experimental data.",
        "A laboratory measured the physical properties.",
        "The study reported new scientific findings.",
    ],
    "travel": [
        "The traveler booked a hotel room.",
        "The flight arrived at the airport.",
        "Tourists explored the city.",
        "The passenger purchased an airline ticket.",
        "The journey included several destinations.",
    ],
}

def generate_documents(
    count: int = 5000,
):
    categories = list(TEMPLATES)

    documents = []

    for index in range(count):
        category_index = (
            index % len(categories)
        )

        category = categories[
            category_index
        ]

        templates = TEMPLATES[
            category
        ]

        template_index = (
            index // len(categories)
        ) % len(templates)

        template = templates[
            template_index
        ]

        content = (
            f"{template} "
            f"Reference document {index}."
        )

        documents.append(
            (content, category)
        )

    return documents

def insert_benchmark_documents(
    connection,
    documents,
    embeddings,
):
    rows = [
        (
            content,
            category,
            Vector(embedding),
        )
        for (
            content,
            category
        ), embedding in zip(
            documents,
            embeddings,
        )
    ]

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO benchmark_documents (
                content,
                category,
                embedding
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (content)
            DO NOTHING
            """,
            rows,
        )

    connection.commit()

def create_hnsw_index(connection):
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
            benchmark_embedding_hnsw_idx
        ON benchmark_documents
        USING hnsw (
            embedding vector_cosine_ops
        )
        WITH (
            m = 16,
            ef_construction = 64
        )
        """
    )

    connection.execute(
        """
        ANALYZE benchmark_documents
        """
    )

    connection.commit()

def exact_search(
    connection,
    query_embedding,
    top_k: int = 10,
):
    with connection.transaction():

        connection.execute(
            """
            SET LOCAL enable_indexscan = off
            """
        )

        rows = connection.execute(
            """
            SELECT
                id,
                content,
                category,
                1 - (
                    embedding <=> %s
                ) AS similarity
            FROM benchmark_documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                Vector(query_embedding),
                Vector(query_embedding),
                top_k,
            ),
        ).fetchall()

    return rows

def hnsw_search(
    connection,
    query_embedding,
    top_k: int = 10,
    ef_search: int = 40,
):
    if ef_search <= 0:
        raise ValueError(
            "ef_search must be greater than 0."
        )

    with connection.transaction():

        connection.execute(
            """
            SELECT set_config(
                'hnsw.ef_search',
                %s,
                true
            )
            """,
            (str(ef_search),),
        )

        rows = connection.execute(
            """
            SELECT
                id,
                content,
                category,
                1 - (
                    embedding <=> %s
                ) AS similarity
            FROM benchmark_documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                Vector(query_embedding),
                Vector(query_embedding),
                top_k,
            ),
        ).fetchall()

    return rows

def recall_at_k(
    exact_results,
    approximate_results,
):
    if not exact_results:
        return 0.0

    exact_ids = {
        row[0]
        for row in exact_results
    }

    approximate_ids = {
        row[0]
        for row in approximate_results
    }

    matches = len(
        exact_ids
        & approximate_ids
    )

    return matches / len(exact_ids)

def timed_search(
    search_function,
    *args,
    **kwargs,
):
    start = perf_counter()

    results = search_function(
        *args,
        **kwargs,
    )

    latency_ms = (
        perf_counter() - start
    ) * 1000

    return results, latency_ms

def explain_hnsw_search(
    connection,
    query_embedding,
):
    rows = connection.execute(
        """
        EXPLAIN (
            ANALYZE,
            BUFFERS
        )
        SELECT
            id,
            content
        FROM benchmark_documents
        ORDER BY embedding <=> %s
        LIMIT 10
        """,
        (
            Vector(query_embedding),
        ),
    ).fetchall()

    print("\nQuery Plan")
    print("-" * 60)

    for row in rows:
        print(row[0])