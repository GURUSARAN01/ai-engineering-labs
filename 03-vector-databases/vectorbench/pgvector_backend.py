import os

import psycopg

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector


load_dotenv()


class PgVectorBackend:

    def __init__(self):
        database_url = os.getenv(
            "DATABASE_URL"
        )

        if not database_url:
            raise ValueError(
                "DATABASE_URL is missing."
            )

        self.connection = psycopg.connect(
            database_url
        )

        self.connection.execute(
            """
            CREATE EXTENSION
            IF NOT EXISTS vector
            """
        )

        register_vector(
            self.connection
        )

    def setup(
        self,
        documents,
        embeddings,
    ):
        self.connection.execute(
            """
            DROP TABLE IF EXISTS
            vectorbench_documents
            """
        )

        self.connection.execute(
            """
            CREATE TABLE
            vectorbench_documents (
                id BIGINT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                embedding VECTOR(384)
                    NOT NULL
            )
            """
        )

        rows = [
            (
                document["id"],
                document["text"],
                document["category"],
                Vector(embedding),
            )
            for document, embedding in zip(
                documents,
                embeddings,
            )
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO
                    vectorbench_documents (
                        id,
                        content,
                        category,
                        embedding
                    )
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )

        self.connection.execute(
            """
            CREATE INDEX
                vectorbench_hnsw_idx
            ON vectorbench_documents
            USING hnsw (
                embedding vector_cosine_ops
            )
            WITH (
                m = 16,
                ef_construction = 64
            )
            """
        )

        self.connection.execute(
            """
            ANALYZE vectorbench_documents
            """
        )

        self.connection.commit()

    def search(
        self,
        query_embedding,
        top_k: int = 10,
        ef_search: int = 40,
    ):
        with self.connection.transaction():

            self.connection.execute(
                """
                SELECT set_config(
                    'hnsw.ef_search',
                    %s,
                    true
                )
                """,
                (str(ef_search),),
            )

            # Force HNSW for our small benchmark.
            self.connection.execute(
                """
                SET LOCAL enable_seqscan = off
                """
            )

            rows = self.connection.execute(
                """
                SELECT id
                FROM vectorbench_documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    Vector(
                        query_embedding
                    ),
                    top_k,
                ),
            ).fetchall()

        return [
            int(row[0])
            for row in rows
        ]

    def close(self):
        self.connection.close()