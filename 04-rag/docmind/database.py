import psycopg

from pgvector.psycopg import (register_vector,)

from config import get_database_url

def create_connection():
    connection = psycopg.connect(get_database_url())

    connection.execute(
        """
            CREATE EXTENSION IF NOT EXISTS vector
        """
    )

    connection.commit()

    register_vector(connection)

    return connection