import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def create_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing"
                         "Add it to root .env file")

    connection = psycopg.connect(DATABASE_URL)

    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")

    register_vector(connection)

    return connection