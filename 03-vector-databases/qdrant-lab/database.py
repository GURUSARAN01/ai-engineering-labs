from qdrant_client import QdrantClient

QDRANT_URL =  "http://localhost:6333"

def create_client():
    return QdrantClient(url=QDRANT_URL)

