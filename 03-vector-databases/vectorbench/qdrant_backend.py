import os

from dotenv import load_dotenv
from qdrant_client import (
    QdrantClient,
    models,
)


load_dotenv()


COLLECTION_NAME = "vectorbench"
VECTOR_SIZE = 384


class QdrantBackend:

    def __init__(self):
        qdrant_url = os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.client = QdrantClient(
            url=qdrant_url
        )

    def setup(
    self,
    documents,
    embeddings,
):
        if self.client.collection_exists(
        collection_name=COLLECTION_NAME
    ):
            self.client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        self.client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=64,
            full_scan_threshold=10,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10,
        ),
    )

        batch_size = 500

        for start in range(
        0,
        len(documents),
        batch_size,
    ):
            end = start + batch_size

            document_batch = documents[
            start:end
        ]

            embedding_batch = embeddings[
            start:end
        ]

            points = [
            models.PointStruct(
                id=document["id"],
                vector=embedding.tolist(),
                payload={
                    "content": document["text"],
                    "category": document["category"],
                },
            )
            for document, embedding
            in zip(
                document_batch,
                embedding_batch,
            )
        ]

            self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_embedding,
        top_k: int = 10,
        ef_search: int = 40,
    ):
        response = (
            self.client.query_points(
                collection_name=(
                    COLLECTION_NAME
                ),
                query=(
                    query_embedding.tolist()
                ),
                search_params=(
                    models.SearchParams(
                        hnsw_ef=ef_search,
                        exact=False,
                    )
                ),
                limit=top_k,
                with_payload=False,
            )
        )

        return [
            int(point.id)
            for point
            in response.points
        ]