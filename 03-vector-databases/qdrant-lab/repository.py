from qdrant_client import models


COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 384


def create_collection(client):
    exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )

    if exists:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

def upsert_documents(client, documents, embeddings):
    points=[]
    for index, (document, embedding,) in enumerate(zip(documents, embeddings), start=1):
        content, category = document

        points.append(models.PointStruct(id=index, 
                                         vector=embedding.tolist(), 
                                         payload={
                                             "content": content,
                                             "category": category
                                             }))
    client.upsert(collection_name=COLLECTION_NAME,
                  points=points,
                  wait=True,)

def search_documents(
    client,
    query_embedding,
    top_k: int = 3,
):
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )

    return response.points

def search_by_category(
    client,
    query_embedding,
    category: str,
    top_k: int = 3,
):
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(
                        value=category
                    ),
                )
            ]
        ),
        with_payload=True,
        limit=top_k,
    )

    return response.points

