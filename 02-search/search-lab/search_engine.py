import numpy as np

from sentence_transformers import SentenceTransformer

from schemas import Document, SearchResult

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model():
    return SentenceTransformer(MODEL_NAME)

class InMemoryVectorSearch:
    def __init__(self, model):
        self.model = model
        self.documents: list[Document] = []
        self.embeddings = None

    def index_documents(self, documents: list[Document],):
        if not documents:
            raise ValueError("Documents cannot be empty")
        texts = [document.text for document in documents]

        embeddings = self.model.encode(texts, convert_to_numpy = True, normalize_embeddings = True)

        self.documents = documents
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 3, min_score: float | None = None) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <=0:
            raise ValueError("top_k must be greater than 0.")
        if self.embeddings is None:
            raise ValueError("No documents have been indexed.")

        query_embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        scores = (self.embeddings @ query_embedding)

        ranked_indexes = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indexes:
            score = float(scores[index])

            if(min_score is not None and score < min_score):
                continue

            results.append(SearchResult(document=self.documents[int(index)], score=score,))

            if len(results) >= top_k:
                break

        return results