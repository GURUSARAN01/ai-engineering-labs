from sentence_transformers import (
    SentenceTransformer,
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def load_embedding_model():
    return SentenceTransformer(
        MODEL_NAME
    )


def embed_text(
    text: str,
    model,
):
    if not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embed_texts(
    texts: list[str],
    model,
):
    if not texts:
        raise ValueError(
            "Texts cannot be empty."
        )

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )