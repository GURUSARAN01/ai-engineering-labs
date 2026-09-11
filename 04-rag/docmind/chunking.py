


def chunk_text(text: str, chunk_size: int=120, overlap: int=30):
    if not text.strip():
        raise ValueError("Text cannot be empty")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap <= 0:
        raise ValueError("overlap must be greater than zero.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be larger than chunk_size.")

    words = text.split()

    chunks = []

    start=0
    chunk_index = 0

    while start < len(words):
        end = min(start+chunk_size, len(words),)
        content = " ".join(words[start:end])

        chunks.append(
            {
                "chunk_index" : chunk_index,
                "content" : content,
            }
        )
        chunk_index +=1

        if end == len(words):
            break

        start = end - overlap

    return chunks
