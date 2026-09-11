from pydantic import (Field, BaseModel)

class RetrievedChunk(BaseModel):
    document_name: str

    chunk_index: int

    content: str

    score: float

class RagAnswer(BaseModel):
    answer: str = Field(min_length=1)

    sources: list[str]