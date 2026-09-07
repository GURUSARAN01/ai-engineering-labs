from typing import Any

from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    document: Document
    score: float