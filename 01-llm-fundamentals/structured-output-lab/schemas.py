from typing import Literal

from pydantic import BaseModel, Field

class ArticleAnalysis(BaseModel):
    summary: str = Field(
        min_length=10,
        description="A concise summary of the article.",
    )

    category: Literal[
        "technology",
        "business",
        "science",
        "sports",
        "other",
    ]

    topics: list[str]

    importance: int = Field(
        ge=1,
        le=5,
        description="Importance from 1(low) to 5(high)"
    )