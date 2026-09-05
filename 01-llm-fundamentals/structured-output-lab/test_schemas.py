import pytest

from pydantic import ValidationError

from schemas import ArticleAnalysis

def test_valid_article_analysis():
    analysis = ArticleAnalysis(
        summary="A new AI model was released.",
        category="technology",
        topics=["AI", "LLMs"],
        importance=4,
    )

    assert analysis.summary == "A new AI model was released."
    assert analysis.category == "technology"
    assert analysis.topics == ["AI", "LLMs"]
    assert analysis.importance == 4

def test_importance_above_five_is_rejected():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="A test article.",
            category="technology",
            topics=["AI"],
            importance=10,
        )

def test_importance_below_one_is_rejected():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="A test article.",
            category="technology",
            topics=["AI"],
            importance=0,
        )

def test_invalid_category_is_rejected():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="A test article.",
            category="food",
            topics=["AI"],
            importance=4,
        )

def test_empty_summary_is_rejected():
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="",
            category="technology",
            topics=["AI"],
            importance=3,
        )