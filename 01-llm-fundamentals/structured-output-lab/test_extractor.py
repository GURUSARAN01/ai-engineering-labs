import pytest
from extractor import extract_article

class FakeResponse:
    text = """
        {
        "summary": "Researchers released an efficient AI model.",
        "category": "technology",
        "topics": ["AI", "language models"],
        "importance": 4
    }
    """ 

class FakeModels:
    def __init__(self):
        self.received_model = None
        self.received_contents = None
        self.received_config = None

    def generate_content(self, model, contents, config):
        self.received_model = model
        self.received_contents = contents
        self.received_config = config

        return FakeResponse()

class FakeClient:
    def __init__(self):
        self.models = FakeModels()

def test_extract_article_returns_structured_analysis():
    client = FakeClient()
    result = extract_article(
        text="Researchers released a new AI model.",
        client=client,
        model_name="fake-model",
    )

    analysis = result['analysis']

    assert analysis.category == "technology"
    assert analysis.importance == 4
    assert analysis.topics == [
        "AI",
        "language models",
    ]

    assert result["model"] == "fake-model"
    assert result["latency_ms"] >= 0

    assert client.models.received_model == "fake-model"
    assert (
        "Researchers released a new AI model."
        in client.models.received_contents
    )

def test_extract_article_rejects_empty_text():
    client = FakeClient()

    with pytest.raises(
        ValueError,
        match="Text cannot be empty",
    ):
        extract_article(
            text="   ",
            client=client,
            model_name="fake-model",
        )