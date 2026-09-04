import pytest
from gemini_client import generate

class FakeUsageMetadata:
    prompt_token_count = 8
    candidates_token_count = 12
    thoughts_token_count = None
    total_token_count = 20

class FakeResponse:
    text = "AI is machine intelligence."
    usage_metadata = FakeUsageMetadata()

class FakeModels:
    def __init__(self):
        self.received_model = None
        self.received_contents = None
    def generate_content(self, model, contents):
        self.received_model = model
        self.received_contents = contents
        return FakeResponse()

class FakeClient:
    def __init__(self):
        self.models = FakeModels()

def test_generate_rejects_empty_prompt():
    client = FakeClient()

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        generate(
            prompt="   ",
            client=client,
            model_name="fake-model",
        )


def test_generate_returns_expected_results():
    client = FakeClient()

    result = generate(
        prompt = "What is AI?",
        client = client,
        model_name = "fake-model"
    )

    assert result["prompt"] == "What is AI?"
    assert result["response"] == "AI is machine intelligence."
    assert result["model"] == "fake-model"
    assert client.models.received_model == "fake-model"
    assert client.models.received_contents == "What is AI?"

    assert result["input_tokens"] == 8
    assert result["output_tokens"] == 12
    assert result["thinking_tokens"] == 0
    assert result["total_tokens"] == 20

    assert result["latency_ms"] >= 0