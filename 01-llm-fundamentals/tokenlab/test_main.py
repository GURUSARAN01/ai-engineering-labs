from transformers import AutoTokenizer
from main import analyze_text, MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def test_analyze_text():
    result = analyze_text("AI Engineering is awesome!", tokenizer)

    assert result["text"] == "AI Engineering is awesome!"
    assert result["token_count"] == len(result['token_ids'])
    assert result["token_count"] == len(result['tokens'])
    assert result["decoded_text"] == result['text']


def test_empty_text():
    result = analyze_text("", tokenizer)

    assert result["text"] == ""
    assert result["token_count"] == 0
    assert result["token_ids"] == []
    assert result["tokens"] == []
    assert result["decoded_text"] == ""