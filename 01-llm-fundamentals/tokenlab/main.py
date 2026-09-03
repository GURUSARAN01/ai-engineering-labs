from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


def analyze_text(text: str, tokenizer):
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_text = tokenizer.decode(token_ids)

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded_text": decoded_text,
    }


def display_analysis(analysis):
    print("\nTokenLab")
    print("-" * 40)

    print("Original Text:", analysis["text"])
    print("Character Count:", len(analysis["text"]))
    print("Token Count:", analysis["token_count"])

    print("\nTokens:", analysis["tokens"])
    print("Token IDs:", analysis["token_ids"])
    print("\nDecoded Text:", analysis["decoded_text"])

    print("-" * 40)
    print("\nToken Breakdown:")
    print("-" * 40)
    print(f"{'Index':<8}{'Token ID':<12}{'Token'}")
    print("-" * 40)

    for index, (token_id, token) in enumerate(
        zip(analysis["token_ids"], analysis["tokens"])
    ):
        print(f"{index:<8}{token_id:<12}{token}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    text = input("Enter text: ")

    analysis = analyze_text(text, tokenizer)
    display_analysis(analysis)


if __name__ == "__main__":
    main()