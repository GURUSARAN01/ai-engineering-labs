from gemini_client import (
    create_client,
    generate,
    get_model_name,
)


def display_result(result):
    print("\nInferenceLab")
    print("-" * 50)

    print("Model:", result["model"])
    print(f"Latency: {result['latency_ms']:.2f} ms")

    if result["error"]:
        print("\nRequest failed:")
        print(result["error"])
        print("-" * 50)
        return

    print("\nToken Usage")
    print("Input Tokens:", result["input_tokens"])
    print("Output Tokens:", result["output_tokens"])
    print("Thinking Tokens:", result["thinking_tokens"])
    print("Total Tokens:", result["total_tokens"])

    print("\nResponse:")
    print(result["response"])

    print("-" * 50)
    

def main():
    client = create_client()
    model_name = get_model_name()

    prompt = input(
        "Enter the prompt: "
    )

    result = generate(
        prompt=prompt,
        client=client,
        model_name=model_name,
    )

    display_result(result)


if __name__ == "__main__":
    main()