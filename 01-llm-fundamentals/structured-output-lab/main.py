from schemas import ArticleAnalysis
from extractor import create_client, get_model, extract_article


def display_result(result):
    analysis = result["analysis"]
    print("\nExtractAI")
    print("-" * 50)
    print("\nSummary:")
    print(analysis.summary)
    print("\nCategory:")
    print(analysis.category)
    print("\nTopics:")  
    for topic in analysis.topics:
        print(f"- {topic}")
    print("\nImportance")
    print(f"{analysis.importance} / 5")
    print("\nModel:")
    print(result["model"])

    print("\nLatency")
    print(f"{result['latency_ms']:.2f} ms")

    print("-"*50)

def main():
    client = create_client()
    model_name = get_model()
    text = input("Enter the input: ")

    result = extract_article(text=text, client=client, model_name=model_name)
    display_result(result)


if __name__ == "__main__":
        main()