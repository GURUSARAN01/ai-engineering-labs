import math


TOKENS = ["Paris", "London", "banana"]
LOGITS = [4.0, 2.0, 1.0]


def softmax(logits):
    max_logit = max(logits)

    exponentials = [
        math.exp(logit - max_logit)
        for logit in logits
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


def apply_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    return [
        logit / temperature
        for logit in logits
    ]


def main():
    temperatures = [0.5, 1.0, 2.0]

    for temperature in temperatures:
        adjusted_logits = apply_temperature(
            LOGITS,
            temperature,
        )

        probabilities = softmax(adjusted_logits)

        print(f"\nTemperature: {temperature}")
        print("-" * 30)

        for token, probability in zip(
            TOKENS,
            probabilities,
        ):
            print(f"{token}: {probability:.4f}")

        print(
            "Total:",
            round(sum(probabilities), 4),
        )


if __name__ == "__main__":
    main()