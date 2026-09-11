TEMPLATES = {
    "account": [
        "The user forgot their password.",
        "The account cannot be accessed.",
        "Login authentication has failed.",
        "The password needs to be reset.",
        "The user cannot sign into their profile.",
    ],
    "programming": [
        "Python code processes a list of values.",
        "The application contains a software bug.",
        "The developer created a new API.",
        "The program uses a database connection.",
        "The code runs inside a Python environment.",
    ],
    "sports": [
        "The football team won the match.",
        "The player scored during the game.",
        "The coach prepared the team for competition.",
        "Fans watched the championship match.",
        "The athlete trained before the tournament.",
    ],
    "science": [
        "Researchers conducted a scientific experiment.",
        "Scientists studied a new material.",
        "The research team analyzed experimental data.",
        "A laboratory measured the physical properties.",
        "The study reported new scientific findings.",
    ],
    "travel": [
        "The traveler booked a hotel room.",
        "The flight arrived at the airport.",
        "Tourists explored the city.",
        "The passenger purchased an airline ticket.",
        "The journey included several destinations.",
    ],
}

def generate_documents(
    count: int = 5000,
):
    categories = list(TEMPLATES)

    documents = []

    for index in range(count):
        category = categories[
            index % len(categories)
        ]

        templates = TEMPLATES[category]

        template_index = (
            index // len(categories)
        ) % len(templates)

        text = (
            f"{templates[template_index]} "
            f"Reference document {index}."
        )

        documents.append(
            {
                "id": index + 1,
                "text": text,
                "category": category,
            }
        )

    return documents

QUERIES = [
    "I forgot my password.",
    "I cannot sign into my account.",
    "How do I reset my login credentials?",
    "My Python application has a bug.",
    "I am building a software API.",
    "How do I connect my program to a database?",
    "Who won the football game?",
    "The athlete is preparing for a tournament.",
    "Scientists performed a laboratory experiment.",
    "Researchers analyzed scientific data.",
    "I need to book a hotel.",
    "My flight is arriving at the airport.",
]