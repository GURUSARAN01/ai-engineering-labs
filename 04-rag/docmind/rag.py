from google import genai
from google.genai import types

from schemas import RagAnswer

def create_gemini_client(api_key,):
    return genai.Client(api_key=api_key)

def build_context(retrieved_chunks,):
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source_id = f"S{index}"

        context_parts.append(
            (
                f"[{source_id}]\n"
                f"Document: "
                f"{chunk.document_name}\n"
                f"Chunk: "
                f"{chunk.chunk_index}\n"
                f"Content:\n"
                f"{chunk.content}"
            )
        )

    return "\n\n".join(context_parts)

def generate_answer(question, retrieved_chunks, client, model_name):
    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_chunks:
        return RagAnswer(
            answer=(
                "I do not have enough "
                "information in the "
                "provided documents."
            ),
            sources=[],
        )

    context = build_context(
        retrieved_chunks
    )
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied context.

Rules:

1. Do not use outside knowledge.
2. If the context does not contain enough information,
   clearly say that the documents do not provide the answer.
3. Cite supporting context using source IDs such as [S1].
4. Do not invent source IDs.
5. Keep the answer concise.
6. The `sources` field must contain only source IDs
   actually used in the answer.

QUESTION:

{question}

CONTEXT:

{context}
"""

    response = (
        client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=(
                types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type=(
                        "application/json"
                    ),
                    response_schema=(
                        RagAnswer
                    ),
                )
            ),
        )
    )

    return (
        RagAnswer.model_validate_json(
            response.text
        )
    )        