import os
from time import perf_counter

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")


def create_client():
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to the root .env file."
        )

    return genai.Client(api_key=API_KEY)


def get_model_name():
    if not MODEL_NAME:
        raise ValueError(
            "GEMINI_MODEL is missing. "
            "Add it to the root .env file."
        )

    return MODEL_NAME


def generate(prompt: str, client, model_name: str):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    start = perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

    except errors.ServerError as error:
        if error.code == 503:
            return {
                "prompt": prompt,
                "response": None,
                "model": model_name,
                "latency_ms": (perf_counter() - start) * 1000,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "error": "Model temporarily unavailable due to high demand.",
            }

        raise

    latency_ms = (perf_counter() - start) * 1000
    usage = response.usage_metadata

    return {
        "prompt": prompt,
        "response": response.text,
        "model": model_name,
        "latency_ms": latency_ms,
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
        "thinking_tokens": usage.thoughts_token_count or 0,
        "total_tokens": usage.total_token_count or 0,
        "error": None,
    }