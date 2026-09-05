import os
from time import perf_counter

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

from schemas import ArticleAnalysis

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

def create_client():
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing"
            "Add it to the root .env file.")

    return genai.Client(api_key=API_KEY)

def get_model():
    if not MODEL_NAME:
        raise ValueError(
            "GEMINI_MODEL is missing"
            "Add it to the root .env file."
        )
    return MODEL_NAME

def extract_article(text: str, client, model_name: str):
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    prompt = f"""
Analyze the following text.

Extract:
- a concise summary
- the most appropriate category
- the main topics
- an importance score

TEXT:
{text}
"""
    start = perf_counter()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ArticleAnalysis,
        ),
        )
    latency_ms = (perf_counter() - start) * 1000
    analysis = ArticleAnalysis.model_validate_json(
        response.text
        )

    return {
    "analysis": analysis,
    "model": model_name,
    "latency_ms": latency_ms,
}
