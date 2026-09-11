import os

from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing")

    return database_url

def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    return api_key

def get_gemini_model():
    model = os.getenv("GEMINI_MODEL")

    if not model:
        raise ValueError("GEMINI_MODEL is missing")

    return model

