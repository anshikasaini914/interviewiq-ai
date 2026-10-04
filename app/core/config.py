import os
from dotenv import load_dotenv
import redis


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

REDIS_URL = os.getenv("REDIS_URL")
print(f"DEBUG — REDIS_URL value: {repr(REDIS_URL)}")  # YE LINE TEMPORARY HAI

if REDIS_URL:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
else:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)