import os
import redis
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Redis connection
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)