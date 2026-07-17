from openai import OpenAI, APIStatusError
from app.core.config import GROQ_API_KEY

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

class GroqServiceError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)

def get_ai_response(messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return response.choices[0].message.content

    except APIStatusError as e:
        if e.status_code == 429:
            raise GroqServiceError(429, "Too many requests right now, please try again in a moment.")
        elif e.status_code == 401:
            raise GroqServiceError(500, "AI service configuration error. Please contact support.")
        else:
            raise GroqServiceError(502, "AI service is currently unavailable. Please try again later.")