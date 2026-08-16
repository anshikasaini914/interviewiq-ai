import json
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


def evaluate_answer(question: str, answer: str) -> dict:
    """Candidate ke answer ko evaluate karta hai, structured JSON return karta hai."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an evaluator for a Data Science interview. Given a question and a candidate's answer, determine if the answer is substantively correct.

                    Respond ONLY with a JSON object in this exact format:
                   {"is_correct": true or false, "confidence": "high" or "medium" or "low"}

                    Consider an answer correct if it demonstrates real understanding, even if not perfectly worded. Consider "I don't know" or clearly wrong answers as incorrect."""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nCandidate's answer: {answer}"
                }
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        return json.loads(result)

    except Exception as e:
        # If evaluation failed, return sade default msg
        return {"is_correct": None, "confidence": "low"}