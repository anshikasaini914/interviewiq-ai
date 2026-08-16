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

def generate_report(conversation_history: list, correct_count: int, incorrect_count: int, total_questions: int) -> dict:
    """Poori conversation ko analyze karke structured feedback report banata hai."""

    # System aur RAG-context messages hata do, sirf actual conversation chahiye
    clean_conversation = [
        msg for msg in conversation_history
        if msg["role"] in ("user", "assistant")
    ]

    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in clean_conversation
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are analyzing a completed Data Science interview transcript. Based on the conversation, identify the candidate's strong and weak topics.

Respond ONLY with a JSON object in this exact format:
{"strong_topics": ["topic1", "topic2"], "weak_topics": ["topic1", "topic2"], "summary": "a 2-3 sentence overall assessment"}

Base topics on subject areas actually discussed (e.g., Python, SQL, Statistics, Machine Learning, Pandas). Keep the summary professional and constructive."""
                },
                {
                    "role": "user",
                    "content": f"Interview transcript:\n{conversation_text}"
                }
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)

    except Exception:
        result = {"strong_topics": [], "weak_topics": [], "summary": "Report generation unavailable."}

    return {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "strong_topics": result.get("strong_topics", []),
        "weak_topics": result.get("weak_topics", []),
        "summary": result.get("summary", "")
    }