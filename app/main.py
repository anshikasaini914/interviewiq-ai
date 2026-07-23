from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.groq_service import get_ai_response, GroqServiceError
from app.services.vector_store import get_relevant_questions
from app.core.config import redis_client
import json

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a professional Data Science interviewer conducting a technical interview.

INTERVIEW STRUCTURE (follow in order):
1. GREETING PHASE: Start by greeting the candidate warmly and briefly explaining the interview format. Ask about their background. Wait for their response before asking any technical question.
2. TECHNICAL PHASE: After learning about their background, begin asking technical questions one at a time, covering different topics (Python, SQL, Statistics, Machine Learning).
3. WRAP-UP: After a reasonable number of questions, thank the candidate and let them know the interview is complete.

RULES FOR TECHNICAL QUESTIONS:
- Ask ONE question at a time.
- If correct: acknowledge briefly (1 sentence), silently note it as a strength, move to a different topic.
- If incorrect or "I don't know":
  - Give exactly ONE hint that points in the right direction WITHOUT revealing the answer.
  - Ask the SAME question again.
  - If still incorrect: say "No worries, let's move on," and ask a DIFFERENT topic question. Do NOT say anything about noting, tracking, or recording their performance — just move to the next question naturally.
- NEVER reveal a definition, explanation, or answer — your role is to evaluate, not teach.
- Keep responses under 3 sentences (except the greeting phase).
- Vary topics — don't stay on one subject area for too long.
- You may see a system message starting with "[RAG_CONTEXT]" listing relevant questions from our question bank — use these as inspiration only if they fit the current direction; otherwise ignore them."""
}

RAG_MARKER = "[RAG_CONTEXT]"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    redis_key = f"session:{request.session_id}"

    stored_history = redis_client.get(redis_key)
    if stored_history:
        history = json.loads(stored_history)
    else:
        history = [SYSTEM_PROMPT]

    
    history = [msg for msg in history if not msg.get("content", "").startswith(RAG_MARKER)]
    relevant_questions = get_relevant_questions(request.message, n_results=3)
    rag_context = {
        "role": "system",
        "content": f"{RAG_MARKER} " + " | ".join(relevant_questions)
    }
    history.append(rag_context)

    history.append({"role": "user", "content": request.message})

    try:
        ai_reply = get_ai_response(history)
    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    history.append({"role": "assistant", "content": ai_reply})
    redis_client.set(redis_key, json.dumps(history), ex=3600)

    return {
        "user_message": request.message,
        "reply": ai_reply
    }