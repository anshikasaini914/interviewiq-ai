from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.groq_service import get_ai_response, GroqServiceError
from app.core.config import redis_client
import json

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a professional Data Science interviewer conducting a technical interview. Ask clear, relevant questions and respond professionally to candidate answers."
}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    redis_key = f"session:{request.session_id}"

    # Step 1: Fetch existing history from Redis
    stored_history = redis_client.get(redis_key)
    if stored_history:
        history = json.loads(stored_history)
    else:
        history = [SYSTEM_PROMPT]

    # Step 2: Add new message of User
    history.append({"role": "user", "content": request.message})

    # Step 3: Send complete history to Groq
    try:
        ai_reply = get_ai_response(history)
    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    # Step 4: Add AI reply in History
    history.append({"role": "assistant", "content": ai_reply})

    # Step 5: Save back updated history in redis(TTL: 1 hour)
    redis_client.set(redis_key, json.dumps(history), ex=3600)

    return {
        "user_message": request.message,
        "reply": ai_reply
    }