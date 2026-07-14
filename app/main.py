from fastapi import FastAPI
from pydantic import BaseModel
from app.services.groq_service import get_ai_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    ai_reply = get_ai_response(request.message)
    return {
        "user_message": request.message,
        "reply": ai_reply
    }