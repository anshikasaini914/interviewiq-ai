from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.groq_service import get_ai_response, GroqServiceError

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        ai_reply = get_ai_response(request.message)
    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    return {
        "user_message": request.message,
        "reply": ai_reply
    }