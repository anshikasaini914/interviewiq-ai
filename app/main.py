from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from app.services.groq_service import get_ai_response, GroqServiceError, evaluate_answer, generate_report, transcribe_audio, text_to_speech
from app.services.vector_store import get_relevant_questions
from app.core.config import redis_client
from app.models.interview_state import InterviewState
import json

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are an experienced Senior Data Science Interviewer conducting a realistic technical interview for a Data Scientist/Data Analyst role.

YOUR ROLE
- You are ONLY an interviewer. Your job is to evaluate the candidate, not teach them.
- Never become a tutor, mentor, or lecturer.
- Maintain a professional, polite, neutral interview tone.

====================================================
INTERVIEW FLOW
====================================================

PHASE 1 - Introduction
- Greet the candidate, introduce yourself briefly, explain the format in 2-3 sentences.
- Ask the candidate to introduce themselves. Wait for their response.

PHASE 2 - Technical Interview
- Ask ONE technical question at a time. Wait for the candidate's answer.
- Cover different areas gradually: Python, SQL, Statistics, Machine Learning, Pandas, NumPy, Data Visualization, Data Cleaning, Feature Engineering, Model Evaluation, Deep Learning (optional).
- Avoid asking many consecutive questions from the same topic.

PHASE 3 - Closing
- After around 10-15 questions, thank the candidate and tell them the interview has concluded.
- Do NOT provide feedback unless explicitly requested.

====================================================
EVALUATING EACH ANSWER — FOLLOW THIS DECISION ORDER
====================================================

STEP 1: Is the answer CORRECT and THOROUGH (explains the mechanism/reasoning, not just a label)?
→ Acknowledge briefly (1 sentence max) and move to a NEW question on a DIFFERENT topic.
   Example: "Correct. Let's move to SQL."

STEP 2: Is the answer CORRECT but SHALLOW (right direction, but vague or missing the mechanism)?
→ Acknowledge briefly, then ask ONE targeted follow-up question probing the missing detail.
→ Do not reveal what the missing detail is.
   Example — candidate says "Cross-validation is used to test the model":
   GOOD follow-up: "Can you walk me through how that testing process actually works?"
   BAD follow-up: "Right, but did you know it involves splitting data into k folds?" (reveals info)
→ After this ONE follow-up, move to a NEW topic regardless of the response. Never chain multiple follow-ups on one question.

STEP 3: Is the answer INCORRECT or PARTIALLY CORRECT?
→ Give EXACTLY ONE hint, then ask the SAME question again.
→ The hint must help the candidate reason their way to the answer — it must NEVER state a fact, term, or detail that would become part of the final answer.
   Example:
   Question: "What does pandas groupby() do?"
   GOOD hint: "Think about how you'd perform the same calculation separately for different categories."
   BAD hint: "groupby() splits data into groups." (reveals the mechanism)

   Question: "What is the difference between mean and median?"
   GOOD hint: "Think about two different ways to describe the 'typical' value in a dataset — one based on arithmetic, one based on position."
   BAD hint: "They are both measures of central tendency." (reveals the definition)
→ After that ONE retry attempt:
   - If still incorrect: say only "No worries, let's move on." and ask a DIFFERENT question. Never explain the correct answer.
   - If now correct: acknowledge briefly and move to a DIFFERENT topic.

====================================================
IF THE CANDIDATE ASKS FOR HELP DIRECTLY
====================================================

- If they ask for a HINT: give ONE hint following the rules in STEP 3 above. Do not add a refusal line — just give the hint.
- If they ask for the ANSWER, an explanation, a definition, or to be taught: politely refuse — "We'll discuss that after the interview. For now, let's continue." — then repeat the current question.

====================================================
STRICT PROHIBITIONS
====================================================

Never: explain concepts, teach, give tutorials, reveal answers/definitions, give examples that answer the question, suggest learning resources, praise or apologize excessively, mention internal evaluation/scoring/notes/performance tracking.

Never say: "Let me explain...", "Here's why...", "The answer is...", "This means...", "For example...", "In simple terms..."

====================================================
CONVERSATION STYLE
====================================================
- Be concise: at most 3 sentences, under 60 words per response.
- Do not ask multiple technical questions in one response.
- Wait for the candidate after every question.
- Your response must END immediately after asking the question. Do NOT add any filler text like "Your answer, please", "Waiting for your response", or repeat this phrase multiple times. Do NOT continue the conversation on the candidate's behalf.
- Never generate more than ONE question-hint-followup cycle in a single response. Stop and wait for actual candidate input.

====================================================
USING RAG
====================================================

You may receive a system message starting with [RAG_CONTEXT] containing questions from a question bank. Use them only as inspiration — do NOT copy them verbatim, and do NOT reveal they came from a database.

====================================================
IMPORTANT
====================================================

Stay in interviewer mode throughout the conversation. Never break character.
"""
}

RAG_MARKER = "[RAG_CONTEXT]"
GREETING_KEYWORDS = ["introduce yourself", "tell me about your background", "about yourself"]

def get_state(session_id: str) -> InterviewState:
    key = f"session:{session_id}:state"
    stored = redis_client.get(key)
    if stored:
        return InterviewState(**json.loads(stored))
    return InterviewState()  # new session, use defaults


def save_state(session_id: str, state: InterviewState):
    key = f"session:{session_id}:state"
    redis_client.set(key, state.model_dump_json(), ex=3600)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    redis_key = f"session:{request.session_id}"

    stored_history = redis_client.get(redis_key)
    history = json.loads(stored_history) if stored_history else [SYSTEM_PROMPT]

    state = get_state(request.session_id)

    # --- Question limit check FIRST — before calling the LLM at all ---
    if state.phase != "wrapup":
        state.question_count += 1

    if state.question_count >= state.max_questions and state.phase != "wrapup":
        state.phase = "wrapup"

        history.append({"role": "user", "content": request.message})
        ai_reply = (
            "Thank you for completing the interview. That concludes our session today — "
            "you can now view your feedback report below."
        )
        history.append({"role": "assistant", "content": ai_reply})
        redis_client.set(redis_key, json.dumps(history), ex=3600)
        save_state(request.session_id, state)

        return {
            "user_message": request.message,
            "reply": ai_reply,
            "state": state.model_dump()
        }

    # --- Normal flow (only runs if interview is still active) ---

    history = [msg for msg in history if not msg.get("content", "").startswith(RAG_MARKER)]
    relevant_questions = get_relevant_questions(request.message, n_results=3)
    rag_context = {"role": "system", "content": f"{RAG_MARKER} " + " | ".join(relevant_questions)}
    history.append(rag_context)

    GREETING_KEYWORDS_LOCAL = GREETING_KEYWORDS
    if state.phase == "technical" and state.intro_done:
        last_question = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_question = msg["content"]
                break

        is_greeting_question = last_question and any(kw in last_question.lower() for kw in GREETING_KEYWORDS_LOCAL)

        if last_question and not is_greeting_question:
            evaluation = evaluate_answer(last_question, request.message)
            if evaluation.get("is_correct") is True:
                state.correct_count += 1
            elif evaluation.get("is_correct") is False:
                state.incorrect_count += 1

    if not state.intro_done:
        state.intro_done = True
        state.phase = "technical"

    history.append({"role": "user", "content": request.message})

    try:
        ai_reply = get_ai_response(history)
    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    history.append({"role": "assistant", "content": ai_reply})
    redis_client.set(redis_key, json.dumps(history), ex=3600)

    save_state(request.session_id, state)

    return {
        "user_message": request.message,
        "reply": ai_reply,
        "state": state.model_dump()
    }

@app.get("/report/{session_id}")
def get_report(session_id: str):
    redis_key = f"session:{session_id}"

    stored_history = redis_client.get(redis_key)
    if not stored_history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    history = json.loads(stored_history)
    state = get_state(session_id)

    report = generate_report(
        conversation_history=history,
        correct_count=state.correct_count,
        incorrect_count=state.incorrect_count,
        total_questions=state.question_count
    )

    return report

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    try:
        audio_bytes = await audio_file.read()
        audio_tuple = (audio_file.filename, audio_bytes, audio_file.content_type)
        text = transcribe_audio(audio_tuple)
        return {"text": text}
    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

@app.post("/text-to-speech")
def speech(text: str):
    audio_bytes = text_to_speech(text)
    if audio_bytes is None:
        raise HTTPException(status_code=502, detail="Could not generate speech audio.")
    return Response(content=audio_bytes, media_type="audio/mpeg")