from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import json
import io
import PyPDF2

from app.services.groq_service import (
    get_ai_response,
    GroqServiceError,
    evaluate_answer,
    generate_report,
    transcribe_audio,
    text_to_speech,
    extract_skills_from_resume,
)
from app.services.vector_store import get_relevant_questions
from app.core.config import redis_client
from app.models.interview_state import InterviewState


app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    target_role: str = ""


class SessionMetrics(BaseModel):
    overall_seconds: float = 0.0
    answer_times: list[float] = []
    evaluated_answers: int = 0

def build_system_prompt(role: str) -> dict:
    role_label = role.strip() if role and role.strip() else "the candidate's stated field"

    return {
        "role": "system",
        "content": f"""
You are an experienced Senior Technical Interviewer conducting a realistic interview for a {role_label} role.
- NEVER simulate, write, or assume the candidate's answer.
- NEVER write placeholder text like "Your answer, please" or continue the conversation on the candidate's behalf.
- After asking ONE question, STOP completely and wait for the real candidate response. Do not generate any further dialogue turns.

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
- Ask ONE technical question at a time, relevant to a {role_label} role and the candidate's stated background/skills. Wait for the candidate's answer.
- Use your own judgment of what topics, tools, and concepts a {role_label} candidate should be able to discuss, and cover different relevant areas gradually.
- Avoid asking many consecutive questions from the same topic.

PHASE 3 - Closing
- After around 10-15 questions, thank the candidate and tell them the interview has concluded.
- Do NOT provide feedback unless explicitly requested.

====================================================
EVALUATING EACH ANSWER — FOLLOW THIS DECISION ORDER
====================================================

STEP 1: Is the answer CORRECT and THOROUGH (explains the mechanism/reasoning, not just a label)?
→ Acknowledge briefly (1 sentence max) and move to a NEW question on a DIFFERENT topic.

STEP 2: Is the answer CORRECT but SHALLOW (right direction, but vague or missing the mechanism)?
→ Acknowledge briefly, then ask ONE targeted follow-up question probing the missing detail.
→ Do not reveal what the missing detail is.

STEP 3: Is the answer INCORRECT or PARTIALLY CORRECT?
→ Give EXACTLY ONE hint, then ask the SAME question again.
→ The hint must never state a fact/term/detail that would become part of the final answer.

====================================================
IF THE CANDIDATE ASKS FOR HELP DIRECTLY
====================================================

- If they ask for a HINT: give only ONE hint per question not more than 1. Do not add a refusal line.
- If they ask for the ANSWER: politely refuse — "We'll discuss that after the interview. For now, let's continue." — then repeat the current question.

====================================================
STRICT PROHIBITIONS
====================================================

Never: explain concepts, teach, give tutorials,thank after every answer, reveal answers/definitions, suggest learning resources, mention internal evaluation/scoring/notes/performance tracking.

====================================================
CONVERSATION STYLE
====================================================
- Be concise: at most 3 sentences, under 60 words per response.
- Do not ask multiple technical questions in one response.
- Wait for the candidate after every question.
- Your response must END immediately after asking the question. Do NOT add filler text like "Your answer, please".
- Never generate more than ONE question-hint-followup cycle in a single response.

====================================================
USING RAG
====================================================

You may receive a system message starting with [RAG_CONTEXT] containing questions from a question bank. Use them only as inspiration if relevant to the {role_label} role — do NOT copy them verbatim, and ignore them entirely if they don't fit this role.

====================================================
IMPORTANT
====================================================

Stay in interviewer mode throughout the conversation. Never break character.
"""
    }


RAG_MARKER = "[RAG_CONTEXT]"
GREETING_KEYWORDS = [
    "introduce yourself",
    "tell me about your background",
    "about yourself",
    "your background",
    "professional background",
    "your experience",
    "brief introduction",
    "your role",
]

# Domain-neutral — ab ye kisi bhi role ke liye safe fallback hai,
# sirf Data Science-specific nahi.
FALLBACK_QUESTIONS = [
    "Can you walk me through a challenging project you've worked on relevant to this role?",
    "What tools or technologies do you use most often in your work, and why?",
    "Can you describe a time you solved a difficult technical problem?",
    "What would you say is your strongest skill relevant to this role, and how have you applied it?",
    "How do you approach learning a new tool or technology quickly?",
    "Can you describe how you'd approach debugging an issue you've never seen before?",
]


# ------------------------------------------------------------
# Candidate-response guard
# ------------------------------------------------------------
NON_ANSWER_PHRASES = {
    "no", "nope", "nah", "hey", "hi", "hii", "hello",
    "ok", "okay", "yes", "yeah", "yep", "haan", "haan ji",
    "thanks", "thank you", "thankyou", "bye", "goodbye",
    "hmm", "hm", "hmmm", "umm", "um", "uh", "uhh",
    "nothing",
}

FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "hmm", "hm", "hmmm",
    "okay", "ok", "yeah", "yes", "no", "hey", "hi",
    "hello", "thanks", "thank", "you", "like", "so",
}

REPEAT_REQUESTS = {
    "repeat", "repeat please", "please repeat", "can you repeat",
    "could you repeat", "say that again", "ask again",
    "repeat the question", "please repeat the question",
}


def normalize_response(text: str) -> str:
    text = (text or "").strip().lower()
    return " ".join(text.split())


def is_non_answer(text: str) -> bool:
    normalized = normalize_response(text)

    if not normalized:
        return True

    if normalized in NON_ANSWER_PHRASES:
        return True

    words = normalized.replace(",", "").replace(".", "").split()

    if len(words) <= 8 and words and all(word in FILLER_WORDS for word in words):
        return True

    return False


def is_repeat_request(text: str) -> bool:
    normalized = normalize_response(text)
    return normalized in REPEAT_REQUESTS


def save_chat_history(session_id: str, history):
    redis_key = f"session:{session_id}"
    redis_client.set(redis_key, json.dumps(history), ex=3600)


def get_state(session_id: str) -> InterviewState:
    key = f"session:{session_id}:state"
    stored = redis_client.get(key)
    if stored:
        return InterviewState(**json.loads(stored))
    return InterviewState()


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
    state = get_state(request.session_id)

    if not stored_history:
        if not state.target_role:
            state.target_role = request.target_role
        history = [build_system_prompt(state.target_role)]
    else:
        history = json.loads(stored_history)
    candidate_text = normalize_response(request.message)

    if state.intro_done:
        last_question = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_question = msg.get("content", "")
                break

        if last_question:
            is_greeting_question = any(
                kw in last_question.lower()
                for kw in GREETING_KEYWORDS
            )

            if not is_greeting_question and is_repeat_request(request.message):
                history.append({"role": "user", "content": request.message})
                ai_reply = last_question
                history.append({"role": "assistant", "content": ai_reply})
                save_chat_history(request.session_id, history)
                save_state(request.session_id, state)

                return {
                    "user_message": request.message,
                    "reply": ai_reply,
                    "state": state.model_dump(),
                    "same_question": True,
                    "new_question": False,
                    "ignored_as_noise": False,
                    "answer_evaluated": False,
                }

            if not is_greeting_question and is_non_answer(request.message):
                history.append({"role": "user", "content": request.message})
                ai_reply = (
                    "Please answer the current interview question. "
                    "Take a moment and give your best response."
                )
                history.append({"role": "assistant", "content": ai_reply})
                save_chat_history(request.session_id, history)
                save_state(request.session_id, state)

                return {
                    "user_message": request.message,
                    "reply": ai_reply,
                    "state": state.model_dump(),
                    "same_question": True,
                    "new_question": False,
                    "ignored_as_noise": True,
                    "answer_evaluated": False,
                }

    answer_was_evaluated = False
    answer_was_correct = None

    if state.intro_done:
        last_question = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_question = msg.get("content", "")
                break

        is_greeting_question = last_question and any(
            kw in last_question.lower()
            for kw in GREETING_KEYWORDS
        )

        if last_question and not is_greeting_question:
            evaluation = evaluate_answer(last_question, request.message)
            answer_was_evaluated = True
            answer_was_correct = evaluation.get("is_correct")

            if answer_was_correct is True:
                state.correct_count += 1
                state.consecutive_wrong = 0
            elif answer_was_correct is False:
                state.incorrect_count += 1
                state.consecutive_wrong += 1

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

        save_chat_history(request.session_id, history)
        save_state(request.session_id, state)

        return {
            "user_message": request.message,
            "reply": ai_reply,
            "state": state.model_dump(),
            "same_question": False,
            "new_question": False,
            "interview_finished": True,
            "ignored_as_noise": False,
            "answer_evaluated": answer_was_evaluated,
        }

    if state.consecutive_wrong >= 2:
        state.consecutive_wrong = 0

        fallback_q = FALLBACK_QUESTIONS[
            state.question_count % len(FALLBACK_QUESTIONS)
        ]
        ai_reply = f"No worries, let's move on. {fallback_q}"

        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": ai_reply})

        save_chat_history(request.session_id, history)
        save_state(request.session_id, state)

        return {
            "user_message": request.message,
            "reply": ai_reply,
            "state": state.model_dump(),
            "same_question": False,
            "new_question": True,
            "ignored_as_noise": False,
            "answer_evaluated": answer_was_evaluated,
        }

    if not state.intro_done:
        state.greeting_exchange_count += 1

        is_still_greeting = any(
            kw in candidate_text
            for kw in ["hello", "hi ", "hii", "hey"]
        ) or len(candidate_text) < 15

        if state.greeting_exchange_count >= 2 or not is_still_greeting:
            state.intro_done = True
            state.phase = "technical"

    history = [
        msg for msg in history
        if not msg.get("content", "").startswith(RAG_MARKER)
    ]

    relevant_questions = get_relevant_questions(
        request.message,
        n_results=3
    )

    rag_context = {
        "role": "system",
        "content": f"{RAG_MARKER} " + " | ".join(relevant_questions)
    }

    history.append(rag_context)
    history.append({"role": "user", "content": request.message})

    try:
        ai_reply = get_ai_response(history)
    except GroqServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )

    history.append({"role": "assistant", "content": ai_reply})

    save_chat_history(request.session_id, history)
    save_state(request.session_id, state)

    same_question = answer_was_evaluated and answer_was_correct is False

    return {
        "user_message": request.message,
        "reply": ai_reply,
        "state": state.model_dump(),
        "same_question": same_question,
        "new_question": not same_question,
        "interview_finished": False,
        "ignored_as_noise": False,
        "answer_evaluated": answer_was_evaluated,
    }


@app.post("/metrics/{session_id}")
def save_metrics(session_id: str, metrics: SessionMetrics):
    key = f"session:{session_id}:metrics"
    redis_client.set(key, metrics.model_dump_json(), ex=3600)
    return {"success": True}


@app.get("/report/{session_id}")
def get_report(session_id: str):
    redis_key = f"session:{session_id}"

    stored_history = redis_client.get(redis_key)
    if not stored_history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    history = json.loads(stored_history)
    state = get_state(session_id)

    metrics_key = f"session:{session_id}:metrics"
    stored_metrics = redis_client.get(metrics_key)
    metrics = json.loads(stored_metrics) if stored_metrics else {
        "overall_seconds": 0.0,
        "answer_times": [],
        "evaluated_answers": state.correct_count + state.incorrect_count,
    }

    evaluated_answers = state.correct_count + state.incorrect_count
    performance = (
        state.correct_count / evaluated_answers * 100
        if evaluated_answers > 0 else 0.0
    )

    answer_times = [float(x) for x in metrics.get("answer_times", []) if float(x) >= 0]
    avg_hands_on = (sum(answer_times) / len(answer_times)) if answer_times else 0.0

    report = generate_report(
        conversation_history=history,
        correct_count=state.correct_count,
        incorrect_count=state.incorrect_count,
        total_questions=state.question_count
    )

    report["overall_time_seconds"] = float(metrics.get("overall_seconds", 0.0) or 0.0)
    report["average_hands_on_time_seconds"] = avg_hands_on
    report["evaluated_answers"] = evaluated_answers
    report["average_performance"] = performance
    report["performance_basis"] = "Only backend-evaluated question answers; filler/noise/repeat requests excluded."

    return report


@app.post("/extract-resume-skills")
async def extract_resume_skills(resume_file: UploadFile = File(...)):
    try:
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text() or ""

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        result = extract_skills_from_resume(resume_text)
        return {
            "suggested_role": result.get("suggested_role", ""),
            "skills": result.get("skills", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process resume: {str(e)}")


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):

    try:

        if not audio_file.content_type:
            raise HTTPException(status_code=400, detail="Audio file type is missing.")

        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Please upload a valid audio file.")

        audio_bytes = await audio_file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        audio_tuple = (
            audio_file.filename or "recording.wav",
            audio_bytes,
            audio_file.content_type
        )

        text = transcribe_audio(audio_tuple)

        return {
            "success": True,
            "text": text.strip()
        }

    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/text-to-speech")
def speech(text: str):

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_bytes = text_to_speech(text)

        if audio_bytes is None:
            raise HTTPException(status_code=502, detail="Could not generate speech audio.")

        return Response(content=audio_bytes, media_type="audio/mpeg")

    except GroqServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")