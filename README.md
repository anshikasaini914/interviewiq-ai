InterviewIQ AI

An AI-powered Data Science interview simulator that conducts realistic, adaptive technical interviews — asking questions, evaluating answers, giving hints without revealing solutions, and generating a structured feedback report at the end.

Built as a hands-on learning project covering FastAPI, LLM integration, RAG, conversation memory, and structured evaluation with LLMs.

Features
Conversational interview flow — greets the candidate, asks technical questions one at a time, and wraps up naturally after a set number of questions.
Retrieval-Augmented Question Bank (RAG) — questions are dynamically retrieved from a curated 25-question Data Science question bank (ChromaDB + sentence-transformers embeddings) based on the direction the candidate takes the conversation.
Persistent conversation memory — powered by Redis, so context survives across turns and backend restarts within a session's TTL.
Guided hints, not answers — if a candidate doesn't know an answer, the interviewer gives one directional hint (never revealing the answer) before moving on.
Automated scoring — every technical answer is evaluated by a dedicated LLM call that returns a structured correct/incorrect judgment, tracked via server-side state (not just LLM text).
End-of-interview feedback report — a structured report with total questions, correct/incorrect counts, strong topics, weak topics, and a written summary.
Streamlit chat interface — a clean, professional chat UI with session reset and live report generation.
Tech Stack
Layer	Technology
Backend	FastAPI, Pydantic
LLM Inference	Groq API (llama-3.3-70b-versatile), OpenAI-compatible SDK
Conversation Memory	Redis (Docker)
Vector Search / RAG	ChromaDB, sentence-transformers (all-MiniLM-L6-v2)
Frontend	Streamlit
Language	Python

Note on LangChain: LangChain was evaluated during development (see Week 5 in the project history) and deliberately not adopted — the raw SDK approach already handled this project's complexity (memory, retrieval, error handling) without the added abstraction overhead.

Architecture
Streamlit UI  →  FastAPI Backend  →  Groq API (LLM)
                       │
                       ├── Redis (conversation history + interview state)
                       └── ChromaDB (question bank retrieval)

Each /chat request:

Fetches conversation history and interview state from Redis.
Retrieves relevant questions from ChromaDB based on the candidate's latest message.
Evaluates the candidate's previous answer (if applicable) via a dedicated structured LLM call.
Sends the full context to Groq for the interviewer's next reply.
Updates and persists history and state back to Redis.
Setup Instructions
Prerequisites
Python 3.11+
Docker Desktop (for Redis)
A free Groq API key — get one at console.groq.com (no credit card required)
1. Clone the repository
bash
git clone https://github.com/anshikasaini914/interviewiq-ai.git
cd interviewiq-ai
2. Create and activate a virtual environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up environment variables

Copy .env.example to a new file named .env, and add your own Groq API key:

GROQ_API_KEY=your_actual_key_here
5. Start Redis (via Docker)
bash
docker run -d --name redis-interviewiq -p 6379:6379 redis

If the container already exists from a previous run, use docker start redis-interviewiq instead.

6. Load the question bank into ChromaDB
bash
python load_questions.py

This only needs to be run once — it populates the local vector database at ./chroma_data.

7. Run the backend
bash
uvicorn app.main:app --reload

API docs available at http://127.0.0.1:8000/docs.

8. Run the frontend

In a separate terminal (with the virtual environment activated):

bash
streamlit run streamlit_app.py

The app will open at http://localhost:8501.

Project Structure
interviewiq-ai/
├── app/
│   ├── main.py                  # FastAPI app, routes, interview flow logic
│   ├── core/
│   │   └── config.py             # Environment variables, Redis connection
│   ├── models/
│   │   └── interview_state.py    # Pydantic model for interview state
│   ├── routers/
│   └── services/
│       ├── groq_service.py       # LLM calls: interviewer replies, evaluation, report generation
│       └── vector_store.py       # ChromaDB setup and retrieval
├── data/
│   └── questions.json            # Curated Data Science question bank
├── streamlit_app.py              # Frontend chat interface
├── load_questions.py             # One-time script to populate ChromaDB
├── requirements.txt
├── .env.example
└── README.md
Known Limitations
Hint leakage: The interviewer's hint-generation relies on prompt-based instructions to avoid revealing answers. In testing, the LLM occasionally reveals partial definitions or key terms despite explicit "never reveal" rules in the system prompt — a known constraint of prompt-only guardrails with smaller open-source models. A planned improvement is a code-level post-processing check that validates hint responses before sending them to the candidate.
Question counting is approximate: each candidate message (including hint retries) currently counts toward the question limit, rather than only counting distinct new questions. This is a reasonable approximation for the current scope but not perfectly precise.
Session persistence in the UI: conversation state is tied to a session_id held in Streamlit's session state. A browser refresh starts a new session on the frontend, even though the previous session's data remains in Redis until its TTL (1 hour) expires. Proper cross-refresh persistence would require authentication (planned for a later phase).