# InterviewIQ AI

> **An AI-powered, role-agnostic Interview Simulator** that conducts realistic, adaptive technical interviews for any job role, evaluates candidate answers in real time, provides guided hints, and generates a structured feedback report.

InterviewIQ AI is a hands-on AI project built to explore **FastAPI, LLM integration, RAG, Redis-based conversation memory, vector databases, structured LLM evaluation, and voice I/O**.

---

## 🚀 Features

### 🎯 Role-Agnostic Interviews

* Not limited to a single domain — can conduct interviews for **any role** (Data Science, Backend Development, Product Management, etc.).
* Upload a resume and the system **suggests a target role and relevant skills** automatically.
* The candidate can **confirm or override** the suggested role before starting.
* The interview questions and evaluation criteria adapt dynamically based on the confirmed role.

### 💬 Conversational Interview Flow

* Conducts a realistic technical interview through natural conversation.
* Asks one question at a time and waits for a real candidate response.
* Gives exactly one guided hint on an incorrect/incomplete answer, then moves on.
* Code-level safeguards (not just prompting) ensure the interview progresses correctly — question limits, stuck-question detection, and empty-response handling are all enforced in the backend rather than left to the LLM.

### 🎙️ Voice or Text Interview

* Candidates can answer by **speaking** (transcribed via Groq Whisper) or by **typing**.
* AI interviewer questions are read aloud via text-to-speech (gTTS) in voice mode.
* Silence and transcription hallucinations are filtered out before being treated as an answer.

### 🔎 Retrieval-Augmented Question Bank (RAG)

* Uses **ChromaDB** + **Sentence Transformers** (`all-MiniLM-L6-v2`) to retrieve topic-relevant questions as *inspiration* for the interviewer.
* The LLM is instructed to use retrieved questions only as a guide — never copy them verbatim, and to ignore them if they don't fit the candidate's role.

### 🧠 Persistent Conversation Memory

* Uses **Redis** (hosted on Upstash) to maintain:
  * Conversation history
  * Interview state (phase, question count, correctness streaks)
  * Candidate progress
* Context persists across requests within the session TTL.

### 📊 Automated Answer Evaluation

* Every technical answer is evaluated using a dedicated, structured LLM call (JSON output).
* Results are tracked via **server-side interview state**, not by trusting LLM-generated text — this is a deliberate design principle used throughout the backend.

### ⏱️ Live Interview Timer

* Overall interview countdown and per-question answer timer, both visible in a minimal status bar.
* Timer accounts for interviewer speech duration in voice mode before starting the candidate's answer clock.

### 📈 End-of-Interview Feedback Report

* Total questions, correct/incorrect counts, accuracy
* Strong and weak topics
* Overall performance summary
* Total interview time and average answer time
* Visual charts (correct vs. incorrect, strong vs. weak topics)

### 🖥️ Minimal, Chat-Style Interface

* Clean, centered Streamlit interface inspired by ChatGPT/Claude.
* Simple message bubbles, slim status bar, minimal sidebar.

---

## 🛠️ Tech Stack

| Layer                | Technology                                  |
| --------------------- | -------------------------------------------- |
| Backend               | FastAPI, Pydantic                            |
| LLM Inference         | Groq API, `openai/gpt-oss-120b`              |
| LLM SDK               | OpenAI-compatible SDK                        |
| Speech-to-Text        | Groq Whisper (`whisper-large-v3`)            |
| Text-to-Speech        | gTTS                                         |
| Conversation Memory   | Redis (hosted on Upstash)                    |
| Vector Database       | ChromaDB                                     |
| Embeddings            | Sentence Transformers (`all-MiniLM-L6-v2`)   |
| Resume Parsing        | PyPDF2                                       |
| Frontend              | Streamlit                                    |
| Language              | Python                                       |

### Why not LangChain?

LangChain was evaluated during development but deliberately not adopted. The project uses the **raw SDK approach** because the application's requirements around memory, retrieval, evaluation, and error handling could be implemented directly without adding another abstraction layer.

### Why not Docker?

The project initially used Docker for local Redis and planned containerized deployment. Docker Desktop's background resource usage (a persistent WSL2 VM) proved heavy for local development, so Redis was moved to a managed cloud instance (Upstash) instead — removing the need for Docker entirely during development.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │   Groq API  │  │ Redis Cloud │  │  ChromaDB   │
       │  LLM + STT  │  │  (Upstash)  │  │  Question   │
       │   + TTS     │  │ Memory+State│  │    Bank     │
       └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 🔄 Interview Flow

For every `/chat` request, the backend follows this flow:

```text
Candidate Message
       │
       ▼
Fetch Conversation History + State (Redis)
       │
       ▼
Filter Noise / Filler / Repeat Requests
       │
       ▼
Evaluate Previous Answer (if applicable)
       │
       ▼
Check Question Limit → Wrap Up If Reached
       │
       ▼
Check Consecutive-Wrong Streak → Force-Skip If Stuck
       │
       ▼
Retrieve Relevant Questions (RAG)
       │
       ▼
Generate Next Interviewer Response (role-aware prompt)
       │
       ▼
Update + Persist Interview State (Redis)
       │
       ▼
Return Response to Streamlit
```

---

## 📁 Project Structure

```text
interviewiq-ai/
│
├── app/
│   ├── main.py
│   │   └── FastAPI application, routes & interview flow
│   │
│   ├── core/
│   │   └── config.py
│   │       └── Environment variables & Redis configuration
│   │
│   ├── models/
│   │   └── interview_state.py
│   │       └── Pydantic interview state model
│   │
│   └── services/
│       ├── groq_service.py
│       │   └── LLM calls, evaluation, transcription, TTS & report generation
│       │
│       └── vector_store.py
│           └── ChromaDB setup & question retrieval
│
├── data/
│   └── questions.json
│       └── Question bank used for RAG inspiration
│
├── streamlit_app.py
│   └── Streamlit chat interface
│
├── load_questions.py
│   └── Loads questions into ChromaDB
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Setup & Installation

## Prerequisites

* Python **3.11+**
* Git
* A free **Groq API key**
* A free **Upstash Redis database**

> No credit card is required for either Groq or Upstash free tiers.

---

## 1. Clone the Repository

```bash
git clone https://github.com/anshikasaini914/interviewiq-ai.git
cd interviewiq-ai
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Set Up Redis (Upstash)

1. Create a free account at [upstash.com](https://upstash.com)
2. Create a new Redis database
3. Open the **Connect** tab and copy the `rediss://` connection string (not the REST API URL)

---

## 5. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```env
GROQ_API_KEY=your_actual_groq_api_key
REDIS_URL=rediss://default:your_password@your-db.upstash.io:6379
```

---

## 6. Load the Question Bank

Populate the local ChromaDB vector store:

```bash
python load_questions.py
```

This only needs to be run once unless the question bank is changed. The vector database is created locally at `./chroma_data`.

---

## 7. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Backend: `http://127.0.0.1:8000`
API docs: `http://127.0.0.1:8000/docs`

---

## 8. Start the Streamlit Frontend

Open a **new terminal**, activate the virtual environment again, then:

```bash
streamlit run streamlit_app.py
```

App: `http://localhost:8501`

---

# 🧪 Example Interview Flow

```text
Candidate uploads a resume → system suggests "Backend Developer"
as the target role with skills: Python, FastAPI, SQL.
Candidate confirms the role and starts.

Interviewer:
Hello, I'm Alex, a senior technical interviewer. We'll have a
brief interview covering topics relevant to a Backend Developer
role. Could you start by introducing yourself?

Candidate:
[introduces themselves]

Interviewer:
Can you explain the difference between SQL and NoSQL databases,
and when you'd choose one over the other?

Candidate:
I don't know.

Interviewer:
Think about how each stores and structures data — one uses fixed
tables, the other is more flexible. How would you compare them?

Candidate:
[answers]

Interviewer:
Correct. Let's move to a different topic...
```

---

# 🧠 Core AI Components

## 1. Dynamic, Role-Aware Prompting

The interviewer's system prompt is generated at runtime based on the candidate's confirmed target role, rather than being hardcoded to a single domain.

## 2. RAG-Based Question Retrieval

Question embeddings (`all-MiniLM-L6-v2`) are stored in ChromaDB. The system retrieves semantically relevant questions as inspiration for the LLM — the LLM is explicitly instructed to ignore them if they don't fit the candidate's role.

## 3. Conversation Memory

Redis stores session-specific information:

```text
Conversation History
Interview State (phase, question count, correctness streak)
Correct / Incorrect Answers
Target Role
Timing Metrics
```

## 4. LLM-Based Evaluation

Candidate answers are evaluated using a separate, structured LLM call that returns correctness as JSON — this result drives the backend state rather than being inferred from conversational text.

## 5. Code-Enforced Interview Control

Several critical behaviors are enforced in code rather than left to prompting alone:

* Question-count limits and wrap-up
* Force-skipping a question after repeated incorrect answers
* Filtering filler/noise/repeat-request speech before evaluation
* Retry-and-fallback handling for empty LLM responses

## 6. Feedback Generation

After the interview finishes, a dedicated LLM call analyzes the full transcript to produce strong/weak topics and a summary, combined with backend-tracked counts and timing.

---

# 📊 End-of-Interview Report

Example structure:

```text
Interview Performance Report

Total Questions: 10
Correct: 7
Incorrect: 3
Accuracy: 70%

Strong Topics:
- API Design
- SQL

Weak Topics:
- System Design

Total Interview Time: 14:32
Average Answer Time: 00:48

Overall Summary:
The candidate demonstrates solid understanding of API design and
SQL fundamentals, but would benefit from more practice with
system design concepts.
```

---

# ⚠️ Known Limitations

### 1. Question Bank Coverage

The RAG question bank was originally curated for Data Science and has since been cleared out as the project generalized to any role. The LLM currently generates questions using its own judgment, guided by the candidate's stated role and skills, rather than a populated multi-role question bank.

**Planned improvement:** build out a multi-role question bank with role metadata for more consistent, curated questions across domains.

### 2. Hint Leakage

The hint-generation system relies primarily on prompt-based instructions to avoid revealing answers. The model occasionally reveals partial definitions despite explicit instructions — a known limitation of prompt-only guardrails with smaller open-source models.

**Planned improvement:** a code-level post-processing layer that validates generated hints before returning them to the candidate.

### 3. Session Persistence

The frontend stores `session_id` in Streamlit session state. A browser refresh starts a new Streamlit session (and a new `session_id`); the previous Redis session remains until its 1-hour TTL expires, but isn't automatically reconnected to.

**Planned improvement:** authentication and persistent user/session management.

---

# 🔮 Future Improvements

* [ ] Authentication and user accounts
* [ ] Persistent interview history across sessions
* [ ] Multi-role, curated question bank with role metadata
* [ ] Difficulty levels: Beginner / Intermediate / Advanced
* [ ] Better hint validation (code-level, not just prompt-level)
* [ ] Interview analytics dashboard
* [ ] Cloud deployment (Render / Railway)
* [ ] Multiple interview modes (behavioral, system design, etc.)

---

# 🎯 Learning Outcomes

This project was built as a hands-on way to learn and apply:

* FastAPI and REST API design
* Pydantic data models
* LLM integration via Groq and OpenAI-compatible SDKs
* Retrieval-Augmented Generation (RAG) with ChromaDB
* Sentence embeddings
* Redis-based conversation memory and state management
* Structured LLM evaluation (JSON-mode outputs)
* Prompt engineering for controllable, dynamic LLM behavior
* Speech-to-text and text-to-speech integration
* Streamlit for interactive frontends
* Debugging real-world LLM instruction-following failures and
  designing code-level safeguards around them

---