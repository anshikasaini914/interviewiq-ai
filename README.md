# InterviewIQ AI

> **An AI-powered Data Science Interview Simulator** that conducts realistic, adaptive technical interviews, evaluates candidate answers, provides guided hints, and generates a structured feedback report.

InterviewIQ AI is a hands-on AI project built to explore **FastAPI, LLM integration, RAG, Redis-based conversation memory, vector databases, and structured LLM evaluation**.

---

## 🚀 Features

### 💬 Conversational Interview Flow

* Conducts a realistic technical interview through conversation.
* Asks questions one at a time.
* Adapts the interview based on the candidate's responses.
* Naturally concludes after a predefined number of questions.

### 🔎 Retrieval-Augmented Question Bank (RAG)

* Uses a curated **25-question Data Science question bank**.
* Dynamically retrieves relevant questions using:

  * **ChromaDB**
  * **Sentence Transformers**
  * `all-MiniLM-L6-v2` embeddings
* Question retrieval is influenced by the direction of the conversation.

### 🧠 Persistent Conversation Memory

* Uses **Redis** to maintain:

  * Conversation history
  * Interview state
  * Candidate progress
* Context persists across requests and backend restarts within the session TTL.

### 💡 Guided Hints — Not Answers

* Candidates can ask for help when they don't know an answer.
* The interviewer provides a **directional hint** instead of revealing the solution.
* Helps simulate a realistic interview environment.

### 📊 Automated Answer Evaluation

Every technical answer is evaluated using a dedicated LLM call.

The evaluation returns a structured result containing:

* Correct / Incorrect judgment
* Topic information
* Evaluation reasoning

The result is tracked using **server-side interview state**, rather than relying only on generated LLM text.

### 📈 End-of-Interview Feedback Report

At the end of the interview, the system generates a structured report containing:

* Total questions
* Correct answers
* Incorrect answers
* Strong topics
* Weak topics
* Overall performance summary

### 🖥️ Streamlit Interface

A clean chat-based frontend built with Streamlit provides:

* Real-time interview interaction
* Session reset
* Live conversation history
* Interview report generation

---

## 🛠️ Tech Stack

| Layer               | Technology                          |
| ------------------- | ----------------------------------- |
| Backend             | FastAPI, Pydantic                   |
| LLM Inference       | Groq API, `llama-3.3-70b-versatile` |
| LLM SDK             | OpenAI-compatible SDK               |
| Conversation Memory | Redis                               |
| Vector Database     | ChromaDB                            |
| Embeddings          | Sentence Transformers               |
| Embedding Model     | `all-MiniLM-L6-v2`                  |
| Frontend            | Streamlit                           |
| Containerization    | Docker                              |
| Language            | Python                              |

### Why not LangChain?

LangChain was evaluated during development but deliberately not adopted.

The project uses the **raw SDK approach** because the application's requirements around memory, retrieval, evaluation, and error handling could be implemented directly without adding another abstraction layer.

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
       │   Groq API  │  │    Redis    │  │  ChromaDB   │
       │     LLM     │  │   Memory    │  │  Question   │
       │             │  │   + State   │  │    Bank     │
       └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 🔄 Interview Flow

For every `/chat` request, the backend follows this flow:

```text
Candidate Message
       │
       ▼
Fetch Conversation History
       │
       ▼
Fetch Interview State
       │
       ▼
Retrieve Relevant Questions
       │
       ▼
Evaluate Previous Answer
       │
       ▼
Send Context + Instructions to LLM
       │
       ▼
Generate Next Interviewer Response
       │
       ▼
Update Interview State
       │
       ▼
Persist Data in Redis
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
│   ├── routers/
│   │
│   └── services/
│       ├── groq_service.py
│       │   └── LLM calls, evaluation & report generation
│       │
│       └── vector_store.py
│           └── ChromaDB setup & question retrieval
│
├── data/
│   └── questions.json
│       └── Curated Data Science question bank
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

Make sure the following are installed:

* Python **3.11+**
* Docker Desktop
* Git
* A free **Groq API key**

> No credit card is required for the Groq API key.

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

## 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Replace `your_actual_groq_api_key` with your Groq API key.

---

## 5. Start Redis Using Docker

Run:

```bash
docker run -d --name redis-interviewiq -p 6379:6379 redis
```

If the container already exists:

```bash
docker start redis-interviewiq
```

You can verify that Redis is running with:

```bash
docker ps
```

---

## 6. Load the Question Bank

Populate the local ChromaDB vector store:

```bash
python load_questions.py
```

This only needs to be executed once unless the question bank is changed.

The vector database will be created locally at:

```text
./chroma_data
```

---

## 7. Start the FastAPI Backend

Run:

```bash
uvicorn app.main:app --reload
```

The backend will start at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Start the Streamlit Frontend

Open a **new terminal** and activate the virtual environment again.

Then run:

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧪 Example Interview Flow

```text
Interviewer:
Welcome! Let's begin your Data Science interview.

Interviewer:
What is the difference between supervised and unsupervised learning?

Candidate:
Supervised learning uses labelled data while unsupervised
learning works with unlabelled data.

Interviewer:
Good. Let's move to the next question.

Interviewer:
What is overfitting and how can you prevent it?

Candidate:
I don't know.

Interviewer:
Think about a model that performs very well on training
data but poorly on unseen data. What problem could cause this?

Interviewer:
Let's continue with another question.
```

The system evaluates answers in the backend and maintains the candidate's progress throughout the interview.

---

# 🧠 Core AI Components

## 1. RAG-Based Question Retrieval

The question bank is converted into embeddings using:

```text
all-MiniLM-L6-v2
```

These embeddings are stored in ChromaDB.

When the candidate responds, the system retrieves semantically relevant questions instead of selecting questions completely at random.

---

## 2. Conversation Memory

Redis stores session-specific information such as:

```text
Conversation History
Interview State
Question Count
Correct Answers
Incorrect Answers
Topics Covered
```

This allows the backend to maintain context across multiple `/chat` requests.

---

## 3. LLM-Based Evaluation

Candidate answers are evaluated using a separate LLM call.

Conceptually:

```text
Candidate Answer
       │
       ▼
Evaluation LLM
       │
       ├── Correct / Incorrect
       ├── Topic
       └── Evaluation
```

The evaluation result is then stored in server-side interview state.

---

## 4. Feedback Generation

After the interview finishes:

```text
Interview State
      │
      ▼
Feedback LLM
      │
      ▼
Structured Report
```

The report summarizes the candidate's overall performance and highlights strong and weak areas.

---

# 📊 End-of-Interview Report

Example structure:

```text
Interview Performance Report

Total Questions: 10
Correct: 7
Incorrect: 3

Strong Topics:
- Probability
- Machine Learning
- Statistics

Weak Topics:
- SQL
- Feature Engineering

Overall Summary:
The candidate demonstrates a good understanding of core
Data Science concepts but needs more practice with SQL
and feature engineering.
```

---

# ⚠️ Known Limitations

### 1. Hint Leakage

The hint-generation system relies primarily on prompt-based instructions to prevent the LLM from revealing answers.

During testing, the model occasionally reveals partial definitions or important keywords despite explicit instructions.

This is a known limitation of **prompt-only guardrails**, particularly with smaller open-source models.

### Planned Improvement

Implement a code-level post-processing layer that validates generated hints before returning them to the candidate.

---

### 2. Approximate Question Counting

Currently, every candidate message can contribute toward the question limit.

For example:

```text
Candidate Answer
      ↓
Hint Request
      ↓
Hint Retry
```

These interactions may affect the question count even though they do not represent completely new interview questions.

A future version will track **distinct questions** instead.

---

### 3. Session Persistence

The frontend stores the `session_id` using Streamlit session state.

Therefore:

```text
Browser Refresh
      ↓
New Streamlit Session
      ↓
New session_id
```

The previous Redis session may still exist until its **1-hour TTL** expires, but the refreshed browser does not automatically reconnect to it.

### Planned Improvement

Add authentication and persistent user/session management.

---

# 🔮 Future Improvements

* [ ] Authentication and user accounts
* [ ] Persistent interview history
* [ ] Resume-based question generation
* [ ] Difficulty levels: Beginner / Intermediate / Advanced
* [ ] More comprehensive question bank
* [ ] Better hint validation
* [ ] Precise distinct-question tracking
* [ ] Interview analytics dashboard
* [ ] Voice-based interviews
* [ ] Speech-to-text integration
* [ ] Text-to-speech interviewer
* [ ] Personalized interview difficulty
* [ ] Multiple interview modes
* [ ] Deployment to a cloud platform

---

# 🎯 Learning Outcomes

This project was built as a hands-on way to learn and apply:

* FastAPI
* REST APIs
* Pydantic
* LLM integration
* Groq API
* OpenAI-compatible SDKs
* Retrieval-Augmented Generation (RAG)
* Vector databases
* ChromaDB
* Sentence Transformers
* Embeddings
* Redis
* Conversation memory
* Structured LLM evaluation
* Prompt engineering
* Streamlit
* Docker
* Backend state management
* Error handling

---

